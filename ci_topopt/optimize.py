"""Path objectives and MMA design updates."""

import csv
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io

from .config import loading_path, validate_config
from .fem import NonlinearMechanism
from .geometry import passive_masks
from .mma import mma_optimizer
from .parameterize import DensityMap


def root_io(comm, action):
    result, error = None, None
    if comm.rank == 0:
        try:
            result = action()
        except OSError as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)
    return result


class PathProblem:
    def __init__(self, config, comm=MPI.COMM_WORLD):
        validate_config(config)
        self.config = config
        self.model = NonlinearMechanism(config, comm)
        self.comm = comm
        self.density = DensityMap(self.model.rho, self.model.rho_phys,
                                  config["optimization"]["filter_radius"])
        self.targets, self.weights = loading_path(config)
        self.coefficients = [fem.Constant(self.model.mesh, PETSc.ScalarType(0)) for _ in range(2)]
        self.loss_form = sum(c * f for c, f in zip(self.coefficients, self.model.output_forms))
        self.n = self.model.rho.function_space.dofmap.index_map.size_local
        coords = self.model.rho.function_space.tabulate_dof_coordinates()[:self.n]
        self.solid, self.void = passive_masks(coords, config["geometry"])
        self.free = ~(self.solid | self.void)
        if comm.allreduce(int(self.free.sum()), op=MPI.SUM) == 0:
            raise ValueError("The geometry leaves no free design cells.")
        initial = np.full(self.n, config["optimization"]["initial_density"])
        self.set_design(initial)

    def set_design(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n,) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("Design must contain one density in [0, 1] per owned cell.")
        self.model.rho.x.array[:self.n] = values
        self.model.rho.x.array[:self.n][self.solid] = 1
        self.model.rho.x.array[:self.n][self.void] = 0
        self.model.rho.x.scatter_forward()

    def design(self):
        return self.model.rho.x.array[:self.n].copy()

    def gradient(self, functional):
        vector = self.model.functional_gradient(functional)
        try:
            return self.density.backward(vector)
        finally:
            vector.destroy()

    def evaluate(self, beta, gradients=True):
        self.density.forward(beta)
        opt, loading = self.config["optimization"], self.config["loading"]
        volume = self.model.scalar(self.model.volume_form)
        constraints = [volume / opt["volume_fraction"] - 1]
        derivatives = [self.gradient(self.model.volume_form) / opt["volume_fraction"]] if gradients else []
        objective = 0.0
        objective_gradient = np.zeros(self.n)
        samples = []
        stroke2 = loading["stroke"]**2

        def sample(target):
            nonlocal objective, objective_gradient
            index = len(samples)
            output = np.array([self.model.scalar(f) for f in self.model.output_forms])
            error = output + loading["gain"] * target
            loss = float(0.5 * np.dot(error, error) / stroke2)
            energy = self.model.scalar(self.model.energy_form)
            objective += self.weights[index] * loss
            constraints.append(energy / opt["energy_bound"] - 1)
            if gradients:
                for coefficient, value in zip(self.coefficients, self.weights[index] * error / stroke2):
                    coefficient.value = PETSc.ScalarType(value)
                objective_gradient += self.gradient(self.loss_form)
                derivatives.append(self.gradient(self.model.energy_form) / opt["energy_bound"])
            samples.append({"input": float(target), "output_a": float(output[0]),
                            "output_b": float(output[1]), "loss": loss,
                            "strain_energy": energy, "minimum_j": self.model.minimum_j()})

        diagnostics = self.model.solve_path(self.targets, sample)
        values = np.asarray(constraints)
        jacobian = np.asarray(derivatives) if gradients else None
        finite = np.isfinite(objective) and np.isfinite(volume) and np.all(np.isfinite(values))
        if gradients:
            finite = finite and np.all(np.isfinite(objective_gradient)) and np.all(np.isfinite(jacobian))
        if not self.comm.allreduce(bool(finite), op=MPI.LAND):
            raise RuntimeError("Nonfinite objective, constraint, or sensitivity.")
        return {"objective": objective, "volume": volume, "constraints": values,
                "gradient": objective_gradient if gradients else None, "jacobian": jacobian,
                "samples": samples, "diagnostics": diagnostics}

    def close(self):
        self.density.close()
        self.model.close()


def save_fields(problem, directory, label):
    model = problem.model
    space = fem.VectorFunctionSpace(model.mesh, ("Lagrange", 1))
    displacement = fem.Function(space, name="displacement")
    displacement.interpolate(model.u)
    model.rho.name, model.rho_phys.name = "design_density", "physical_density"
    with io.XDMFFile(problem.comm, str(directory / f"{label}.xdmf"), "w") as output:
        output.write_mesh(model.mesh)
        for field in (model.rho, model.rho_phys, displacement):
            output.write_function(field)


def optimize(config, comm=MPI.COMM_WORLD):
    validate_config(config)
    directory = Path(config["output"]["directory"]).expanduser().resolve()
    error = None
    if comm.rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {directory}")
            (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        except OSError as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    comm.barrier()
    problem = PathProblem(config, comm)
    opt = config["optimization"]
    x = problem.design()
    old1, old2 = x[problem.free].copy(), x[problem.free].copy()
    low = upp = None
    previous_objective, previous_beta = None, None
    last_feasible = None
    history = None
    columns = ["iteration", "beta", "objective", "volume_fraction", "max_constraint",
               "design_change", "minimum_j"]
    status = "iteration_limit"
    try:
        def open_history():
            stream = (directory / "history.csv").open("w", newline="")
            csv_writer = csv.DictWriter(stream, fieldnames=columns)
            csv_writer.writeheader()
            return stream, csv_writer

        handles = root_io(comm, open_history)
        if comm.rank == 0:
            history, writer = handles
        for iteration in range(1, opt["max_iter"] + 1):
            stage = min((iteration - 1) // opt["beta_interval"],
                        int(np.ceil(np.log2(opt["beta_max"] / opt["beta_init"]))))
            beta = min(opt["beta_max"], opt["beta_init"] * 2.0**stage)
            for retry in range(9):
                problem.set_design(x)
                try:
                    result = problem.evaluate(beta)
                    break
                except RuntimeError:
                    if last_feasible is None or retry == 8 or beta != previous_beta:
                        raise
                    x = 0.5 * (x + last_feasible)
                    if comm.rank == 0:
                        print(f"iteration={iteration} retry={retry + 1} reducing design update", flush=True)
            x = problem.design()
            current = x[problem.free]
            proposal, change, low, upp = mma_optimizer(
                len(result["constraints"]), current.size, iteration,
                current, np.zeros(current.size), np.ones(current.size), old1, old2,
                result["gradient"][problem.free], result["constraints"],
                result["jacobian"][:, problem.free], low, upp, move=opt["move"], comm=comm,
            )
            max_constraint = float(result["constraints"].max())
            minimum_j = min(row["minimum_j"] for row in result["samples"])
            row = dict(zip(columns, (iteration, beta, result["objective"], result["volume"],
                                     max_constraint, float(change), minimum_j)))
            def append_history():
                writer.writerow(row)
                history.flush()

            root_io(comm, append_history)
            if comm.rank == 0:
                print(f"iteration={iteration:04d} beta={beta:g} objective={result['objective']:.6g} "
                      f"volume={result['volume']:.4f} violation={max_constraint:.3e} "
                      f"change={change:.3e} minJ={minimum_j:.4g}", flush=True)
            if iteration == 1 or iteration % config["output"]["save_every"] == 0:
                save_fields(problem, directory, f"iteration_{iteration:04d}")
            stable_objective = (previous_objective is not None and beta == previous_beta
                                and abs(result["objective"] - previous_objective)
                                <= opt["objective_tolerance"] * max(1, abs(previous_objective)))
            if (beta == opt["beta_max"] and stable_objective and change <= opt["design_tolerance"]
                    and max_constraint <= opt["constraint_tolerance"]):
                status = "converged"
                break
            if iteration == opt["max_iter"]:
                break
            old2, old1 = old1.copy(), current.copy()
            previous_objective, previous_beta = result["objective"], beta
            last_feasible = x.copy()
            x[problem.free] = proposal
        save_fields(problem, directory, "final")
        summary = {"status": status, "iteration": iteration, "objective": result["objective"],
                   "volume_fraction": result["volume"], "constraints": result["constraints"].tolist(),
                   "beta": beta, "samples": result["samples"]}
        def write_results():
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            with (directory / "load_path.csv").open("w", newline="") as path_file:
                path_writer = csv.DictWriter(path_file, fieldnames=list(result["samples"][0]))
                path_writer.writeheader()
                path_writer.writerows(result["samples"])

        root_io(comm, write_results)
        return summary
    except Exception as exc:
        save_fields(problem, directory, "failed_design")
        root_io(comm, lambda: (directory / "failure.json").write_text(
            json.dumps({"error": str(exc)}, indent=2) + "\n"))
        raise
    finally:
        try:
            root_io(comm, lambda: history.close() if history is not None else None)
        finally:
            problem.close()

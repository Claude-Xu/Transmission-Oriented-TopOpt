"""Finite-strain mechanics with the HuHu third-medium regularization of Bluhm et al.

See https://doi.org/10.1007/s00466-021-01974-x.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc


class NonlinearConvergenceError(RuntimeError):
    pass


class NonlinearMechanism:
    def __init__(self, config: dict, comm=MPI.COMM_WORLD):
        self.config = config
        self.comm = comm
        geometry = config.get("geometry", {})
        material = config.get("material", {})
        newton = config.get("newton", {})
        loading = config.get("loading", {})
        huhu = config.get("huhu", {})
        self.length = float(geometry.get("length", 10.0))
        self.stroke = float(loading.get("stroke", 1.0))
        self.max_step = float(loading.get("max_step", newton.get("max_step", 0.025)))
        self.min_step = float(newton.get("min_step", 1e-5))
        self.min_j = float(newton.get("min_j", 1e-8))
        self.rtol = float(newton.get("rtol", 1e-7))
        self.atol = float(newton.get("atol", 1e-9))
        self.max_it = int(newton.get("max_it", 60))
        self.line_search_steps = int(newton.get("line_search_steps", 16))
        if not (self.length > 0 and self.max_step >= self.min_step > 0):
            raise ValueError("Require length > 0 and max_step >= min_step > 0.")
        if not (0 < self.min_j < 1 and self.rtol > 0 and self.atol > 0 and self.max_it > 0):
            raise ValueError("Invalid Newton tolerances or determinant threshold.")
        if self.stroke == 0:
            raise ValueError("The prescribed stroke must be nonzero.")
        nx, ny = int(geometry.get("nx", 100)), int(geometry.get("ny", 100))
        if nx < 2 or ny < 2:
            raise ValueError("The mesh requires at least two cells per direction.")
        self.mesh = mesh.create_rectangle(
            comm, [[0.0, 0.0], [self.length, self.length]],
            [nx, ny], cell_type=mesh.CellType.quadrilateral,
        )
        self.V = fem.functionspace(self.mesh, ("Lagrange", 2, (2,)))
        self.Q0 = fem.functionspace(self.mesh, ("DG", 0))
        self.Q1 = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.rho = fem.Function(self.Q0, name="design_density")
        self.rho_phys = fem.Function(self.Q1, name="physical_density")
        self.u = fem.Function(self.V, name="displacement")
        self._adjoint = fem.Function(self.V)
        self.rho.x.array[:] = 0.5
        self.rho_phys.x.array[:] = 0.5

        points = np.array([0.0, 0.5, 1.0])
        weights = np.array([1.0, 4.0, 1.0]) / 6.0
        self.quadrature_points = np.array([(x, y) for x in points for y in points])
        quadrature_weights = np.array([x * y for x in weights for y in weights])
        self.dx = ufl.Measure("dx", domain=self.mesh, metadata={
            "quadrature_rule": "custom",
            "quadrature_points": self.quadrature_points.tolist(),
            "quadrature_weights": quadrature_weights.tolist(),
        })
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        boundary = mesh.exterior_facet_indices(self.mesh.topology)
        midpoints = mesh.compute_midpoints(self.mesh, fdim, boundary)
        center = self.length / 2
        half_width = float(geometry.get("port_half_width", 1.0))
        sub_width = float(geometry.get("subport_width", 0.35))
        if not (0 < sub_width <= half_width < center):
            raise ValueError("Require 0 < subport_width <= port_half_width < length/2.")
        input_mask = np.isclose(midpoints[:, 0], 0) & (np.abs(midpoints[:, 1] - center) < half_width)
        right = np.isclose(midpoints[:, 0], self.length)
        top_mask = right & (midpoints[:, 1] > center + half_width - sub_width) & (midpoints[:, 1] < center + half_width)
        bottom_mask = right & (midpoints[:, 1] > center - half_width) & (midpoints[:, 1] < center - half_width + sub_width)
        all_facets = np.concatenate([boundary[m] for m in (input_mask, top_mask, bottom_mask)]).astype(np.int32)
        all_values = np.concatenate([np.full(np.count_nonzero(m), i, dtype=np.int32) for i, m in enumerate((input_mask, top_mask, bottom_mask), 1)])
        order = np.argsort(all_facets)
        self.facet_tags = mesh.meshtags(self.mesh, fdim, all_facets[order], all_values[order])
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags, metadata={"quadrature_degree": 5})
        one = fem.Constant(self.mesh, PETSc.ScalarType(1.0))
        measures = [self.scalar(one * self.ds(i)) for i in (1, 2, 3)]
        if min(measures) <= 0:
            raise ValueError("An input or output port is empty; refine the mesh or increase port widths.")
        self.output_forms = tuple(self.u[0] / measures[i - 1] * self.ds(i) for i in (2, 3))

        corner = float(geometry.get("corner_size", 0.4))
        if not (0 < corner < center - half_width):
            raise ValueError("Invalid corner clamp size.")

        def corner_locator(x):
            left_distance = np.minimum(x[0], self.length - x[0])
            bottom_distance = np.minimum(x[1], self.length - x[1])
            return left_distance + bottom_distance <= corner + 1e-12

        fixed_dofs = fem.locate_dofs_geometrical(self.V, corner_locator)
        fixed = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), fixed_dofs, self.V)
        input_dofs = fem.locate_dofs_topological(self.V.sub(0), fdim, boundary[input_mask])
        self.input_displacement = fem.Constant(self.mesh, PETSc.ScalarType(0.0))
        prescribed = fem.dirichletbc(self.input_displacement, input_dofs, self.V.sub(0))
        self.bcs = [fixed, prescribed]

        young = float(material.get("young", 40.0))
        poisson = float(material.get("poisson", 0.45))
        penalty = float(material.get("penalty", 3.0))
        epsilon = float(material.get("epsilon", 1e-6))
        alpha = float(huhu.get("alpha", 2e-6))
        if not (young > 0 and -1 < poisson < 0.5 and penalty >= 1 and 0 < epsilon < 1 and alpha > 0):
            raise ValueError("Invalid material or HuHu parameters.")
        mu = young / (2 * (1 + poisson))
        lmbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        bulk = young / (3 * (1 - 2 * poisson))
        F = ufl.Identity(2) + ufl.grad(self.u)
        self.jacobian_expression = ufl.det(F)
        J = self.jacobian_expression
        psi = mu / 2 * (ufl.inner(F, F) - 2) - mu * ufl.ln(J) + lmbda / 2 * ufl.ln(J) ** 2
        density_scale = epsilon + (1 - epsilon) * self.rho_phys**penalty
        self.energy_form = density_scale * psi * self.dx
        self.volume_form = self.rho_phys / self.length**2 * self.dx
        test, trial = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        hessian = lambda field: ufl.grad(ufl.grad(field))
        regularization = alpha * self.length**2 * bulk * ufl.exp(-5 * J) * ufl.inner(hessian(test), hessian(self.u)) * self.dx
        self.residual = ufl.derivative(self.energy_form, self.u, test) + regularization
        self.tangent = ufl.derivative(self.residual, self.u, trial)
        self._residual_form = fem.form(self.residual)
        self._tangent_form = fem.form(self.tangent)
        self._matrix = fem_petsc.create_matrix(self._tangent_form)
        self._rhs = fem_petsc.create_vector(self._residual_form)
        self._increment = self.u.vector.duplicate()
        self._solver = PETSc.KSP().create(comm)
        self._solver.setType("preonly")
        self._solver.getPC().setType("lu")
        self._solver.getPC().setFactorSolverType(newton.get("linear_solver", "mumps"))
        self._solver.setErrorIfNotConverged(True)
        self._determinants = fem.Expression(J, self.quadrature_points)
        self._cells = np.arange(self.mesh.topology.index_map(self.mesh.topology.dim).size_local, dtype=np.int32)
        self._functional_cache = {}
        self.last_path_diagnostics = []
        self._closed = False

    def scalar(self, functional) -> float:
        value = fem.assemble_scalar(fem.form(functional))
        return float(self.comm.allreduce(value, op=MPI.SUM))

    def minimum_j(self) -> float:
        values = self._determinants.eval(self.mesh, self._cells)
        local_minimum = float(np.min(values)) if values.size else np.inf
        if not np.isfinite(local_minimum):
            local_minimum = -np.inf
        return float(self.comm.allreduce(local_minimum, op=MPI.MIN))

    def _assemble_residual(self) -> float:
        with self._rhs.localForm() as local:
            local.set(0)
        fem_petsc.assemble_vector(self._rhs, self._residual_form)
        self._rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(self._rhs, self.bcs, x0=self.u.vector)
        return float(self._rhs.norm())

    def _assemble_tangent(self):
        self._matrix.zeroEntries()
        fem_petsc.assemble_matrix(self._matrix, self._tangent_form, bcs=self.bcs)
        self._matrix.assemble()
        self._solver.setOperators(self._matrix)

    def _solve_increment(self, target: float) -> tuple[bool, int]:
        self.input_displacement.value = PETSc.ScalarType(target)
        fem_petsc.set_bc(self.u.vector, self.bcs)
        self.u.x.scatter_forward()
        if self.minimum_j() <= self.min_j:
            return False, 0
        norm = self._assemble_residual()
        reference = norm
        tolerance = self.atol + self.rtol * reference
        for iteration in range(self.max_it + 1):
            if np.isfinite(norm) and norm <= tolerance:
                return True, iteration
            if iteration == self.max_it or not np.isfinite(norm):
                return False, iteration
            self._assemble_tangent()
            self._rhs.scale(-1)
            try:
                self._solver.solve(self._rhs, self._increment)
            except PETSc.Error:
                return False, iteration + 1
            self._increment.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            old = self.u.x.array.copy()
            with self._increment.localForm() as local:
                increment = local.array.copy()
            accepted = False
            scale = 1.0
            for _ in range(self.line_search_steps):
                self.u.x.array[:] = old + scale * increment
                self.u.x.scatter_forward()
                if self.minimum_j() > self.min_j:
                    trial_norm = self._assemble_residual()
                    if np.isfinite(trial_norm) and (trial_norm <= (1 - 1e-4 * scale) * norm or trial_norm <= tolerance):
                        norm = trial_norm
                        accepted = True
                        break
                scale *= 0.5
            if not accepted:
                self.u.x.array[:] = old
                self.u.x.scatter_forward()
                return False, iteration + 1
        return False, self.max_it

    def solve_path(self, samples: Sequence[float], callback: Callable[[float], object] | None = None):
        targets = np.asarray(samples, dtype=float)
        if targets.ndim != 1 or not targets.size or not np.all(np.isfinite(targets)):
            raise ValueError("The loading path must be a finite, nonempty one-dimensional sequence.")
        nonzero = targets[targets != 0]
        if nonzero.size and np.any(np.sign(nonzero) != np.sign(nonzero[0])):
            raise ValueError("All loading samples must have the same sign.")
        if np.any(np.diff(np.abs(targets)) <= 0):
            raise ValueError("Sample magnitudes must be strictly increasing.")
        self.u.x.array[:] = 0
        self.u.x.scatter_forward()
        self.input_displacement.value = PETSc.ScalarType(0)
        self.last_path_diagnostics = []
        current = 0.0
        step = self.max_step
        sample_iterations = 0
        sample_steps = 0
        for target in targets:
            target = float(target)
            while abs(target - current) > 1e-12 * max(1, abs(target)):
                increment = np.sign(target - current) * min(step, abs(target - current))
                trial_target = current + increment
                old = self.u.x.array.copy()
                converged, iterations = self._solve_increment(trial_target)
                sample_iterations += iterations
                if not converged:
                    self.u.x.array[:] = old
                    self.u.x.scatter_forward()
                    self.input_displacement.value = PETSc.ScalarType(current)
                    step = abs(increment) / 2
                    if step < self.min_step:
                        raise NonlinearConvergenceError(
                            f"Nonlinear equilibrium failed between {current:.8g} and {trial_target:.8g}; "
                            f"the minimum step {self.min_step:g} was reached."
                        )
                    continue
                current = trial_target
                sample_steps += 1
                step = min(self.max_step, 1.5 * abs(increment))
            diagnostics = {
                "target": target,
                "newton_iterations": sample_iterations,
                "substeps": sample_steps,
                "minimum_j": self.minimum_j(),
            }
            if callback is not None:
                diagnostics["callback_result"] = callback(target)
            self.last_path_diagnostics.append(diagnostics)
            sample_iterations = sample_steps = 0
        return self.last_path_diagnostics

    def functional_gradient(self, functional) -> PETSc.Vec:
        if functional not in self._functional_cache:
            du = ufl.derivative(functional, self.u, ufl.TestFunction(self.V))
            drho = ufl.derivative(
                functional - ufl.action(self.residual, self._adjoint),
                self.rho_phys, ufl.TestFunction(self.Q1),
            )
            state_form = fem.form(du) if du.integrals() else None
            density_form = fem.form(drho)
            self._functional_cache[functional] = state_form, density_form
        state_form, density_form = self._functional_cache[functional]
        self._adjoint.x.array[:] = 0
        if state_form is not None:
            with self._rhs.localForm() as local:
                local.set(0)
            fem_petsc.assemble_vector(self._rhs, state_form)
            self._rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem_petsc.set_bc(self._rhs, self.bcs, scale=0)
            self._assemble_tangent()
            self._solver.solveTranspose(self._rhs, self._adjoint.vector)
        self._adjoint.x.scatter_forward()
        gradient = fem_petsc.create_vector(density_form)
        with gradient.localForm() as local:
            local.set(0)
        fem_petsc.assemble_vector(gradient, density_form)
        gradient.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        gradient.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return gradient

    def close(self):
        if self._closed:
            return
        self._solver.destroy()
        self._matrix.destroy()
        self._rhs.destroy()
        self._increment.destroy()
        self._functional_cache.clear()
        self._closed = True

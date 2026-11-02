import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from ci_topopt.fem import NonlinearMechanism
from ci_topopt.parameterize import DensityMap


@pytest.fixture
def mechanism():
    config = {
        "geometry": {
            "nx": 8, "ny": 10, "length": 10.0,
            "corner_size": 0.4, "port_half_width": 1.0,
            "subport_width": 1.0,
        },
        "material": {"young": 40.0, "poisson": 0.45, "penalty": 3.0, "epsilon": 1e-6},
        "huhu": {"alpha": 2e-6},
        "loading": {"stroke": 0.2},
        "newton": {
            "rtol": 1e-10, "atol": 1e-11, "max_it": 60,
            "max_step": 0.05, "min_step": 1e-5, "min_j": 1e-8,
        },
    }
    problem = NonlinearMechanism(config)
    problem.rho_phys.x.array[:] = 0.65
    problem.rho_phys.x.scatter_forward()
    yield problem
    problem.close()


def _assemble_residual(problem):
    vector = assemble_vector(fem.form(problem.residual))
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vector


def test_loading_path_reaches_every_sample(mechanism):
    requested = np.array([0.05, 0.11, 0.2])
    observed = []

    def record(target):
        observed.append((target, mechanism.minimum_j(), mechanism.scalar(mechanism.output_forms[0])))

    mechanism.solve_path(requested, record)
    np.testing.assert_allclose([entry[0] for entry in observed], requested, rtol=0, atol=1e-14)
    assert all(entry[1] > 1e-8 for entry in observed)
    assert abs(observed[-1][2] - observed[0][2]) > 1e-5
    final = mechanism.u.x.array.copy()
    mechanism.solve_path(requested, lambda target: None)
    np.testing.assert_allclose(mechanism.u.x.array, final, rtol=1e-7, atol=1e-9)


def test_residual_tangent_matches_directional_difference(mechanism):
    mechanism.solve_path([0.2], lambda target: None)
    direction = fem.Function(mechanism.V)
    direction.interpolate(lambda x: np.vstack((
        0.03 * np.sin(np.pi*x[0]/10)*np.sin(np.pi*x[1]/10),
        0.02 * np.sin(2*np.pi*x[0]/10)*np.sin(np.pi*x[1]/10),
    )))
    direction.x.scatter_forward()
    tangent = assemble_matrix(fem.form(mechanism.tangent))
    tangent.assemble()
    predicted = direction.vector.duplicate()
    tangent.mult(direction.vector, predicted)
    saved = mechanism.u.x.array.copy()
    step = 1e-5
    mechanism.u.x.array[:] = saved + step*direction.x.array
    mechanism.u.x.scatter_forward()
    plus = _assemble_residual(mechanism)
    mechanism.u.x.array[:] = saved - step*direction.x.array
    mechanism.u.x.scatter_forward()
    minus = _assemble_residual(mechanism)
    mechanism.u.x.array[:] = saved
    mechanism.u.x.scatter_forward()
    plus.axpy(-1, minus)
    plus.scale(1/(2*step))
    plus.axpy(-1, predicted)
    assert plus.norm() / predicted.norm() < 2e-6
    for item in (tangent, predicted, plus, minus):
        item.destroy()


@pytest.mark.parametrize("functional_name", ["output", "energy"])
def test_density_adjoint_matches_finite_difference(mechanism, functional_name):
    density_map = DensityMap(mechanism.rho, mechanism.rho_phys, 0.4)
    coordinates = mechanism.rho.function_space.tabulate_dof_coordinates()
    count = mechanism.rho.function_space.dofmap.index_map.size_local
    base = 0.6 + 0.05*np.sin(coordinates[:, 0])*np.sin(coordinates[:, 1])
    direction = 0.15*np.cos(coordinates[:, 0]/3)*np.sin(coordinates[:, 1]/4)
    functional = mechanism.output_forms[0] if functional_name == "output" else mechanism.energy_form

    def evaluate(shift):
        mechanism.rho.x.array[:] = base + shift*direction
        mechanism.rho.x.scatter_forward()
        density_map.forward(2.0)
        mechanism.solve_path([0.2], lambda target: None)
        return mechanism.scalar(functional)

    evaluate(0)
    physical_gradient = mechanism.functional_gradient(functional)
    gradient = density_map.backward(physical_gradient)
    predicted = MPI.COMM_WORLD.allreduce(float(gradient @ direction[:count]), op=MPI.SUM)
    step = 2e-5
    observed = (evaluate(step) - evaluate(-step))/(2*step)
    assert abs(predicted-observed) < 3e-5*max(abs(predicted), abs(observed), 1e-6)
    physical_gradient.destroy()
    density_map.close()


def test_huhu_penalizes_nonaffine_deformation(mechanism):
    mechanism.u.interpolate(lambda x: np.vstack((0.01*x[0]**2, 0.02*x[1]**2)))
    mechanism.u.x.scatter_forward()
    direction = ufl.TestFunction(mechanism.V)
    hessian = ufl.grad(ufl.grad(mechanism.u))
    magnitude = mechanism.scalar(ufl.inner(hessian, hessian)*ufl.dx)
    assert magnitude > 0.01
    full = _assemble_residual(mechanism)
    linear = fem.form(ufl.derivative(mechanism.energy_form, mechanism.u, direction))
    without_huhu = assemble_vector(linear)
    without_huhu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    full.axpy(-1, without_huhu)
    assert full.norm() > 1e-7
    full.destroy()
    without_huhu.destroy()


def test_inverted_state_is_not_hidden_by_log_regularization(mechanism):
    mechanism.u.interpolate(lambda x: np.vstack((-2.0*x[0], np.zeros_like(x[1]))))
    mechanism.u.x.scatter_forward()
    assert mechanism.minimum_j() == pytest.approx(-1.0, abs=1e-12)


def test_soft_medium_resists_near_collapse(mechanism):
    mechanism.rho_phys.x.array[:] = 0.01
    mechanism.rho_phys.x.scatter_forward()
    energy = []
    for stretch in (0.1, 1e-4):
        mechanism.u.interpolate(lambda x: np.vstack(((stretch-1)*x[0], np.zeros_like(x[1]))))
        mechanism.u.x.scatter_forward()
        energy.append(mechanism.scalar(mechanism.energy_form))
        assert mechanism.minimum_j() == pytest.approx(stretch, rel=1e-10, abs=1e-12)
    assert energy[1] > 4*energy[0] > 0


def test_full_path_objective_and_constraints_gradients():
    from ci_topopt.config import load_config
    from ci_topopt.optimize import PathProblem

    config = load_config()
    config["geometry"].update(nx=10, ny=10, subport_width=1.0, passive_frame=False)
    config["loading"].update(stroke=0.2, fractions=[0.3, 0.6, 1.0], weights=[0.2, 0.3, 0.5])
    config["newton"].update(rtol=1e-10, atol=1e-11, max_step=0.05)
    config["optimization"]["filter_radius"] = 0.4
    problem = PathProblem(config)
    try:
        coordinates = problem.model.rho.function_space.tabulate_dof_coordinates()[:problem.n]
        base = 0.6 + 0.03*np.sin(coordinates[:, 0])*np.cos(coordinates[:, 1])
        direction = 0.1*np.cos(coordinates[:, 0]/3)*np.sin(coordinates[:, 1]/4)
        direction[~problem.free] = 0
        problem.set_design(base)
        result = problem.evaluate(2.0)
        predicted = np.append(result["gradient"] @ direction, result["jacobian"] @ direction)
        predicted = problem.comm.allreduce(predicted, op=MPI.SUM)
        assert len(result["samples"]) == 3
        np.testing.assert_allclose([sample["input"] for sample in result["samples"]], [0.06, 0.12, 0.2])
        expected = np.dot([0.2, 0.3, 0.5], [sample["loss"] for sample in result["samples"]])
        assert result["objective"] == pytest.approx(expected, rel=1e-12)
        assert abs(result["objective"]-result["samples"][-1]["loss"]) > 1e-3
        values = []
        step = 2e-5
        for shift in (step, -step):
            problem.set_design(base + shift*direction)
            observed = problem.evaluate(2.0, gradients=False)
            values.append(np.append(observed["objective"], observed["constraints"]))
        observed = (values[0] - values[1])/(2*step)
        np.testing.assert_allclose(predicted, observed, rtol=4e-5, atol=2e-9)
    finally:
        problem.close()


def test_failed_increment_restores_last_equilibrium(mechanism):
    from ci_topopt.fem import NonlinearConvergenceError

    mechanism.max_it = 1
    mechanism.rtol = mechanism.atol = 1e-14
    mechanism.max_step = mechanism.min_step = 0.2
    observed = []
    with pytest.raises(NonlinearConvergenceError, match="minimum step"):
        mechanism.solve_path([0.2], observed.append)
    np.testing.assert_array_equal(mechanism.u.x.array, np.zeros_like(mechanism.u.x.array))
    assert float(mechanism.input_displacement.value) == 0.0
    assert mechanism.minimum_j() == pytest.approx(1.0, abs=1e-12)
    assert observed == []
    assert mechanism.last_path_diagnostics == []


def test_negative_loading_path_keeps_signed_targets(mechanism):
    requested = [-0.05, -0.12, -0.2]
    observed = []
    mechanism.solve_path(requested, observed.append)
    np.testing.assert_allclose(observed, requested, rtol=0, atol=1e-14)
    assert float(mechanism.input_displacement.value) == pytest.approx(-0.2)
    assert mechanism.minimum_j() > mechanism.min_j
    assert mechanism.scalar(mechanism.output_forms[0]) < -1e-3

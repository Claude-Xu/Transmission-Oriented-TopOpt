import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from ci_topopt.mma import mma_optimizer
from ci_topopt.parameterize import DensityMap


@pytest.mark.parametrize("root_only", [False, True])
def test_mma_constrained_quadratic(root_only):
    comm = MPI.COMM_WORLD
    count = 2 if not root_only or comm.rank == 0 else 0
    total = comm.allreduce(count, op=MPI.SUM)
    current = np.full(count, 0.3)
    old1, old2 = current.copy(), current.copy()
    low = upper = None
    for iteration in range(1, 31):
        constraint = comm.allreduce(float(current.sum()), op=MPI.SUM) / total - 0.5
        proposal, change, low, upper = mma_optimizer(
            1, count, iteration, current, np.zeros(count), np.ones(count), old1, old2,
            2 * (current - 0.8) / total, np.array([constraint]),
            np.full((1, count), 1 / total), low, upper, move=0.1,
        )
        assert np.isfinite(change)
        assert np.all((proposal >= 0) & (proposal <= 1))
        old2, old1, current = old1.copy(), current.copy(), proposal
    error = comm.allreduce(float(np.max(np.abs(current - 0.5), initial=0)), op=MPI.MAX)
    final_constraint = comm.allreduce(float(current.sum()), op=MPI.SUM) / total - 0.5
    assert error < 2e-4
    assert final_constraint <= 1e-6


def test_density_filter_adjoint_and_constant_field():
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, 6, 6, cell_type=mesh.CellType.quadrilateral)
    design_space = fem.FunctionSpace(domain, ("DG", 0))
    physical_space = fem.FunctionSpace(domain, ("Lagrange", 1))
    design, physical = fem.Function(design_space), fem.Function(physical_space)
    coordinates = design_space.tabulate_dof_coordinates()
    design.x.array[:] = 0.45 + 0.1 * np.sin(3 * coordinates[:, 0]) * np.cos(2 * coordinates[:, 1])
    count = design_space.dofmap.index_map.size_local
    direction = 0.3 * np.cos(5 * coordinates[:count, 0]) * np.sin(3 * coordinates[:count, 1])
    density = DensityMap(design, physical, 0.12)
    functional = (physical**2 + 0.3 * physical) * ufl.dx
    value_form = fem.form(functional)
    derivative_form = fem.form(ufl.derivative(functional, physical, ufl.TestFunction(physical_space)))
    gradient = None
    try:
        density.forward(2)
        gradient = fem_petsc.assemble_vector(derivative_form)
        gradient.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        predicted = comm.allreduce(float(density.backward(gradient) @ direction), op=MPI.SUM)
        saved = design.x.array[:count].copy()
        values = []
        step = 1e-6
        for shift in (step, -step):
            design.x.array[:count] = saved + shift * direction
            density.forward(2)
            values.append(comm.allreduce(fem.assemble_scalar(value_form), op=MPI.SUM))
        observed = (values[0] - values[1]) / (2 * step)
        assert predicted == pytest.approx(observed, rel=1e-7, abs=1e-9)
        design.x.array[:count] = 0.4
        density.forward(2)
        expected = (np.tanh(1) + np.tanh(-0.2)) / (2 * np.tanh(1))
        np.testing.assert_allclose(physical.x.array, expected, atol=1e-12, rtol=0)
    finally:
        if gradient is not None:
            gradient.destroy()
        density.close()


def test_mma_on_independent_communicators():
    limit = 0.35 + 0.05 * MPI.COMM_WORLD.rank
    current = np.array([0.2])
    old1, old2 = current.copy(), current.copy()
    low = upper = None
    for iteration in range(1, 26):
        proposal, _, low, upper = mma_optimizer(
            1, 1, iteration, current, np.zeros(1), np.ones(1), old1, old2,
            2 * (current - 0.8), current - limit, np.ones((1, 1)),
            low, upper, move=0.1, comm=MPI.COMM_SELF,
        )
        old2, old1, current = old1.copy(), current.copy(), proposal
    assert current[0] == pytest.approx(limit, abs=2e-4)

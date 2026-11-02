"""Helmholtz filtering and density projection, adapted from FEniTop (GPL-3.0)."""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc


class DensityMap:
    def __init__(self, rho, rho_phys, radius):
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("The filter radius must be positive.")
        self.rho = rho
        self.rho_phys = rho_phys
        self.filtered = fem.Function(rho_phys.function_space)
        space = rho_phys.function_space
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        design_trial = ufl.TrialFunction(rho.function_space)
        dx = ufl.Measure("dx", domain=space.mesh, metadata={"quadrature_degree": 2})
        lhs = (radius**2 * ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * dx
        self.matrix = fem_petsc.assemble_matrix(fem.form(lhs))
        self.matrix.assemble()
        self.transfer = fem_petsc.assemble_matrix(fem.form(design_trial * test * dx))
        self.transfer.assemble()
        self.solver = PETSc.KSP().create(space.mesh.comm)
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self.matrix)
        self.rhs = self.transfer.createVecLeft()
        self.adjoint = self.matrix.createVecRight()
        self.design_gradient = self.transfer.createVecRight()
        self.derivative = None
        self._closed = False

    def forward(self, beta):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("The projection parameter must be positive.")
        self.rho.x.scatter_forward()
        self.transfer.mult(self.rho.vector, self.rhs)
        self.solver.solve(self.rhs, self.filtered.vector)
        self._check_solve()
        self.filtered.x.scatter_forward()
        self._check_range(self.filtered.x.array)
        values = np.tanh(beta * (self.filtered.x.array - 0.5))
        denominator = 2.0 * np.tanh(0.5 * beta)
        self.derivative = beta * (1.0 - values**2) / denominator
        self.rho_phys.x.array[:] = (np.tanh(0.5 * beta) + values) / denominator
        self.rho_phys.x.scatter_forward()
        self._check_range(self.rho_phys.x.array)

    def backward(self, gradient):
        if self.derivative is None:
            raise RuntimeError("Call forward before backward.")
        owned = self.rhs.getLocalSize()
        self.rhs.array[:] = gradient.array * self.derivative[:owned]
        self.solver.solveTranspose(self.rhs, self.adjoint)
        self._check_solve()
        self.transfer.multTranspose(self.adjoint, self.design_gradient)
        return self.design_gradient.array.copy()

    def _check_solve(self):
        if self.solver.getConvergedReason() <= 0:
            raise RuntimeError("The density filter solve did not converge.")

    def _check_range(self, values):
        valid = np.all(np.isfinite(values)) and np.all((values >= -1e-10) & (values <= 1 + 1e-10))
        if not self.rho.function_space.mesh.comm.allreduce(bool(valid), op=MPI.LAND):
            raise RuntimeError(
                "The Helmholtz filter produced densities outside [0, 1]; "
                "increase optimization.filter_radius or refine the mesh."
            )

    def close(self):
        if self._closed:
            return
        for obj in (self.solver, self.rhs, self.adjoint, self.design_gradient,
                    self.matrix, self.transfer):
            obj.destroy()
        self._closed = True

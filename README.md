# Nonlinear Compliant Inverter Topology Optimization

A FEniCSx implementation for designing a compliant inverter (CI) under finite displacement. The formulation uses plane-strain Neo-Hookean elasticity, SIMP interpolation, HuHu regularization, a Helmholtz density filter, Heaviside projection, and parallel MMA.

## Run

Use the supplied environment, which targets DOLFINx 0.7.3 and real-valued PETSc with MUMPS:

```bash
conda env create -f environment.yml
conda activate ci-topopt
python -m ci_topopt --config configs/inverter.json
```

Parallel execution and a short check:

```bash
mpiexec -n 8 python -m ci_topopt --config configs/inverter.json --output results/inverter-mpi
python scripts/TransmissionOrientedModularDesign_2d.py --config configs/smoke.json
python -m ci_topopt --config configs/coarse.json
python -m pytest -q
mpiexec -n 2 python -m pytest -q
```

Run commands from the repository root. Each run requires an empty output directory. An existing FEniCSx 0.7.3 environment can be used directly; DOLFINx and PETSc must be installed together through FEniCSx or conda rather than through this package's pip dependencies.

## Formulation

The reference domain is a 10 × 10 square with the original curved passive frame, corner supports, a prescribed horizontal input on the left edge, and two output measurement regions on the right edge. Free design cells start at a uniform density of 0.5. Passive solid and void cells have fixed design densities of 1 and 0; the filter smooths their interfaces. No external density or mask file is required.

At each optimization iteration, one displacement continuation starts from the undeformed state and visits every configured sampling point. The default input displacements are 0.25, 0.5, 0.75, and 1.0. Adaptive increment subdivision and Newton backtracking retain the prescribed sample positions and final stroke.

For sample weights summing to one, the objective is

\[
f = \sum_s w_s\frac{(\bar u_{A,s}+g d_s)^2+(\bar u_{B,s}+g d_s)^2}{2d_{\max}^2}.
\]

Here \(d_s\) is the imposed input, \(\bar u_A,\bar u_B\) are boundary-averaged horizontal output displacements, and \(g=1\) targets equal and opposite motion. The constraints bound physical volume fraction and Neo-Hookean strain energy at **every** sample. These are displacement matching and energy constraints; positive definiteness of the structural tangent is not imposed.

The material uses the true deformation gradient \(F=I+\nabla u\) throughout the solid and soft ersatz medium, with stiffness scale \(\epsilon+(1-\epsilon)\rho^p\). HuHu adds

\[
R_H(u;v)=\alpha L^2K\int_\Omega e^{-5\det F}\,\nabla^2v:\nabla^2u\,d\Omega.
\]

Q2 quadrilateral displacement elements and a 3 × 3 Gauss–Lobatto rule follow the reference implementation. The complete residual is differentiated for the Newton tangent, and the adjoint uses its transpose, including the displacement dependence of the HuHu gate. Projection and filter derivatives are included in every design gradient.

The compressible soft medium transmits forces when strongly compressed; HuHu regularizes its distortion. This is a third-medium contact approximation with finite pre-contact stiffness, rather than surface gap constraints or friction. Accepted states must have positive \(\det F\) at the integration points. Contact accuracy and mesh dependence should be assessed for the selected design, ersatz stiffness, and regularization parameters. The included tests verify the discrete solver and sensitivities; they do not establish contact-accuracy convergence.

## Configuration and results

`configs/inverter.json` contains the full settings. `loading.fractions` must contain at least two increasing positive values ending at 1. Optional `loading.weights` must be positive and match the sampling points. `optimization.initial_density` sets the uniform free-cell initial guess. Output facets are selected by their midpoints; refine the mesh to resolve the configured port widths and passive frame.

The output directory contains the resolved configuration, `history.csv`, the final sampled `load_path.csv`, `summary.json`, and XDMF/HDF5 density and displacement fields. Displacement is interpolated to CG1 for visualization; nonlinear analysis uses Q2. The final files contain the last evaluated design. The summary distinguishes convergence from reaching the iteration limit. A failed analysis writes the current design and a failure record, then stops; it does not substitute a shorter loading path.

## Validation

The 14 tests pass in serial and with two MPI ranks on DOLFINx 0.7.3. They check nonlinear tangents, transpose adjoints, the complete sampled objective and constraint gradients, failed-step rollback, signed loading, filtering, and MMA. Serial and parallel smoke optimizations agree. The default 100 × 100 mesh also completes one design evaluation through all four samples at full stroke.

The 60-iteration `configs/coarse.json` run reduces the objective from 0.831231 to 0.000150757 with final volume fraction 0.499000. At input displacement 1.0, the two outputs are −0.990471 and −0.976972. All sampled energy constraints are satisfied, and the minimum sampled determinant is 0.358729. This run ends at its iteration limit.

## References and license

- [FEniTop](https://github.com/missionlab/fenitop), Jia, Wang and Zhang, [2024](https://doi.org/10.1007/s00158-024-03818-7). The filter and parallel MMA are adapted from this implementation.
- Bluhm, Sigmund and Poulios, [Internal contact modeling for finite strain topology optimization](https://doi.org/10.1007/s00466-021-01974-x), 2021.
- Frederiksen et al., [Improved third medium formulation for 3D topology optimization with contact](https://doi.org/10.1016/j.cma.2024.117595), 2025.

GNU GPL v3. See `LICENSE` and `NOTICE`.

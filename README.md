# formoniq

`formoniq` is a Rust finite element library for differential forms on
simplicial meshes. It implements the topology, metric operators, Whitney
finite element spaces, weak formulations, and boundary reductions used by the
FEEC–GMRF statistical models.

Finite element exterior calculus discretizes the de Rham complex while
preserving the identity \(d^2=0\):

```text
C^0 --D0--> C^1 --D1--> C^2 --D2--> C^3.
```

The incidence matrices depend only on mesh topology. Metric information enters
through mass matrices and reconstruction. This separation makes it possible to
represent scalar potentials, vector potentials, fluxes, and densities in
spaces with compatible differential operators.

## Capabilities

The workspace provides:

- simplicial topology, geometry, mesh generation, and mesh I/O;
- exterior algebra and Whitney basis functions;
- cochain projection and physical reconstruction;
- incidence, mass, inverse-mass, and weak Hodge–Laplacian assembly;
- homogeneous and prescribed essential-boundary reduction;
- natural boundary-vector assembly;
- reduced linear PDE systems;
- residual and sparse Jacobian assembly for nonlinear electromagnetic models;
- transient heat, wave, and eddy-current operators;
- VTK/VTU output;
- optional PETSc and SLEPc helper programs for large solves and eigenproblems.

The main crates are:

- `manifold`: simplicial topology, coordinates, metric data, mesh generation,
  and mesh I/O;
- `exterior`: algebraic differential forms;
- `ddf`: discrete differential forms, cochains, and Whitney elements;
- `common`: shared linear-algebra and geometry utilities;
- `formoniq`: finite element assembly, PDE formulations, reduction,
  reconstruction, and examples.

## Weak Hodge–Laplacian

For a \(k\)-form \(u\), the Hodge–Laplacian is

```text
Delta u = d delta u + delta d u.
```

The mixed weak formulation introduces the adjacent-degree auxiliary variable
needed to avoid forming a strong codifferential. `MixedGalmats` collects the
mass and incidence-derived blocks used by the reduced Hodge–Laplacian systems.
The same assembled mass matrices and weak operators are consumed by the
Matérn prior builders in FEEC–GMRF.

Boundary conditions are expressed through `DofLayout` and reduced assemblies.
Prescribed values contribute an affine boundary vector; they are not silently
discarded when the operator is restricted to active coefficients.

## Examples

The `formoniq` examples include:

- scalar Poisson and electrostatic problems;
- Darcy flow;
- Hodge–Laplacian source problems and eigenproblems;
- mixed-boundary Hodge–Laplacian systems;
- magnetostatics;
- heat and wave evolution;
- spherical harmonics and torus convergence;
- curvature flow.

Run an example from this workspace with:

```text
cargo run --release -p formoniq --example poisson
cargo run --release -p formoniq --example mixed_bc_hodge_laplacian
```

## Building and testing

```text
cargo test --release --workspace
cargo clippy --release --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --workspace --no-deps
```

The optional PETSc/SLEPc helpers are built from source. Select a real-scalar
PETSc configuration with MUMPS and a matching SLEPc installation:

```text
export PETSC_DIR=/absolute/path/to/petsc
export PETSC_ARCH=arch-mumps-opt
export SLEPC_DIR=/absolute/path/to/slepc
make -C petsc-solver clean all
cargo test --release -p formoniq --features external-solver-tests
```

For prefix installations, omit `PETSC_ARCH`. The parent FEEC–GMRF repository
also supplies `scripts/build-petsc-helpers.sh` to locate and build these
programs.

## Relationship to FEEC–GMRF

This workspace provides deterministic discretization and PDE assembly. It does
not implement Gaussian conditioning or sparse precision inference. The parent
integration library converts the assembled matrices and residuals into GMRF
priors, observations, and Laplace models.

## Acknowledgements and license

`formoniq` was created by Luis Wirth in connection with his ETH Zürich
bachelor's thesis on coordinate-free Whitney finite element exterior calculus.
The current implementation is derived from his
[public project](https://github.com/luiswirth/formoniq) and has been extended
by Patrick Dowd and contributors.

The workspace is available under either the MIT license or the Apache License
2.0. Both license texts are included.

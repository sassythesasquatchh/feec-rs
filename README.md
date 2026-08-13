# formoniq — FEEC thesis fork

This standalone workspace is an attributed thesis fork of Luis Wirth's
`formoniq`, a Rust finite-element library based on finite element exterior
calculus (FEEC). The crate name `formoniq` is retained for lineage and downstream
compatibility.

The recorded upstream baseline is Luis Wirth's commit
`65b98b55f3fee1c28bc37acb68981a3f3bd63e9e`, described upstream as the
“bsc-thesis version”. This fork has since added boundary-aware assembly,
nonlinear electromagnetic residuals and Jacobians, spatiotemporal slice
assembly, reconstruction operators, and backend-neutral integration contracts.
These additions distinguish the thesis fork from the upstream package.

The parent FEEC–GMRF integration workspace depends on this standalone
workspace. Verify it directly with:

```text
cargo test --release --workspace
cargo clippy --release --workspace --all-targets -- -D warnings
cargo doc --release --workspace --no-deps
```

## Repository contents

- `crates/` contains the FEEC workspace and its end-to-end Rust examples and
  tests.
- `petsc-solver/` contains the source and Makefile for optional external
  PETSc/SLEPc helper executables.
- `geometries/` contains the source geometry used by the maintained
  magnetostatic workflow.

Generated solver inputs, build output, editor configuration, and standalone
plotting tools are not part of this release repository.

To build the helpers from an in-place PETSc/SLEPc installation, select the same
MUMPS-enabled configuration used for both packages:

```text
export PETSC_DIR=/absolute/path/to/petsc
export PETSC_ARCH=arch-mumps-opt
export SLEPC_DIR=/absolute/path/to/slepc
make -C petsc-solver clean all
cargo test --release -p formoniq --features external-solver-tests
```

For prefix installations, omit `PETSC_ARCH`. Helper link failures are fatal;
precompiled executables are deliberately not distributed. The parent
FEEC--GMRF repository provides `scripts/build-petsc-helpers.sh` for validated
explicit-first selection with a `pkg-config` fallback.

The additional `parent-fixture-tests` feature exercises torus convergence data
owned by the parent integration repository and is run from that recursive
checkout; it is not part of the standalone FEEC gate.

FEEC owns topology, degrees of freedom, quadrature, exterior derivatives, mass
matrices, weak Hodge–Laplacian assembly, boundary reduction, PDE residual and
Jacobian assembly, and physical reconstruction. Gaussian conditioning and
sparse precision algorithms live outside this repository.

See `UPSTREAM.md` and `PROVENANCE.md` for lineage. This workspace retains
upstream dual licensing under MIT or Apache-2.0; both license texts are
included. Luis Wirth's current public repository remains the reference layout
for the upstream project: <https://github.com/luiswirth/formoniq>.

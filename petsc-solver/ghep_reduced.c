#include <petscsys.h>
#include <petscksp.h>
#include <slepceps.h>
#include <petscviewer.h>

static char help[] =
    "Reduced generalized Hermitian eigenproblem for the Hodge Laplacian.\n"
    "Solves (L + D * Mkm1^{-1} * C) u = lambda * Mk * u using a MatShell.\n"
    "Writes eigenvalues to out/eigenvals.bin, eigenvectors u to out/eigenvecs_u.bin,\n"
    "and corresponding sigma = Mkm1^{-1} * C * u to out/eigenvecs_sigma.bin.\n";

typedef struct
{
  Mat L;
  Mat D;
  Mat C;
  Mat Mkm1;
  KSP ksp_mkm1;
  Vec work_t; /* t = C*x (size n_sigma) */
  Vec work_z; /* z = Mkm1^{-1}*t (size n_sigma) */
} KShellCtx;

static PetscErrorCode KShellMult(Mat K, Vec x, Vec y)
{
  PetscErrorCode ierr;
  KShellCtx *ctx = NULL;

  ierr = MatShellGetContext(K, (void **)&ctx);
  CHKERRQ(ierr);

  /* t = C*x */
  ierr = MatMult(ctx->C, x, ctx->work_t);
  CHKERRQ(ierr);

  /* z = Mkm1^{-1}*t */
  ierr = KSPSolve(ctx->ksp_mkm1, ctx->work_t, ctx->work_z);
  CHKERRQ(ierr);

  /* y = L*x */
  ierr = MatMult(ctx->L, x, y);
  CHKERRQ(ierr);

  /* y += D*z */
  ierr = MatMultAdd(ctx->D, ctx->work_z, y, y);
  CHKERRQ(ierr);

  return 0;
}

static PetscErrorCode load_mat(const char *path, Mat *mat)
{
  PetscErrorCode ierr;
  PetscViewer viewer;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, path, FILE_MODE_READ, &viewer);
  CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, mat);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(*mat);
  CHKERRQ(ierr);
  ierr = MatLoad(*mat, viewer);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);
  CHKERRQ(ierr);

  return 0;
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  ierr = SlepcInitialize(&argc, &argv, NULL, help);
  if (ierr)
    return ierr;

  Mat L = NULL, D = NULL, C = NULL, Mkm1 = NULL, Mk = NULL;

  ierr = load_mat("in/L.bin", &L);
  CHKERRQ(ierr);
  ierr = load_mat("in/D.bin", &D);
  CHKERRQ(ierr);
  ierr = load_mat("in/C.bin", &C);
  CHKERRQ(ierr);
  ierr = load_mat("in/Mkm1.bin", &Mkm1);
  CHKERRQ(ierr);
  ierr = load_mat("in/Mk.bin", &Mk);
  CHKERRQ(ierr);

  PetscInt n_u = 0, n_sigma = 0;
  ierr = MatGetSize(Mk, &n_u, NULL);
  CHKERRQ(ierr);
  ierr = MatGetSize(Mkm1, &n_sigma, NULL);
  CHKERRQ(ierr);

  /* Shell context */
  KShellCtx *ctx = NULL;
  ierr = PetscNew(&ctx);
  CHKERRQ(ierr);
  ctx->L = L;
  ctx->D = D;
  ctx->C = C;
  ctx->Mkm1 = Mkm1;

  /* Workspace vectors */
  ierr = MatCreateVecs(C, NULL, &ctx->work_t);
  CHKERRQ(ierr); /* range(C) */
  ierr = MatCreateVecs(Mkm1, NULL, &ctx->work_z);
  CHKERRQ(ierr); /* range(Mkm1) */

  /* KSP to apply Mkm1^{-1} */
  ierr = KSPCreate(PETSC_COMM_WORLD, &ctx->ksp_mkm1);
  CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->ksp_mkm1, Mkm1, Mkm1);
  CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ctx->ksp_mkm1, "mkm1_");
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ctx->ksp_mkm1);
  CHKERRQ(ierr);
  ierr = KSPSetUp(ctx->ksp_mkm1);
  CHKERRQ(ierr);

  /* Shell matrix K = L + D * Mkm1^{-1} * C */
  Mat Kshell = NULL;
  ierr = MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n_u, n_u, ctx, &Kshell);
  CHKERRQ(ierr);
  ierr = MatShellSetOperation(Kshell, MATOP_MULT, (void (*)(void))KShellMult);
  CHKERRQ(ierr);

  /* Inform SLEPc/PETSc about structure (only correct if it truly holds) */
  ierr = MatSetOption(Kshell, MAT_SYMMETRIC, PETSC_TRUE);
  CHKERRQ(ierr);
  ierr = MatSetOption(Mk, MAT_SPD, PETSC_TRUE);
  CHKERRQ(ierr);

  /* Eigenproblem: K u = lambda Mk u */
  EPS eps = NULL;
  ierr = EPSCreate(PETSC_COMM_WORLD, &eps);
  CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, Kshell, Mk);
  CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHEP);
  CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
  CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);
  CHKERRQ(ierr);

  ierr = EPSSolve(eps);
  CHKERRQ(ierr);

  PetscInt npairs = 0;
  ierr = EPSGetConverged(eps, &npairs);
  CHKERRQ(ierr);

  /* Vector for u eigenvector */
  Vec xr = NULL;
  ierr = MatCreateVecs(Mk, NULL, &xr);
  CHKERRQ(ierr);

  /* Output: eigenvalues */
  PetscViewer viewer_eigenvals = NULL;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "out/eigenvals.bin", FILE_MODE_WRITE,
                               &viewer_eigenvals);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer_eigenvals, &npairs, 1, PETSC_INT);
  CHKERRQ(ierr);

  /* Output: u eigenvectors */
  PetscViewer viewer_u = NULL;
  PetscInt u_size = n_u;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "out/eigenvecs_u.bin", FILE_MODE_WRITE,
                               &viewer_u);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer_u, &u_size, 1, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer_u, &npairs, 1, PETSC_INT);
  CHKERRQ(ierr);

  /* Output: sigma eigenvectors */
  PetscViewer viewer_sigma = NULL;
  PetscInt sigma_size = n_sigma;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "out/eigenvecs_sigma.bin", FILE_MODE_WRITE,
                               &viewer_sigma);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer_sigma, &sigma_size, 1, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer_sigma, &npairs, 1, PETSC_INT);
  CHKERRQ(ierr);

  for (PetscInt i = 0; i < npairs; ++i)
  {
    PetscScalar kr = 0.0;

    /* Get eigenpair (lambda, u) */
    ierr = EPSGetEigenpair(eps, i, &kr, NULL, xr, NULL);
    CHKERRQ(ierr);

    /* sigma = Mkm1^{-1} * C * u */
    ierr = MatMult(C, xr, ctx->work_t);
    CHKERRQ(ierr);
    ierr = KSPSolve(ctx->ksp_mkm1, ctx->work_t, ctx->work_z);
    CHKERRQ(ierr);

    /* Write lambda */
    ierr = PetscViewerBinaryWrite(viewer_eigenvals, &kr, 1, PETSC_SCALAR);
    CHKERRQ(ierr);

    /* Write u and sigma separately */
    ierr = VecView(xr, viewer_u);
    CHKERRQ(ierr);
    ierr = VecView(ctx->work_z, viewer_sigma);
    CHKERRQ(ierr);
  }

  /* Clean up viewers */
  ierr = PetscViewerDestroy(&viewer_sigma);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_u);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_eigenvals);
  CHKERRQ(ierr);

  /* Clean up PETSc/SLEPc objects */
  ierr = VecDestroy(&xr);
  CHKERRQ(ierr);

  ierr = MatDestroy(&Kshell);
  CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp_mkm1);
  CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->work_t);
  CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->work_z);
  CHKERRQ(ierr);
  ierr = PetscFree(ctx);
  CHKERRQ(ierr);

  ierr = MatDestroy(&L);
  CHKERRQ(ierr);
  ierr = MatDestroy(&D);
  CHKERRQ(ierr);
  ierr = MatDestroy(&C);
  CHKERRQ(ierr);
  ierr = MatDestroy(&Mkm1);
  CHKERRQ(ierr);
  ierr = MatDestroy(&Mk);
  CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);
  CHKERRQ(ierr);

  ierr = SlepcFinalize();
  return ierr;
}

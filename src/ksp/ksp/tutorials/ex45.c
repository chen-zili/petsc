
/*
Laplacian in 3D. Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

with boundary conditions

   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

   This uses multigrid to solve the linear system

   See src/snes/tutorials/ex50.c

   Can also be run with -pc_type exotic -ksp_type fgmres

*/

static char help[] = "Solves 3D Laplacian using multigrid.\n\n";

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);
extern PetscErrorCode ComputeInitialGuess(KSP,Vec,void*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PetscReal      norm;
  DM             da;
  Vec            x,b,r;
  Mat            A;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,7,7,7,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);
  ierr = KSPSetComputeInitialGuess(ksp,ComputeInitialGuess,NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);

  ierr = MatMult(A,x,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  DM             dm;
  PetscScalar    Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  PetscScalar    ***barray;

  PetscFunctionBeginUser;
  ierr    = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr    = DMDAGetInfo(dm,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx      = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;
  ierr    = DMDAGetCorners(dm,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr    = DMDAVecGetArray(dm,b,&barray);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
          barray[k][j][i] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
        } else {
          barray[k][j][i] = Hx*Hy*Hz;
        }
        // printf("%f ", barray[k][j][i]);
      }
      // printf("\n");
    }
    // printf("\n");
  }
  ierr = DMDAVecRestoreArray(dm,b,&barray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeInitialGuess(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(b,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp,Mat jac,Mat B,void *ctx)
{
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    v[7],Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col[7];

  PetscFunctionBeginUser;
  ierr    = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr    = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx      = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;
  ierr    = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.i = i; row.j = j; row.k = k;
        if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
          v[0] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
          ierr = MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          v[0] = -HxHydHz;col[0].i = i; col[0].j = j; col[0].k = k-1;
          v[1] = -HxHzdHy;col[1].i = i; col[1].j = j-1; col[1].k = k;
          v[2] = -HyHzdHx;col[2].i = i-1; col[2].j = j; col[2].k = k;
          v[3] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
          v[4] = -HyHzdHx;col[4].i = i+1; col[4].j = j; col[4].k = k;
          v[5] = -HxHzdHy;col[5].i = i; col[5].j = j+1; col[5].k = k;
          v[6] = -HxHydHz;col[6].i = i; col[6].j = j; col[6].k = k+1;
          ierr = MatSetValuesStencil(B,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr   = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr   = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -pc_type exotic -ksp_monitor_short -ksp_type fgmres -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi
      output_file: output/ex45_1.out

   test:
      suffix: 2
      nsize: 4
      args: -ksp_monitor_short -da_grid_x 21 -da_grid_y 21 -da_grid_z 21 -pc_type mg -pc_mg_levels 3 -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi

   test:
      suffix: telescope
      nsize: 4
      args: -ksp_type fgmres -ksp_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_levels 2 -da_grid_x 65 -da_grid_y 65 -da_grid_z 65 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_ignore_kspcomputeoperators -mg_coarse_pc_telescope_reduction_factor 4 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_ksp_type richardson -ksp_rtol 1.0e-4

   test:
      suffix: telescope_2
      nsize: 4
      args: -ksp_type fgmres -ksp_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_levels 2 -da_grid_x 65 -da_grid_y 65 -da_grid_z 65 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_reduction_factor 2 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_ksp_type richardson -ksp_rtol 1.0e-4

TEST*/

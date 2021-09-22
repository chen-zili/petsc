/*------------------------------------------------------------------
对比2维泊松方程的解析解和64x64网格的数值解(串行求解)
方程:
						- Laplacian u = 1      -1 < x, y < 1

边界：          
						x = -1  x = 1  y = -1  y = 1    -->    u = 0


解析解:
			  			u = -0.25x^2 + 0.25 - 0.25y^2 + 0.25


离散化:
						- Laplacian u = 1  -->  Ax = b

	A: 
		1.边界处
						x = 0(可以设置为A=1, b=0)
			
		2.非边界处
			  			(-x(i-1, j) + 2x(i, j) - x(i+1, j)) / dx^2 + 
			  			(-x(i, j-1) + 2x(i, j) - x(i, j+1)) / dy^2) = 1

	b:
		1.边界处
						b = 0
				
		2.非边界处
						b = 1
			
------------------------------------------------------------------*/

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <stdlib.h>

static char help[] = "Solves 2D Laplacian using multigrid.\n\n";

extern PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void*);
extern PetscErrorCode ComputeRHS(KSP, Vec, void*);
extern PetscErrorCode ComputeInitialGuess(KSP, Vec, void*);

int main(int argc, char **argv)
{
	PetscErrorCode 	ierr;
	KSP            	ksp;
	PetscReal      	norm;
	DM             	da;
	Vec            	x, b, r;
	Mat				A;
	PetscInt		m = 65;

	// 初始化
	ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;

	// 从命令行获取的参数
	ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL); CHKERRQ(ierr);

	// 创建ksp求解器
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	
	// 创建2d网格结构
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, \
		m, m, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da); CHKERRQ(ierr);
	ierr = DMSetFromOptions(da); CHKERRQ(ierr);
	ierr = DMSetUp(da); CHKERRQ(ierr);
	ierr = KSPSetDM(ksp, da); CHKERRQ(ierr);

	// 设置x的初始化位置、设置b、设置A
	ierr = KSPSetComputeInitialGuess(ksp, ComputeInitialGuess, NULL); CHKERRQ(ierr);
	ierr = KSPSetComputeRHS(ksp, ComputeRHS, NULL); CHKERRQ(ierr);
	ierr = KSPSetComputeOperators(ksp, ComputeMatrix, NULL); CHKERRQ(ierr);
	
	ierr = DMDestroy(&da); CHKERRQ(ierr);

	// 求解Ax=b
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	ierr = KSPSolve(ksp, NULL, NULL); CHKERRQ(ierr);

	// 获取求解后参数
	ierr = KSPGetSolution(ksp, &x); CHKERRQ(ierr);
	ierr = KSPGetRhs(ksp, &b); CHKERRQ(ierr);
	ierr = VecDuplicate(b, &r); CHKERRQ(ierr);
	ierr = KSPGetOperators(ksp, &A, NULL); CHKERRQ(ierr);

	// // 打印A, x, b
	// printf("\nA:\n");
	// ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);\
	// printf("\nx:\n");
	// ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	printf("\nb:\n");
	ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// 输出结果到文件
	PetscViewer myViewer;
	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "poisson2d.txt", &myViewer); CHKERRQ(ierr);
	ierr = VecView(x, myViewer); CHKERRQ(ierr);

	// 调用python脚本绘图
	system("./plot3d.py poisson2d.txt");

	// 计算误差Ax和b的误差
	ierr = MatMult(A, x, r); CHKERRQ(ierr);
	ierr = VecAXPY(r, -1.0, b); CHKERRQ(ierr);
	ierr = VecNorm(r, NORM_2, &norm); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm %g\n", (double)norm); CHKERRQ(ierr);

	// 释放内存
	ierr = VecDestroy(&r); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	ierr = PetscFinalize();

	return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
	PetscErrorCode ierr;
	PetscInt       i, j, mx, my, xm, ym, xs, ys;
	DM             dm;
	PetscScalar    **barray;

	PetscFunctionBeginUser;
	ierr = VecSet(b, 1.); CHKERRQ(ierr);

	// 获取ksp绑定的dm信息
	ierr = KSPGetDM(ksp, &dm); CHKERRQ(ierr);

	// 从dm中获取x, y方向上的网格数(64, 64)
	ierr = DMDAGetInfo(dm, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);

	// 获取左下角(xs, ys)、右上角(xm, ym)坐标
	ierr = DMDAGetCorners(dm, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
	
	// 获得Vec b存储空间的首地址
	ierr = DMDAVecGetArray(dm, b, &barray); CHKERRQ(ierr);

	// 设置b
	for (j=ys; j<ys+ym; j++) 
	{
		for (i=xs; i<xs+xm; i++) 
		{
			// if ((i == 0 && j == 0) || (i == 0 && j == my-1) || (i == mx-1 && j == 0) || (i == mx-1 && j == my-1))
			// {
			// 	barray[j][i] = 0.0;
			// }
			// else if (i==0 || j==0 || i==mx-1 || j==my-1 ) 
			// {
			// 	// 边界 b = 0
			// 	barray[j][i] = 0.0;
			// } 
			// else 
			// {
			// 	barray[j][i] = 1.;
			// }
			if (0 == j)
			{
				// y = -1
				barray[j][i] = 0.0;
			}
			else if (my-1 == j)
			{
				// y = 1
				barray[j][i] = 1.0;
			}
			else if (0 == i || mx-1 == i)
			{
				barray[j][i] = j / 64.0;
			}
			else
			{
				barray[j][i] = 0.0;
			}
			// printf("%f ", barray[j][i]);
		}
		// printf("\n");
	}

	ierr = DMDAVecRestoreArray(dm, b, &barray); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

PetscErrorCode ComputeInitialGuess(KSP ksp, Vec x, void *ctx)
{
	PetscErrorCode ierr;

	PetscFunctionBeginUser;
	ierr = VecSet(x, 0); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat jac, Mat B, void *ctx)
{
	DM             da;
	PetscErrorCode ierr;
	PetscInt       i, j, mx, my, xm, ym, xs, ys;
	PetscScalar    v[5], dHxdHx, dHydHy;
	MatStencil     row, col[5];

	PetscFunctionBeginUser;
	ierr    = KSPGetDM(ksp, &da); CHKERRQ(ierr);
	ierr    = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);

	// dHxdHx = dx^2 = (64 - 1)^2
	dHxdHx = 1.0 / (2.0 / ((PetscReal)((mx-1) * (mx-1))));
	dHydHy = 1.0 / (2.0 / ((PetscReal)((my-1) * (my-1))));

	ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);


	// 设置A
	for (j=ys; j<ys+ym; j++) 
	{
		for (i=xs; i<xs+xm; i++) 
		{
			row.i = i; row.j = j;
			// if ((i == 0 && j == 0) || (i == 0 && j == my-1) || (i == mx-1 && j == 0) || (i == mx-1 && j == my-1))
			// {
			// 	// 边界
			// 	v[0] = 1.0;
			// 	ierr = MatSetValuesStencil(B, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
			// }
			// else if (i==0 || i==mx-1) 
			// {
			// 	// 边界
			// 	v[0] = -dHydHy; col[0].i = i; col[0].j = j-1;
			// 	v[1] = 2.0*dHydHy; col[2].i = row.i; col[2].j = row.j;
			// 	v[2] = -dHydHy; col[4].i = i; col[4].j = j+1;
			// 	ierr = MatSetValuesStencil(B, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);
			// }
			// else if (j ==0 || j == my-1)
			// {
			// 	// 边界
			// 	v[0] = -dHxdHx; col[1].i = i-1; col[1].j = j;
			// 	v[1] = 2.0*dHxdHx; col[2].i = row.i; col[2].j = row.j;
			// 	v[2] = -dHxdHx; col[3].i = i+1; col[3].j = j;
			// 	ierr = MatSetValuesStencil(B, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);
			// }
			if (i == 0 || i == mx-1 || j == 0 || j == my-1)
			{
				// 边界
				v[0] = 1.0;
				ierr = MatSetValuesStencil(B, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
			{
				v[0] = -dHydHy; col[0].i = i; col[0].j = j-1;
				v[1] = -dHxdHx; col[1].i = i-1; col[1].j = j;
				v[2] = 2.0*(dHxdHx + dHydHy); col[2].i = row.i; col[2].j = row.j;
				v[3] = -dHxdHx; col[3].i = i+1; col[3].j = j;
				v[4] = -dHydHy; col[4].i = i; col[4].j = j+1;
				ierr = MatSetValuesStencil(B, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
			}
		}
	}

	ierr   = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr   = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
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

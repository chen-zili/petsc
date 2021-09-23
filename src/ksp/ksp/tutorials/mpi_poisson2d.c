/*-------------------------------------------------------------------
2维泊松方程并行求解，默认格点数(65x65)
方程:
						- Laplacian u = 1      -1 < x, y < 1

边界：          
						上边界为1, 下边界为0, 左右两侧从0到1(从下边界至上边界是0到1)


离散化:
						- Laplacian u = 1  -->  Ax = b

	A: 
		1.边界处
						A = 1
			
		2.非边界处
			  			(-x(i-1, j) + 2x(i, j) - x(i+1, j)) / dx^2 + 
			  			(-x(i, j-1) + 2x(i, j) - x(i, j+1)) / dy^2) = 1

	b:
		1.边界处
						b = 1(上边界)
						b = 0(下边界)
						b = 0~1(左右侧)
				
		2.非边界处
						b = 1


并行:
    迭代次数
    均方误差
    运行时间

-------------------------------------------------------------------*/


#include <petscksp.h>
#include <stdlib.h>

static char help[] = "Solves 2D Laplacian using mpi.\n\n";

extern PetscErrorCode setMatBlockValue(Mat A, int gridNum);
extern PetscErrorCode setVecBlockValue(Vec b, int gridNum);

int main(int argc, char **args)
{
	PetscErrorCode 	ierr;
	KSP            	ksp;
	PetscReal      	norm;
	Vec            	x, b, r;
	Mat				A;
	PetscInt		gridNum = 65, its;
    PetscMPIInt     rank;

    // 初始化
    ierr = PetscInitialize(&argc, &args, (char*)0, help); if (ierr) return ierr;
    
    // 命令行参数
    ierr = PetscOptionsGetInt(NULL, NULL, "-pn", &gridNum, NULL);
    CHKERRQ(ierr);

    // 格点数量
    PetscInt gridPointNum = gridNum*gridNum;

    // 组装矩阵A
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, gridPointNum, \
    gridPointNum); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);

    // 预分配内存, 提升性能
    // 对于A中的每行，有意义的只有A(i,j), A(i-1,j), A(i+1,j), A(i,j-1), A(i,j+1)
    // 所以分配5个就够了
    ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(A,1,5,NULL);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A,1,5,NULL,5,NULL);CHKERRQ(ierr);
    ierr = MatMPISELLSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
    ierr = MatSeqSELLSetPreallocation(A,5,NULL);CHKERRQ(ierr);

    // A的分块赋值
    ierr = setMatBlockValue(A, gridNum); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    // b的创建与分块赋值
    ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
    ierr = VecSetSizes(b, PETSC_DECIDE, gridPointNum); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &r); CHKERRQ(ierr);

    ierr = setVecBlockValue(b, gridNum); CHKERRQ(ierr);


    // ksp
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

    // 设置ksp的迭代参数
    // ierr = KSPSetTolerances(ksp, 1.e-2/gridPointNum, 1.e-50, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);

    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    // 求解
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);


    // 解算结果
    // 只运行一次
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    if (0 == rank)
    {
        ierr = MatMult(A, x, r); CHKERRQ(ierr);
        ierr = VecAXPY(r, -1.0, b); CHKERRQ(ierr);
        ierr = VecNorm(r, NORM_2, &norm); CHKERRQ(ierr);

        ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);

        printf("Residual norm %lf\n", (double)norm);
        printf("Iterations %d\n", its);
    }

    // 释放内存
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);

    ierr = PetscFinalize();
    return ierr;
}


// A的分块赋值
// Mat和Vec其实是指针，这里传递的是地址
PetscErrorCode setMatBlockValue(Mat A, int gridNum)
{
    PetscErrorCode      ierr;
    PetscInt            i, j, Istart, Iend;
    PetscScalar         v[5], dHxdHx, dHydHy;
    MatStencil          row, col[5];

    PetscFunctionBeginUser;
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);

	// dHxdHx = dx^2 = (64 - 1)^2
	dHxdHx = 1.0 / (4.0 / ((PetscReal)((gridNum-1) * (gridNum-1))));
	dHydHy = 1.0 / (4.0 / ((PetscReal)((gridNum-1) * (gridNum-1))));

    for (j = Istart/gridNum; j < Iend/gridNum; j++)
    {
        for (i = 0; i < gridNum; i++)
        {
            row.i = i;
            row.j = j;
			if (i == 0 || i == gridNum-1 || j == 0 || j == gridNum-1)
			{
				// 边界
				v[0] = 1.0;
				ierr = MatSetValuesStencil(A, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
			}
			else
			{
				v[0] = -dHydHy; col[0].i = i; col[0].j = j-1;
				v[1] = -dHxdHx; col[1].i = i-1; col[1].j = j;
				v[2] = 2.0*(dHxdHx + dHydHy); col[2].i = row.i; col[2].j = row.j;
				v[3] = -dHxdHx; col[3].i = i+1; col[3].j = j;
				v[4] = -dHydHy; col[4].i = i; col[4].j = j+1;
				ierr = MatSetValuesStencil(A, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
			}
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode setVecBlockValue(Vec b, int gridNum)
{
    PetscErrorCode      ierr;
    PetscInt            i, j, Istart, Iend;
    PetscScalar    **barray;

    PetscFunctionBeginUser;
    ierr = VecGetOwnershipRange(b, &Istart, &Iend); CHKERRQ(ierr);
    ierr = VecGetArray(b, barray); CHKERRQ(ierr);

    for (j = Istart/gridNum; j < Iend/gridNum; j++)
    {
        for (i = 0; i < gridNum; i++)
        {
			if (0 == j)
			{
				// y = -1
				barray[j][i] = 0.0;
			}
			else if (gridNum-1 == j)
			{
				// y = 1
				barray[j][i] = 1.0;
			}
			else if (0 == i || gridNum-1 == i)
			{
				barray[j][i] = j / (gridNum - 1.0);
			}
			else
			{
				barray[j][i] = 1.0;
			}
        }
    }

    ierr = VecRestoreArray(b, barray); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#define PETSCMAT_DLL

/*
  Defines basic operations for the MATSEQCSRPERM matrix class.
  This class is derived from the MATSEQAIJ class and retains the 
  compressed row storage (aka Yale sparse matrix format) but augments 
  it with some permutation information that enables some operations 
  to be more vectorizable.  A physically rearranged copy of the matrix 
  may be stored if the user desires.

  Eventually a variety of permutations may be supported.
*/


#include "src/mat/impls/aij/seq/aij.h"

typedef struct {
  PetscInt ngroup;
  PetscInt *xgroup;
    /* Denotes where groups of rows with same number of nonzeros 
     * begin and end, i.e., xgroup[i] gives us the position in iperm[] 
     * where the ith group begins. */
  PetscInt *nzgroup;
    /* nzgroup[i] tells us how many nonzeros each row that is a member 
     * of group i has. */
  PetscInt *iperm;  /* The permutation vector. */

  /* Flag that indicates whether we need to clean up permutation 
   * information during the MatDestroy. */
  PetscTruth CleanUpCSRPERM;

  /* Some of this stuff is for Ed's recursive triangular solve.
   * I'm not sure what I need yet. */
  PetscInt blocksize;
  PetscInt nstep;
  PetscInt *jstart_list;
  PetscInt *jend_list;
  PetscInt *action_list;
  PetscInt *ngroup_list;
  PetscInt **ipointer_list;
  PetscInt **xgroup_list;
  PetscInt **nzgroup_list;
  PetscInt **iperm_list;

  /* We need to keep a pointer to MatAssemblyEnd_SeqAIJ because we 
   * actually want to call this function from within the 
   * MatAssemblyEnd_SeqCSRPERM function.  Similarly, we also need 
   * MatDestroy_SeqAIJ and MatDuplicate_SeqAIJ. */
  PetscErrorCode (*AssemblyEnd_SeqAIJ)(Mat,MatAssemblyType);
  PetscErrorCode (*MatDestroy_SeqAIJ)(Mat);
  PetscErrorCode (*MatDuplicate_SeqAIJ)(Mat,MatDuplicateOption,Mat*);
  
} Mat_SeqCSRPERM;




#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqCSRPERM"
PetscErrorCode MatDestroy_SeqCSRPERM(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqCSRPERM *csrperm;

  csrperm = (Mat_SeqCSRPERM *) A->spptr;

  /* We are going to convert A back into a SEQAIJ matrix, since we are 
   * eventually going to use MatDestroy_SeqAIJ() to destroy everything 
   * that is not specific to CSRPERM.
   * In preparation for this, reset the operations pointers in A to 
   * their SeqAIJ versions. */
  A->ops->assemblyend = csrperm->AssemblyEnd_SeqAIJ;
    /* I suppose I don't actually need to point A->ops->assemblyend back 
     * to the right thing, but I do it anyway for completeness. */
  A->ops->destroy = csrperm->MatDestroy_SeqAIJ;
  A->ops->duplicate = csrperm->MatDuplicate_SeqAIJ;

  /* Free everything in the Mat_SeqCSRPERM data structure. */
  if(csrperm->CleanUpCSRPERM) {
    ierr = PetscFree(csrperm->xgroup);CHKERRQ(ierr);
    ierr = PetscFree(csrperm->nzgroup);CHKERRQ(ierr);
    ierr = PetscFree(csrperm->iperm);CHKERRQ(ierr);
  }

  /* Free the Mat_SeqCSRPERM struct itself. */
  ierr = PetscFree(csrperm);CHKERRQ(ierr);

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ() 
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that 
   * is only to be called when *building* a matrix.  I could be wrong, but 
   * that is how things work for the SuperLU matrix class. */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SeqCSRPERM"
PetscErrorCode MatDuplicate_SeqCSRPERM(Mat A, MatDuplicateOption op, Mat *M) {
  PetscErrorCode ierr;
  Mat_SeqCSRPERM *csrperm = (Mat_SeqCSRPERM *) A->spptr;

  PetscFunctionBegin;
  ierr = (*csrperm->MatDuplicate_SeqAIJ)(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,csrperm,sizeof(Mat_SeqCSRPERM));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode SeqCSRPERM_create_perm(Mat A)
{
  PetscInt m;  /* Number of rows in the matrix. */
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)(A)->data;
  PetscInt *ia;
    /* The 'ia' array from the CSR representation; points to the beginning 
     * of each row. */
  PetscInt maxnz; /* Maximum number of nonzeros in any row. */
  PetscInt *rows_in_bucket;
    /* To construct the permutation, we sort each row into one of maxnz 
     * buckets based on how many nonzeros are in the row. */
  PetscInt nz;
  PetscInt *nz_in_row;
    /* nz_in_row[k] is the number of nonzero elements in row k. */
  PetscInt *ipnz;
    /* When constructing the iperm permutation vector, 
     * ipnz[nz] is used to point to the next place in the permutation vector 
     * that a row with nz nonzero elements should be placed.*/
  Mat_SeqCSRPERM *csrperm;
    /* Points to the MATSEQCSRPERM-specific data in the matrix B. */
  PetscErrorCode ierr;
  PetscInt i, ngroup, istart, ipos;

  /* I really ought to put something in here to check if B is of 
   * type MATSEQCSRPERM and return an error code if it is not.
   * Come back and do this! */
   
  m = A->m;
  ia = a->i;
  csrperm = (Mat_SeqCSRPERM*) A->spptr;
   
  /* Allocate the arrays that will hold the permutation vector. */
  ierr = PetscMalloc( m*sizeof(PetscInt), &csrperm->iperm); CHKERRQ(ierr);

  /* Allocate some temporary work arrays that will be used in 
   * calculating the permuation vector and groupings. */
  ierr = PetscMalloc( (m+1)*sizeof(PetscInt), &rows_in_bucket); CHKERRQ(ierr);
  ierr = PetscMalloc( (m+1)*sizeof(PetscInt), &ipnz); CHKERRQ(ierr);
  ierr = PetscMalloc( m*sizeof(PetscInt), &nz_in_row); CHKERRQ(ierr); 

  /* Now actually figure out the permutation and grouping. */

  /* First pass: Determine number of nonzeros in each row, maximum 
   * number of nonzeros in any row, and how many rows fall into each  
   * "bucket" of rows with same number of nonzeros. */
  maxnz = 0;
  for(i=0; i<m; i++) {
	  nz_in_row[i] = ia[i+1]-ia[i];
    if(nz_in_row[i] > maxnz) maxnz = nz;
  }

  for(i=0; i<=maxnz; i++) {
    rows_in_bucket[i] = 0;
  }
  for(i=0; i<m; i++) {
    nz = nz_in_row[i];
    rows_in_bucket[nz]++;
  }

  /* Allocate space for the grouping info.  There will be at most maxnz 
   * groups.  We allocate space for this many; that is potentially a 
   * little wasteful, but not too much so.  Perhaps I should fix it later. */
  ierr = PetscMalloc(maxnz*sizeof(PetscInt), &csrperm->xgroup); CHKERRQ(ierr);
  ierr = PetscMalloc(maxnz*sizeof(PetscInt), &csrperm->nzgroup); CHKERRQ(ierr);

  /* Second pass.  Look at what is in the buckets and create the groupings.
   * Note that it is OK to have a group of rows with no non-zero values. */
  ngroup = 0;
  istart = 0;
  for(i=0; i<=maxnz; i++) {
    if(rows_in_bucket[i] > 0) {
      csrperm->nzgroup[ngroup] = i;
      csrperm->xgroup[ngroup] = istart;
      ngroup++;
      istart += rows_in_bucket[i];
    }
  }

  csrperm->xgroup[ngroup] = istart;
  csrperm->ngroup = ngroup;

  /* Now fill in the permutation vector iperm. */
  ipnz[0] = 0;
  for(i=0; i<maxnz; i++) {
    ipnz[i+1] = ipnz[i] + rows_in_bucket[i];
  }

  for(i=0; i<m; i++) {
    nz = nz_in_row[i];
    ipos = ipnz[nz];
    csrperm->iperm[ipos] = i;
    ipnz[nz]++;
  }

  /* Clean up temporary work arrays. */
  ierr = PetscFree(rows_in_bucket); CHKERRQ(ierr);
  ierr = PetscFree(ipnz); CHKERRQ(ierr);
  ierr = PetscFree(nz_in_row); CHKERRQ(ierr);

  /* Since we've allocated some memory to hold permutation info, 
   * flip the CleanUpCSRPERM flag to true. */
  csrperm->CleanUpCSRPERM = PETSC_TRUE;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqCSRPERM"
PetscErrorCode MatAssemblyEnd_SeqCSRPERM(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqCSRPERM *csrperm;

  csrperm = (Mat_SeqCSRPERM*) A->spptr;

  if(mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  
  /* Since a MATSEQCSRPERM matrix is really just a MATSEQAIJ with some 
   * extra information, call the AssemblyEnd routine for a MATSEQAIJ. 
   * I'm not sure if this is the best way to do this, but it avoids 
   * a lot of code duplication.
   * I also note that currently MATSEQCSRPERM doesn't know anything about 
   * the Mat_CompressedRow data structure that SeqAIJ now uses when there 
   * are many zero rows.  If the SeqAIJ assembly end routine decides to use 
   * this, this may break things.  (Don't know... haven't looked at it.) */
  (*csrperm->AssemblyEnd_SeqAIJ)(A, mode);

  /* Now calculate the permutation and grouping information. */
  ierr = SeqCSRPERM_create_perm(A);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqCSRPERM"
PetscErrorCode MatMult_SeqCSRPERM(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *x,*y,*aa;
  PetscErrorCode ierr;
  PetscInt       m=A->m,*aj,*ai;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  PetscInt       n,i,jrow,j,*ridx=PETSC_NULL;
  PetscScalar    sum;
  PetscTruth     usecprow=a->compressedrow.use;
#endif

  /* Variables that don't appear in MatMult_SeqAIJ. */
  Mat_SeqCSRPERM *csrperm;
  PetscInt *iperm;  /* Points to the permutation vector. */
  PetscInt *xgroup;
    /* Denotes where groups of rows with same number of nonzeros 
     * begin and end in iperm. */
  PetscInt *nzgroup;
  PetscInt ngroup;
  PetscInt igroup;
  PetscInt jstart,jend;
    /* jstart is used in loops to denote the position in iperm where a 
     * group starts; jend denotes the position where it ends.
     * (jend + 1 is where the next group starts.) */
  PetscInt iold,nz;
  PetscInt istart,iend,isize;
  PetscInt ipos;
  const PetscInt NDIM = 512;
    /* NDIM specifies how many rows at a time we should work with when 
     * performing the vectorized mat-vec.  This depends on various factors 
     * such as vector register length, etc., and I really need to add a 
     * way for the user (or the library) to tune this.  I'm setting it to  
     * 512 for now since that is what Ed D'Azevedo was using in his Fortran 
     * routines. */
  PetscScalar yp[NDIM];
  PetscInt ip[NDIM];
    /* yp[] and ip[] are treated as vector "registers" for performing 
     * the mat-vec. */

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj  = a->j;    /* aj[k] gives column index for element aa[k]. */
  aa    = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;   /* ai[k] is the position in aa and aj where row k starts. */

  /* Get the info we need about the permutations and groupings. */
  csrperm = (Mat_SeqCSRPERM *) A->spptr;
  iperm = csrperm->iperm;
  ngroup = csrperm->ngroup;
  xgroup = csrperm->xgroup;
  nzgroup = csrperm->nzgroup;
  
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCSRPERM)
  fortranmultcsrperm_(&m,x,ii,aj,aa,y);
#else

  for(igroup=0; igroup<ngroup; igroup++) {
    jstart = xgroup[igroup];
    jend = xgroup[igroup+1] - 1;

    nz = nzgroup[igroup];

    /* Handle the special cases where the number of nonzeros per row 
     * in the group is either 0 or 1. */
    if(nz == 0) {

    }
    else if(nz == 1) {

    }
    /* For the general case: */
    else {
    
      /* We work our way through the current group in chunks of NDIM rows 
       * at a time. */

      for(istart=jstart; istart<=jend; istart+=NDIM) {
        /* Figure out where the chunk of 'isize' rows ends in iperm.
         * 'isize may of course be less than NDIM for the last chunk. */
        iend = istart + (NDIM - 1);
        if(iend > jend) { iend = jend; }
        isize = iend - istart + 1;

        /* Initialize the yp[] array that will be used to hold part of 
         * the permuted results vector, and figure out where in aa each 
         * row of the chunk will begin. */
        for(i=0; i<isize; i++) {
          iold = iperm[istart + i];
            /* iold is a row number from the matrix A *before* reordering. */
          ip[i] = ai[iold];
            /* ip[i] tells us where the ith row of the chunk begins in aa. */
          yp[i] = (PetscScalar) 0.0;
        }

        /* If the number of zeros per row exceeds the number of rows in 
         * the chunk, we should vectorize along nz, that is, perform the 
         * mat-vec one row at a time as in the usual CSR case. */
        if(nz > isize) {
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI preferstream
#endif
          for(i=0; i<isize; i++) {
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI prefervector
#endif
            for(j=0; j<nz; j++) {
              ipos = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }
        /* Otherwise, there are enough rows in the chunk to make it 
         * worthwhile to vectorize across the rows, that is, to do the 
         * matvec by operating with "columns" of the chunk. */
        else {
          for(j=0; j<nz; j++) {
            for(i=0; i<isize; i++) {
              ipos = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }

#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI ivdep
#endif
        /* Put results from yp[] into non-permuted result vector y. */
        for(i=0; i<isize; i++) {
          y[iperm[istart+i]] = yp[i];
        }
      } /* End processing chunk of isize rows of a group. */
      
    } /* End handling matvec for chunk with nz > 1. */
  } /* End loop over igroup. */

#endif
  ierr = PetscLogFlops(2*a->nz - m);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* MatConvert_SeqAIJ_SeqCSRPERM converts a SeqAIJ matrix into a 
 * SeqCSRPERM matrix.  This routine is called by the MatCreate_SeqCSRPERM() 
 * routine, but can also be used to convert an assembled SeqAIJ matrix 
 * into a SeqCSRPERM one. */
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SeqCSRPERM"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_SeqCSRPERM(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert to MATSEQCSRPERM
   * from MATSEQAIJ, so we can ignore 'MatType Type'. */
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqCSRPERM *csrperm;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*) A->data;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_SeqCSRPERM,&csrperm);CHKERRQ(ierr);
  B->spptr = (void *) csrperm;

  /* Save a pointer to the original SeqAIJ assembly end routine, because we 
   * will want to use it later in the CSRPERM assembly end routine. 
   * Also, save a pointer to the original SeqAIJ Destroy routine, because we 
   * will want to use it in the CSRPERM destroy routine. */
  csrperm->AssemblyEnd_SeqAIJ = A->ops->assemblyend;
  csrperm->MatDestroy_SeqAIJ = A->ops->destroy;
  csrperm->MatDuplicate_SeqAIJ = A->ops->duplicate;

  /* Set function pointers for methods that we inherit from AIJ but 
   * override. */
  B->ops->duplicate = MatDuplicate_SeqCSRPERM;
  B->ops->assemblyend = MatAssemblyEnd_SeqCSRPERM;
  B->ops->destroy = MatDestroy_SeqCSRPERM;
  B->ops->mult = MatMult_SeqCSRPERM;
  /* B->ops->multadd = MatMultAdd_SeqCSRPERM; */
  /* I'll add MatMultAdd and other functions once I've got MatMult 
   * thoroughly tested. */

  /* If A has already been assembled, compute the permutation. */
  if(A->assembled == PETSC_TRUE) {
    ierr = SeqCSRPERM_create_perm(B);
  }
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqCSRPERM"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCSRPERM(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Following the example of the SuperLU class, I change the type name 
   * before calling MatSetType() to force proper construction of SeqAIJ 
   * and MATSEQCSRPERM types. */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQCSRPERM);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqCSRPERM(A,MATSEQCSRPERM,MAT_REUSE_MATRIX,&A);
  CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
EXTERN_C_END


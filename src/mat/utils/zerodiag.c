
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zerodiag.c,v 1.30 1998/11/06 22:40:58 balay Exp bsmith $";
#endif

/*
    This file contains routines to reorder a matrix so that the diagonal
    elements are nonzero.
 */

#include "src/mat/matimpl.h"       /*I  "mat.h"  I*/

#define SWAP(a,b) {int _t; _t = a; a = b; b = _t; }

#undef __FUNC__  
#define __FUNC__ "MatReorderForNonzeroDiagonal"
/*@
    MatReorderForNonzeroDiagonal - Changes matrix ordering to remove
    zeros from diagonal. This may help in the LU factorization to 
    prevent a zero pivot.

    Collective on Mat

    Input Parameters:
+   mat  - matrix to reorder
-   rmap,cmap - row and column permutations.  Usually obtained from 
               MatGetOrdering().

    Notes:
    This is not intended as a replacement for pivoting for matrices that
    have ``bad'' structure. It is only a stop-gap measure. Should be called
    after a call to MatGetOrdering(), this routine changes the column 
    ordering defined in cis.

    Options Database Keys (When using SLES):
+      -pc_ilu_nonzeros_along_diagonal
-      -pc_lu_nonzeros_along_diagonal

    Algorithm Notes:
    Column pivoting is used. 

    1) Choice of column is made by looking at the
       non-zero elements in the troublesome row for columns that are not yet 
       included (moving from left to right).
 
    2) If (1) fails we check all the columns to the left of the current row
       and see if one of them has could be swapped. It can be swapped if
       its corresponding row has a non-zero in the column it is being 
       swapped with; to make sure the previous nonzero diagonal remains 
       nonzero


@*/
int MatReorderForNonzeroDiagonal(Mat mat,double atol,IS ris,IS cis )
{
  int      ierr, prow, k, nz, n, repl, *j, *col, *row, m, *icol,nnz,*jj,kk;
  Scalar   *v,*vv;
  double   repla;
  IS       icis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(ris,IS_COOKIE);
  PetscValidHeaderSpecific(cis,IS_COOKIE);
  
  ierr = ISGetIndices(ris,&row); CHKERRQ(ierr);
  ierr = ISGetIndices(cis,&col); CHKERRQ(ierr);
  ierr = ISInvertPermutation(cis,&icis);CHKERRQ(ierr);
  ierr = ISGetIndices(icis,&icol); CHKERRQ(ierr);
  ierr = MatGetSize(mat,&m,&n); CHKERRQ(ierr);

  for (prow=0; prow<n; prow++) {
    ierr = MatGetRow( mat, row[prow], &nz, &j, &v ); CHKERRQ(ierr);
    for (k=0; k<nz; k++) {if (icol[j[k]] == prow) break;}
    if (k >= nz || PetscAbsScalar(v[k]) <= atol) {
      /* Element too small or zero; find the best candidate */
      repl  = prow;
      repla = (k >= nz) ? 0.0 : PetscAbsScalar(v[k]);
      /*
          Look for a later column we can swap with this one
      */
      for (k=0; k<nz; k++) {
	if (icol[j[k]] > prow && PetscAbsScalar(v[k]) > repla) {
          /* found a suitable later column */
	  repl  = icol[j[k]];   
	  repla = PetscAbsScalar(v[k]);
          SWAP(icol[col[prow]],icol[col[repl]]); 
          SWAP(col[prow],col[repl]); 
          goto found;
        }
      }
      /* 
           Did not find a suitable later column so look for an earlier column
	   We need to be sure that we don't introduce a zero in a previous
	   diagonal 
      */
      for (k=0; k<nz; k++) {
        if (icol[j[k]] < prow && PetscAbsScalar(v[k]) > repla) {
          /* See if this one will work */
          repl  = icol[j[k]];
          ierr = MatGetRow( mat, row[repl], &nnz, &jj, &vv ); CHKERRQ(ierr);
          for (kk=0; kk<nnz; kk++) {
            if (icol[jj[kk]] == prow && PetscAbsScalar(vv[kk]) > atol) {
	      repla = PetscAbsScalar(v[k]);
              ierr = MatRestoreRow( mat, row[repl], &nnz, &jj, &vv ); CHKERRQ(ierr);
              SWAP(icol[col[prow]],icol[col[repl]]); 
              SWAP(col[prow],col[repl]); 
              goto found;
	    }
          }
          ierr = MatRestoreRow( mat, row[repl], &nnz, &jj, &vv ); CHKERRQ(ierr);
        }
      }
      /* 
          No column  suitable; instead check all future rows 
          Note: this will be very slow 
      */
      for (k=prow+1; k<n; k++) {
        ierr = MatGetRow( mat, row[k], &nnz, &jj, &vv ); CHKERRQ(ierr);
        for (kk=0; kk<nnz; kk++) {
          if (icol[jj[kk]] == prow && PetscAbsScalar(vv[kk]) > atol) {
            /* found a row */
            SWAP(row[prow],row[k]);
            goto found;
          }
        }
        ierr = MatRestoreRow( mat, row[k], &nnz, &jj, &vv ); CHKERRQ(ierr);
      }

      found:;
    }
    ierr = MatRestoreRow( mat, row[prow], &nz, &j, &v ); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ris,&row); CHKERRQ(ierr);
  ierr = ISRestoreIndices(cis,&col); CHKERRQ(ierr);
  ierr = ISRestoreIndices(icis,&icol); CHKERRQ(ierr);
  ierr = ISDestroy(icis); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: inherit.c,v 1.19 1997/09/26 02:18:19 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/


#undef __FUNC__  
#define __FUNC__ "PetscHeaderCreate_Private"
/*
    Creates a base PETSc object header and fills in the default values.
   Called by the macro PetscHeaderCreate()
*/
int PetscHeaderCreate_Private(PetscObject h,int cookie,int type,MPI_Comm comm,int (*des)(PetscObject),
                              int (*vie)(PetscObject,Viewer))
{
  h->cookie        = cookie;
  h->type          = type;
  h->prefix        = 0;
  h->refct         = 1;
  h->destroypublic = des;
  h->viewpublic    = vie;
  PetscCommDup_Private(comm,&h->comm,&h->tag);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscHeaderDestroy_Private"
/*
    Destroys a base PETSc object header. Called by macro PetscHeaderDestroy.
*/
int PetscHeaderDestroy_Private(PetscObject h)
{
  int ierr;

  PetscCommFree_Private(&h->comm);
  h->cookie = PETSCFREEDHEADER;
  if (h->prefix) PetscFree(h->prefix);
  if (h->child) {
    ierr = (*h->childdestroy)(h->child); CHKERRQ(ierr);
  }
  if (h->fortran_func_pointers) {
    PetscFree(h->fortran_func_pointers);
  }
  PetscFree(h);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectInherit_DefaultCopy"
/*
    The default copy simply copies the pointer and adds one to the 
  reference counter.

*/
static int PetscObjectInherit_DefaultCopy(void *in, void **out)
{
  PetscObject obj = (PetscObject) in;

  obj->refct++;
  *out = in;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectInherit_DefaultDestroy"
/*
    The default destroy treats it as a PETSc object and calls 
  its destroy routine.
*/
static int PetscObjectInherit_DefaultDestroy(void *in)
{
  int         ierr;
  PetscObject obj = (PetscObject) in;

  ierr = (*obj->destroypublic)(obj); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectReference"
/*@C
   PetscObjectReference - Indicates to any PetscObject that it is being
   referenced by another PetscObject. This increases the reference
   count for that object by one.

   Input Parameter:
.  obj - the PETSc object

.seealso: PetscObjectInherit(), PetscObjectDereference()

@*/
int PetscObjectReference(PetscObject obj)
{
  PetscValidHeader(obj);
  obj->refct++;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectDereference"
/*@
   PetscObjectDereference - Indicates to any PetscObject that it is being
   referenced by one less PetscObject. This decreases the reference
   count for that object by one.

   Input Parameter:
.  obj - the PETSc object

.seealso: PetscObjectInherit(), PetscObjectReference()

@*/
int PetscObjectDereference(PetscObject obj)
{
  int ierr;

  PetscValidHeader(obj);
  if (obj->destroypublic) {
    ierr = (*obj->destroypublic)(obj); CHKERRQ(ierr);
  } else if (--obj->refct == 0) {
    SETERRQ(1,0,"This PETSc object does not have a generic destroy routine");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectInherit"
/*@C
   PetscObjectInherit - Associates another object with a given PETSc object. 
                        This is to provide a limited support for inheritance.

   Input Parameters:
.  obj - the PETSc object
.  ptr - the other object to associate with the PETSc object
.  copy - a function used to copy the other object when the PETSc object 
          is copied, or PETSC_NULL to indicate the pointer is copied.
.  destroy - a function to call to destroy the object or PETSC_NULL to 
             call the standard destroy on the PETSc object.

   Notes:
   When ptr is a PetscObject one should almost always use PETSC_NULL as the 
   third and fourth argument.
   
   PetscObjectInherit() can be used with any PETSc object such at
   Mat, Vec, KSP, SNES, etc, or any user provided object. 

   Current limitation: 
   Each object can have only one child - we may extend this eventually.

.keywords: object, inherit

.seealso: PetscObjectGetChild()
@*/
int PetscObjectInherit(PetscObject obj,void *ptr, int (*copy)(void *,void **),int (*destroy)(void*))
{
/*
  if (obj->child) 
    SETERRQ(1,0,"Child already set;object can have only 1 child");
*/
  if (copy == PETSC_NULL)    copy = PetscObjectInherit_DefaultCopy;
  if (destroy == PETSC_NULL) destroy = PetscObjectInherit_DefaultDestroy;
  obj->child        = ptr;
  obj->childcopy    = copy;
  obj->childdestroy = destroy;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetChild"
/*@C
   PetscObjectGetChild - Gets the child of any PetscObject.

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameter:
.  type - the child, if it has been set (otherwise PETSC_NULL)

.keywords: object, get, child

.seealso: PetscObjectInherit()
@*/
int PetscObjectGetChild(PetscObject obj,void **child)
{
  PetscValidHeader(obj);

  *child = obj->child;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscDataTypeToMPIDataType"
int PetscDataTypeToMPIDataType(PetscDataType ptype,MPI_Datatype* mtype)
{
  if (ptype == PETSC_INT) {
    *mtype = MPI_INT;
  } else if (ptype == PETSC_DOUBLE) {
    *mtype = MPI_DOUBLE;
  } else if (ptype == PETSC_SCALAR) {
    *mtype = MPIU_SCALAR;
#if defined(PETSC_COMPLEX)
  } else if (ptype == PETSC_DCOMPLEX) {
    *mtype = MPIU_COMPLEX;
#endif
  } else if (ptype == PETSC_CHAR) {
    *mtype = MPI_CHAR;
  } else {
    SETERRQ(1,1,"Unknown PETSc datatype");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscDataTypeGetSize"
int PetscDataTypeGetSize(PetscDataType ptype,int *size)
{
  if (ptype == PETSC_INT) {
    *size = PETSC_INT_SIZE;
  } else if (ptype == PETSC_DOUBLE) {
    *size = PETSC_DOUBLE_SIZE;
  } else if (ptype == PETSC_SCALAR) {
    *size = PETSC_SCALAR_SIZE;
#if defined(PETSC_COMPLEX)
  } else if (ptype == PETSC_DCOMPLEX) {
    *size = PETSC_DCOMPLEX_SIZE;
#endif
  } else if (ptype == PETSC_CHAR) {
    *size = PETSC_CHAR_SIZE;
  } else {
    SETERRQ(1,1,"Unknown PETSc datatype");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscDataTypeGetName"
int PetscDataTypeGetName(PetscDataType ptype,char **name)
{
  if (ptype == PETSC_INT) {
    *name = "int";
  } else if (ptype == PETSC_DOUBLE) {
    *name = "double";
  } else if (ptype == PETSC_SCALAR) {
    *name = "Scalar";
#if defined(PETSC_COMPLEX)
  } else if (ptype == PETSC_DCOMPLEX) {
    *name = "complex";
#endif
  } else if (ptype == PETSC_CHAR) {
    *name = "char";
  } else {
    SETERRQ(1,1,"Unknown PETSc datatype");
  }
  return 0;
}



#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_linux
#define PETSC_ARCH_NAME "linux"

#define PETSC_HAVE_POPEN
#define PETSC_HAVE_LIMITS_H
#define PETSC_HAVE_PWD_H 
#define PETSC_HAVE_MALLOC_H 
#define PETSC_HAVE_STRING_H 
#define PETSC_HAVE_GETDOMAINNAME
#define PETSC_HAVE_DRAND48 
#define PETSC_HAVE_UNAME 
#define PETSC_HAVE_UNISTD_H 
#define PETSC_HAVE_SYS_TIME_H 
#define PETSC_HAVE_STDLIB_H
#define PETSC_HAVE_GETCWD
#define PETSC_HAVE_SLEEP
#define PETSC_HAVE_SYS_PARAM_H
#define PETSC_HAVE_SYS_STAT_H
#define PETSC_USE_IDB_DEBUGGER

#define PETSC_HAVE_FORTRAN_UNDERSCORE 

#define PETSC_HAVE_READLINK
#define PETSC_HAVE_MEMMOVE

#define PETSC_HAVE_DOUBLE_ALIGN_MALLOC
#define PETSC_HAVE_MEMALIGN
#define PETSC_HAVE_SYS_RESOURCE_H
#define PETSC_SIZEOF_VOID_P 4
#define PETSC_SIZEOF_INT 4
#define PETSC_SIZEOF_DOUBLE 8
#define PETSC_BITS_PER_BYTE 8
#define PETSC_SIZEOF_FLOAT 4
#define PETSC_SIZEOF_LONG 4
#define PETSC_SIZEOF_LONG_LONG 8

#define PETSC_HAVE_RTLD_GLOBAL 1

#define PETSC_HAVE_SYS_UTSNAME_H

#define PETSC_MISSING_SIGSYS

#ifdef PETSC_USE_MAT_SINGLE
#  define PETSC_MEMALIGN 16
#  define PETSC_HAVE_SSE "src/inline/gccsse.h"
#endif
#define PETSC_HAVE_CXX_NAMESPACE

#define PETSC_HAVE_F90_H "f90impl/f90_intel.h"
#define PETSC_HAVE_F90_C "src/sys/src/f90/f90_intel.c"

#define PETSC_USE_DYNAMIC_LIBRARIES 1

#define PETSC_DIR_SEPARATOR '/'
#define PETSC_PATH_SEPARATOR ':'
#define PETSC_REPLACE_DIR_SEPARATOR '\\'
#define PETSC_HAVE_SOCKET
#define PETSC_HAVE_FORK
#define PETSC_USE_32BIT_INT
#endif

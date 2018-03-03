#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>
#include "petsc.h"

//
//using namespace std;
//
//extern "C" void fortfunc_(double* arr_in, double* arr_out, int* n);
//extern "C" void fortfuncmat_(double* mat_in, double* mat_out, int* m, int* n);
//extern "C" double fortfuncsum_(int *fsize, double* fvec);
//
//int main(int argc, char *argv[])
//{
//    PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
//    PetscPrintf(PETSC_COMM_WORLD,"Hello World\n");
//
//	int m = 5;
//	int n = 5;
//	Eigen::ArrayXXd arr_in(m, n);
//	Eigen::ArrayXXd arr_out(m, n);
//
//	arr_in.fill(1.0);
//
//	auto&& aa = arr_out.row(0);
//
//	fortfunc_(
//		arr_in.row(0).data(),
//		aa.data(),
//		&n
//	);
//
//	std::cout << "\n Input is \n" << arr_in;
//	std::cout << "\n Result is \n" << arr_out;
//	std::cout << "\n Result is \n" << aa;
//
//	std::cout << "\n LAST TEST \n" << aa;
//
//	///rr_in.fill(1.0);
//	arr_in <<  1,  2,  3,  4,  5,
//			   6,  7,  8,  9, 10,
//			  11, 12, 13, 14, 15,
//			  16, 17, 18, 19, 20,
//			  21, 22, 23, 24, 25;
//
//	fortfuncmat_(
//		arr_in.data(),
//		arr_out.data(),
//		&m,
//		&n
//	);
//
//	std::cout << "\n Input is \n" << arr_in;
//	std::cout << "\n Result is \n" << arr_out;
//
//	std::cout << "\n Sum results is " << fortfuncsum_(&n, arr_out.row(0).data());
//
//    PetscFinalize();
//	return 0;
//}



static char help[] = "2d Bratu problem sequentially with SNES.\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular\n\
domain.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -use_fortran_function: use Fortran coded function, rather than C\n";

/*T
   Concepts: SNES Bratu example
   Processors: 1
T*/

/*
     Other useful options:

       -snes_mf : use matrix free operator and no preconditioner
       -snes_mf_operator : use matrix free operator but compute Jacobian via
                           finite differences to form preconditioner

*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation

            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.
    The uniprocessor version of this code is snes/examples/tutorials/ex4.c

  ------------------------------------------------------------------------- */

/*
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines   FormFunction().
*/
typedef struct {
  PetscReal param;             /* test problem parameter */
  int       mx,my;             /* discretization in x, y directions */
} AppCtx;


extern "C" void applicationfunctionfortran_(
		double* lambda,
		int* mx,
		int* my,
		double const* x,
		double* f,
		PetscErrorCode* ierr
		);
/*
   User-defined routines
*/
extern int FormFunction(SNES,Vec,Vec,void*),FormInitialGuess(AppCtx*,Vec);
extern int FormFunctionFortran(SNES,Vec,Vec,void*);

/*
    The main program is written in C while the user provided function
 is given in both Fortran and C. The main program could also be written
 in Fortran;
*/
int main(int argc, char **argv)
{
  SNES           snes;                /* nonlinear solver */
  Vec            x,r;                 /* solution, residual vectors */
  AppCtx         user;                /* user-defined work context */
  int            its;                 /* iterations for convergence */
  int            N,ierr,i,ii,ri,rj;
  ISColoringValue *colors;
  PetscErrorCode (*fnc)(SNES,Vec,Vec,void*);
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
  MatColoring 	 mat_coloring;
  MatFDColoring  fdcoloring;
  ISColoring     iscoloring;
  Mat            J;
  PetscScalar    zero = 0.0;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*
     Initialize problem parameters
  */
  user.mx = 400; user.my = 400; user.param = 6.0;
  ierr    = PetscOptionsGetInt(NULL,NULL,"-mx",&user.mx,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetInt(NULL,NULL,"-my",&user.my,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetReal(NULL,NULL,"-par",&user.param,NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) SETERRQ(PETSC_COMM_SELF,1,"Lambda is out of range");
  N = user.mx*user.my;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreateSeq(PETSC_COMM_WORLD, N, &x);
  ierr = VecDuplicate(x, &r);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-use_fortran_function",&flg);CHKERRQ(ierr);
  flg = PETSC_TRUE;
//  flg = PETSC_FALSE;
  if (flg) fnc = FormFunctionFortran;
  else     fnc = FormFunction;

  /*
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,fnc,&user);CHKERRQ(ierr);


  /*
     Create and set the nonzero pattern for the Jacobian: This is not done
     particularly efficiently.
       Note that for this code we use the "natural" number of the nodes on the
     grid (since that is what is good for the user provided function). In the
     DMDA examples we must use the DMDA numbering where each processor is assigned a
     chunk of data.
  */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, N, N, 5, NULL, &J);CHKERRQ(ierr);

  for (i = 0; i < N; ++i) {
	rj = i % user.mx;         /* column in grid */
	ri = i / user.mx;         /* row in grid */
	if (ri != 0) {     /* first row does not have neighbor below */
	  ii   = i - user.mx;
	  ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
	}
	if (ri != user.my - 1) { /* last row does not have neighbors above */
	  ii   = i + user.mx;
	  ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
	}
	if (rj != 0) {     /* first column does not have neighbor to left */
	  ii   = i - 1;
	  ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
	}
	if (rj != user.mx - 1) {     /* last column does not have neighbor to right */
	  ii   = i + 1;
	  ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
	}
	ierr = MatSetValues(J,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  /*
       Create the data structure that SNESComputeJacobianDefaultColor() uses
       to compute the actual Jacobians via finite differences.
  */
  ierr = MatColoringCreate(J, &mat_coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(mat_coloring, MATCOLORINGSL);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(mat_coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(mat_coloring, &iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetType(fdcoloring, MATMFFD_DS);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))fnc,&user);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(J,iscoloring,fdcoloring);CHKERRQ(ierr);
  /*
        Tell SNES to use the routine SNESComputeJacobianDefaultColor()
      to compute Jacobians.
  */
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&mat_coloring);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  // It is good to iterate at least once, otherwise the simulator
  // may not solve the problem at all if the first residual is
  // sufficiently small
  SNESSetForceIteration(snes, PETSC_TRUE);

  /*
     Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */

/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
int FormInitialGuess(AppCtx *user,Vec X)
{
  int         i,j,row,mx,my,ierr;
  PetscReal   one = 1.0,lambda,temp1,temp,hx,hy,hxdhy,hydhx,sc;
  PetscScalar *x;

  mx = user->mx;               my = user->my;        lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  temp1 = lambda/(lambda + one);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=0; j<my; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0;
        continue;
      }
      x[row] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp));
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
int FormFunction(SNES snes, Vec X, Vec F, void *ptr)
{
	AppCtx            *user = (AppCtx*)ptr;
	int               ierr,i,j,row,mx,my;
	PetscReal         two = 2.0,one = 1.0,lambda,hx,hy,hxdhy,hydhx,sc;
	PetscScalar       u,uxx,uyy,*f;
	const PetscScalar *x;

	mx = user->mx;            my = user->my;            lambda = user->param;
	hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
	sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

	/*
	 Get pointers to vector data
	*/
	ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
	ierr = VecGetArray(F,&f);CHKERRQ(ierr);

	/*
	 Compute function over the entire  grid
	*/
	for (j=0; j<my; j++) {
	for (i=0; i<mx; i++) {
	  row = i + j*mx;
	  if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
		f[row] = x[row];
		continue;
	  }
	  u      = x[row];
	  uxx    = (two*u - x[row-1] - x[row+1])*hydhx;
	  uyy    = (two*u - x[row-mx] - x[row+mx])*hxdhy;
	  f[row] = uxx + uyy - sc*PetscExpScalar(u);
	}
	}

	/*
	 Restore vectors
	*/
	ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
	ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

	ierr = PetscLogFlops(11.0*(mx-2)*(my-2));CHKERRQ(ierr);
	return 0;
}


/* ------------------------------------------------------------------- */
/*
   FormFunctionFortran - Evaluates nonlinear function, F(x) in Fortran.

*/
int FormFunctionFortran(SNES snes, Vec X, Vec F, void *ptr)
{
	AppCtx            *user = (AppCtx*)ptr;
	int               ierr;
	PetscScalar       *f;
	PetscScalar const *x;

	ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
	ierr = VecGetArray(F,&f);CHKERRQ(ierr);
	applicationfunctionfortran_(&user->param,&user->mx,&user->my,x,f,&ierr);
	ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
	ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
	ierr = PetscLogFlops(11.0*(user->mx-2)*(user->my-2));CHKERRQ(ierr);

	return 0;
}


//void create_petsc_matrix_from_scipy_csr(
//    int rows,
//    int cols,
//    EigenIntArray1dConstRef row_pointers,
//    EigenIntArray1dConstRef column_indices,
//    EigenArray1dConstRef values
//) {
//    this->jacobian_mat_csr.row_pointers = row_pointers;
//    this->jacobian_mat_csr.column_indices = column_indices;
//    this->jacobian_mat_csr.values = values;
//
//    MatCreateSeqAIJWithArrays(
//        PETSC_COMM_WORLD,
//        rows,
//        cols,
//        this->jacobian_mat_csr.row_pointers.data(),
//        this->jacobian_mat_csr.column_indices.data(),
//        this->jacobian_mat_csr.values.data(),
//        &(this->jacobian)
//    );
//
//    MatAssemblyBegin(this->jacobian, MAT_FINAL_ASSEMBLY);
//    MatAssemblyEnd(this->jacobian, MAT_FINAL_ASSEMBLY);
//
//    SNESSetJacobian(
//        this->snes,
//        this->jacobian,
//        this->jacobian,
//        NonLinearSolver::calculate_jacobian_matrix_wrapper,
//        this
//    );
//}

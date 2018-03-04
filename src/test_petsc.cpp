#include "petsc.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <stdlib.h>
#include <cassert>


namespace {

using namespace testing;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenMatCSR;

extern "C" {
void applicationfunctionfortran_(
		double* lambda,
		int* mx,
		int* my,
		double const* x,
		double* f,
		PetscErrorCode* ierr
	);

void initialguessfortran_(int* mx, int* my, double* lambda, double* x);
}

void create_petsc_matrix_from_eigen_csr(
	EigenMatCSR& mat_csr,
	Mat& jacobian
);


// Compressed Sparse Row Format
struct MatCSR {
	Eigen::Map<Eigen::ArrayXi> row_pointers;
	Eigen::Map<Eigen::ArrayXi> column_indices;
	Eigen::Map<Eigen::ArrayXd> values;

	MatCSR()
	: row_pointers(nullptr, 0)
	, column_indices(nullptr, 0)
	, values(nullptr, 0)
	{}
};

EigenMatCSR create_laplacian_matrix_1d(int size){
	EigenMatCSR mat_csr(size, size);

	mat_csr.insert(0, 0) = 1;
	mat_csr.insert(0, 1) = -1;
	for(int i = 1; i < size-1; ++i){
		mat_csr.insert(i, i - 1) = -1;
		mat_csr.insert(i, i) = 2;
		mat_csr.insert(i, i + 1) = -1;

	}
	mat_csr.insert(size-1, size-1) = 1;
	mat_csr.insert(size-1, size-2) = -1;

	mat_csr.makeCompressed();

	return mat_csr;
}

EigenMatCSR create_laplacian_matrix_2d(int mx, int my){

	auto N = mx * my;
	EigenMatCSR mat_csr(N, N);

	for (int i = 0; i < N; ++i)
	{
		auto rj = i % mx; // column in grid
		auto ri = i / mx; // row in grid
		if (ri != 0) {    // first row does not have neighbor below
			mat_csr.insert(i, i - mx) = 1;
		}
		if (ri != my - 1) { // last row does not have neighbors above
			mat_csr.insert(i, i + mx) = 1;
		}
		if (rj != 0) {     // first column does not have neighbor to left
			mat_csr.insert(i, i - 1) = 1;
		}
		if (rj != mx - 1) {     // last column does not have neighbor to right
			mat_csr.insert(i, i + 1) = 1;
		}
		mat_csr.insert(i, i) = 1;
	}

	mat_csr.makeCompressed();

	return mat_csr;
}

TEST(Eigen, Sparse)
{
	int N = 5;
	auto mat_csr = create_laplacian_matrix_1d(N);

	auto expected_nnz = 3 * (N - 2) + 4;

	auto nnz = mat_csr.nonZeros();
	auto row_pointers = mat_csr.outerIndexPtr();
	auto column_indices = mat_csr.innerIndexPtr();
	auto values = mat_csr.valuePtr();

	ASSERT_TRUE(nnz == expected_nnz);

	MatCSR mat_csr_struct;
	new (&mat_csr_struct.row_pointers) Eigen::Map<Eigen::ArrayXi>(row_pointers, N);
	new (&mat_csr_struct.column_indices) Eigen::Map<Eigen::ArrayXi>(column_indices, nnz);
	new (&mat_csr_struct.values) Eigen::Map<Eigen::ArrayXd>(values, nnz);

	Eigen::ArrayXi const expected_row_pointers = (Eigen::ArrayXi(5) << 0, 2, 5, 8, 11).finished();
	ASSERT_TRUE(mat_csr_struct.row_pointers.isApprox(expected_row_pointers));

	Eigen::ArrayXi const expected_column_indices = (Eigen::ArrayXi(13) << 0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4).finished();
	ASSERT_TRUE(mat_csr_struct.column_indices.isApprox(expected_column_indices));

	Eigen::ArrayXd const expected_values = (Eigen::ArrayXd(13) << 1, -1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 1).finished();
	ASSERT_TRUE(mat_csr_struct.values.isApprox(expected_values));

	Mat mat_petsc;
	create_petsc_matrix_from_eigen_csr(mat_csr, mat_petsc);
}

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines   FormFunction().
*/
typedef struct {
	PetscReal param;             /* test problem parameter */
	int       mx,my;             /* discretization in x, y directions */
} AppCtx;


// User-defined routines
int FormFunction(SNES,Vec,Vec,void*);
int FormFunctionFortran(SNES,Vec,Vec,void*);
void FormInitialGuess(AppCtx*,Vec, bool);
void initial_guess(int mx, int my, double lambda, double* x);


TEST(Impl, FortranVSCpp)
{
	auto mx = 5;
	auto my = 5;
	auto lambda = 4.75;
	auto N = mx * my;
	Eigen::ArrayXd x_cpp(N);
	Eigen::ArrayXd x_for(N);

	initialguessfortran_(&mx, &my, &lambda, x_for.data());
	initial_guess(mx, my, lambda, x_cpp.data());

	ASSERT_TRUE(x_cpp.isApprox(x_for));
}

class PetscTest : public ::testing::TestWithParam<PetscBool> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
};
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
TEST_P(PetscTest, SNES)
{
	SNES             snes;                /* nonlinear solver */
	Vec              x,r;                 /* solution, residual vectors */
	AppCtx           user;                /* user-defined work context */
	int              its;                 /* iterations for convergence */
	int              N,ierr,i,ii,ri,rj;
	ISColoringValue* colors;
	PetscReal        bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
	MatColoring  	 mat_coloring;
	MatFDColoring    fdcoloring;
	ISColoring       iscoloring;
	Mat              J;
	PetscScalar      zero = 0.0;
	PetscBool        use_fortran;
	PetscErrorCode   (*fnc)(SNES,Vec,Vec,void*);

	/*
	 Initialize problem parameters
	*/
	user.mx = 5; user.my = 5; user.param = 6.0;

	ASSERT_FALSE(user.param >= bratu_lambda_max || user.param <= bratu_lambda_min); // Lambda is out of range
	N = user.mx*user.my;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Create nonlinear solver context
	 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	SNESCreate(PETSC_COMM_WORLD, &snes);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Create vector data structures; set function evaluation routine
	 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	VecCreateSeq(PETSC_COMM_WORLD, N, &x);
	VecDuplicate(x, &r);

	use_fortran = GetParam();
	if (use_fortran) fnc = FormFunctionFortran;
	else     fnc = FormFunction;

	/*
	 Set function evaluation routine and vector
	*/
	SNESSetFunction(snes,r,fnc,&user);

	auto mat_csr = create_laplacian_matrix_2d(user.mx, user.my);
	create_petsc_matrix_from_eigen_csr(mat_csr, J);
	/*
	   Create the data structure that SNESComputeJacobianDefaultColor() uses
	   to compute the actual Jacobians via finite differences.
	*/
	MatColoringCreate(J, &mat_coloring);
	MatColoringSetType(mat_coloring, MATCOLORINGSL);
	MatColoringSetFromOptions(mat_coloring);
	MatColoringApply(mat_coloring, &iscoloring);
	MatFDColoringCreate(J,iscoloring,&fdcoloring);
	MatFDColoringSetType(fdcoloring, MATMFFD_DS);
	MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))fnc,&user);
	MatFDColoringSetFromOptions(fdcoloring);
	MatFDColoringSetUp(J,iscoloring,fdcoloring);
	/*
		Tell SNES to use the routine SNESComputeJacobianDefaultColor()
	  to compute Jacobians.
	*/
	SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);
	ISColoringDestroy(&iscoloring);
	MatColoringDestroy(&mat_coloring);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Customize nonlinear solver; set runtime options
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	// It is good to iterate at least once, otherwise the simulator
	// may not solve the problem at all if the first residual is
	// sufficiently small
	SNESSetForceIteration(snes, PETSC_TRUE);

    SNESSetTolerances(snes, 1e-8, 1e-50, 1e-50, 10, 1e8);

	/*
	 Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
	*/
	SNESSetFromOptions(snes);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Evaluate initial guess; then solve nonlinear system
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	/*
	 Note: The user should initialize the vector, x, with the initial guess
	 for the nonlinear solver prior to calling SNESSolve().  In particular,
	 to employ an initial guess of zero, the user should explicitly set
	 this vector to zero by calling VecSet().
	*/
	FormInitialGuess(&user, x, use_fortran);
	SNESSolve(snes,NULL,x);
	SNESGetIterationNumber(snes,&its);
	PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);

	SNESConvergedReason reason;
	SNESGetConvergedReason(snes, &reason);
	ASSERT_TRUE(reason > 0);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	 Free work space.  All PETSc objects should be destroyed when they
	 are no longer needed.
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	VecDestroy(&x);
	VecDestroy(&r);
	MatFDColoringDestroy(&fdcoloring);
	MatDestroy(&J);
	SNESDestroy(&snes);
} // TEST_P

INSTANTIATE_TEST_CASE_P(
		FotranTest,
		PetscTest,
		::testing::Values(PETSC_TRUE)
);
INSTANTIATE_TEST_CASE_P(
		CppTest,
		PetscTest,
		::testing::Values(PETSC_FALSE)
);

void FormInitialGuess(AppCtx *user, Vec X, bool use_fortran)
{
	PetscScalar *x;

	auto mx = user->mx;
	auto my = user->my;
	auto lambda = user->param;

	VecGetArray(X,&x);

	initial_guess(mx, my, lambda, x);
	if(use_fortran){
		initialguessfortran_(&mx, &my, &lambda, x);
	}
	else{
		initial_guess(mx, my, lambda, x);
	}

	VecRestoreArray(X,&x);
}

void initial_guess(int mx, int my, double lambda, double* x)
{
	auto hx = 1.0 / (double)(mx - 1.0);
	auto hy = 1.0 / (double)(my - 1.0);
	auto temp1 = lambda / (lambda + 1.0);
	for (int j = 0; j < my; ++j)
	{
		auto temp = double(std::min(j, my - j - 1) * hy);
		for (int i = 0; i < mx; ++i)
		{
			auto row = i + j * mx;
			if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
				x[row] = 0.0;
				continue;
			}
			x[row] = temp1 * std::sqrt( std::min(std::min(i, mx - i - 1) * hx, temp));
		}
	}
}

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
	VecGetArrayRead(X,&x);
	VecGetArray(F,&f);

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
	VecRestoreArrayRead(X,&x);
	VecRestoreArray(F,&f);

	return 0;
}

// FormFunctionFortran - Evaluates nonlinear function, F(x) in Fortran.
int FormFunctionFortran(SNES snes, Vec X, Vec F, void *ptr)
{
	AppCtx            *user = (AppCtx*)ptr;
	int               ierr;
	PetscScalar       *f;
	PetscScalar const *x;

	VecGetArrayRead(X,&x);
	VecGetArray(F,&f);
	applicationfunctionfortran_(&user->param,&user->mx,&user->my,x,f,&ierr);
	VecRestoreArrayRead(X,&x);
	VecRestoreArray(F,&f);

	return 0;
}


void create_petsc_matrix_from_eigen_csr(
	EigenMatCSR& mat_csr,
	Mat& jacobian
)
{
	auto n_rows = mat_csr.rows();
	auto n_cols = mat_csr.cols();
	auto nnz = mat_csr.nonZeros();

	auto row_pointers = mat_csr.outerIndexPtr();
	auto column_indices = mat_csr.innerIndexPtr();
	auto values = mat_csr.valuePtr();

	MatCSR jacobian_mat_csr;
	new (&jacobian_mat_csr.row_pointers) Eigen::Map<Eigen::ArrayXi>(row_pointers, n_rows);
	new (&jacobian_mat_csr.column_indices) Eigen::Map<Eigen::ArrayXi>(column_indices, nnz);
	new (&jacobian_mat_csr.values) Eigen::Map<Eigen::ArrayXd>(values, nnz);

    MatCreateSeqAIJWithArrays(
        PETSC_COMM_WORLD,
		n_rows,
		n_cols,
		jacobian_mat_csr.row_pointers.data(),
		jacobian_mat_csr.column_indices.data(),
		jacobian_mat_csr.values.data(),
        &jacobian
    );

    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
}

} // namespace

subroutine fortfunc(arr_in, arr_out, n)
    implicit none
    integer ::  n
    real*8  ::  arr_in(n)
    real*8  ::  arr_out(n)

    arr_out = arr_in + 1.23 * exp(arr_in / 4.56)

    return
end subroutine fortfunc

real*8 function fortfuncsum(fsize, fvec)
    implicit none
    integer fsize,i
    real*8 fvec(fsize)

    fortfuncsum = 0.0

    do i = 1,fsize
        fortfuncsum = fortfuncsum + fvec(i)
    end do

    return

end function fortfuncsum

subroutine fortfuncmat(mat_in, mat_out, m, n)
    implicit none
    integer             ::  m, n
    real*8, intent(in)  ::  mat_in(m, n)
    real*8              ::  mat_out(m, n)


    mat_out(:, :) = 0.9999 + mat_in
    return
end subroutine fortfuncmat


!
! ---------------------------------------------------------------------
!
!  Fortran version of the user function based on shared memory
!  this routine is called only by MPI process 0 in the computation
!  but uses threads to run the loops in parallel.

!
!  Input Parameter:
!  x - global array containing input values
!
!  Output Parameters:
!  f - global array containing output values
!  ierr - error code
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim array.
!
#include <petsc/finclude/petsc.h>
subroutine ApplicationFunctionFortran(lambda,mx,my,x,f,ierr)
    use petscsnes
    implicit none

    integer  ierr,mx,my

    !  Input/output variables:
    PetscScalar   x(mx,my),f(mx,my),lambda

    !  Local variables:
    PetscScalar   two,one,hx,hy,hxdhy,hydhx,sc
    PetscScalar   u,uxx,uyy
    integer  i,j

    hx     = 1.0 / PetscIntToReal(mx-1)
    hy     = 1.0 / PetscIntToReal(my-1)
    sc     = hx * hy * lambda
    hxdhy  = hx / hy
    hydhx  = hy / hx

    !  Compute function over the entire grid

!    do j = 1, my
!        do i = 1, mx
!            if (i .eq. 1 .or. j .eq. 1 .or. i .eq. mx .or. j .eq. my) then
!                f(i,j) = x(i,j)
!            else
!                u = x(i,j)
!                uxx = hydhx * (2.0 * u - x(i-1,j) - x(i+1,j))
!                uyy = hxdhy * (2.0 * u - x(i,j-1) - x(i,j+1))
!                f(i,j) = uxx + uyy - sc * exp(u)
!            endif
!        end do
!    end do

    f(1, :) = x(1, :)
    f(mx,:) = x(mx,:)
    f(:, 1) = x(:, 1)
    f(:,my) = x(:,my)

    f(2:mx-1,2:my-1) = hydhx * (2.0 * x(2:mx-1,2:my-1) - x(1:mx-2,2:my-1) - x(3:mx,2:my-1)) &
                     + hxdhy * (2.0 * x(2:mx-1,2:my-1) - x(2:mx-1,1:my-2) - x(2:mx-1,3:my)) &
                     - sc * exp(x(2:mx-1,2:my-1))

    return
end

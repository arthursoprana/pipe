#include "petsc.h"
#include "gtest/gtest.h"

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0, NULL);if (ierr) return ierr;

  auto result = RUN_ALL_TESTS();
  ierr = PetscFinalize();CHKERRQ(ierr);
  return result;
}

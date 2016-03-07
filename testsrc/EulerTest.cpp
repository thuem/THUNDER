/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Euler.h"

int main(int argc, const char* argv[])
{
    vec3 axis = {0, 0, 1};

    mat33 mat;
    alignZ(mat, axis);
    mat.print("alignZ");

    return 0;
}

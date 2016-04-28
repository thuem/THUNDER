/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Euler.h"

#include <gtest/gtest.h>

int main(int argc, const char* argv[])
{
    vec3 axis = {0, 0, 1};

    mat33 mat;
    alignZ(mat, axis);

    /***
    EXPECT_TRUE(approx_equal(mat,
                             mat33({{1, 0, 0},
                                    {0, 1, 0},
                                    {0, 0, 1}}),
                             "absdiff",
                             0.001));
                             ***/

    return 0;
}

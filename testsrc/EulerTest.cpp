/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Euler.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    vec3 axis = {0, 0, 1};

    mat33 mat;
    alignZ(mat, axis);
#if 0
    if (!approx_equal(mat,
                             mat33({{1, 0, 0},
                                    {0, 1, 0},
                                    {0, 0, 1}}),
                             "absdiff",
                             0.001));
    {
        cout << "Test failed\n";
        return 1;
    }
#endif
    return 0;
}

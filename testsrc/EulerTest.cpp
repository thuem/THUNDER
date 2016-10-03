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
    /***
    vec3 axis = {0, 0, 1};

    mat33 mat;
    alignZ(mat, axis);
    ***/

    mat33 rot;
    randRotate3D(rot);

    cout << rot << endl;

    return 0;
}

/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Euler.h"

#define QUATERNION_MATRIX_TEST

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
#ifdef QUATERNION_MATRIX_TEST
    mat33 rot;
    randRotate3D(rot);

    std::cout << "rot = \n" << rot << std::endl;

    vec4 quat;
    quaternion(quat, rot);

    std::cout << "quat = \n" << quat << std::endl;

    std::cout << "norm of quat = \n" << quat.norm() << std::endl;

    rotate3D(rot, quat);

    std::cout << "rot = \n" << rot << std::endl;
#endif

    return 0;
}

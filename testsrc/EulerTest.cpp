/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Euler.h"

#include "Functions.h"

//#define QUATERNION_MATRIX_TEST

#define QUATERNION_MUL_TEST

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

#ifdef QUATERNION_MUL_TEST

    mat33 rot1, rot2;
    randRotate3D(rot1);
    randRotate3D(rot2);

    std::cout << "rot1 = \n" << rot1 << std::endl;
    std::cout << "rot2 = \n" << rot2 << std::endl;

    vec4 quat1, quat2;
    quaternion(quat1, rot1);
    quaternion(quat2, rot2);

    std::cout << "quat1 = \n" << quat1 << std::endl;
    std::cout << "quat2 = \n" << quat2 << std::endl;

    std::cout << "rot2 * rot1 = \n" << rot2 * rot1 << std::endl;

    vec4 quat;
    quaternion_mul(quat, quat2, quat1);

    std::cout << "quat2 * quat1 = \n" << quat << std::endl;

    mat33 rot;

    rotate3D(rot, quat);

    std::cout << "rotation matix of (quat2 * quat1) = \n" << rot << std::endl;

#endif

    return 0;
}

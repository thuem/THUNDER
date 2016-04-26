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

    /***
    cout << atof(argv[1]) << endl;
    cout << atof(argv[2]) << endl;
    cout << atof(argv[3]) << endl;
    ***/

    /***
    double a = atof(argv[1]);
    double b = atof(argv[2]);
    double c = atof(argv[3]);

    vec4 u = {a, b, c, sqrt(1 - a * a - b * b - c * c)};

    mat33 rot;
    rotate3D(rot, u);

    cout << rot << endl;

    double phi, theta, psi;
    angle(phi, theta, psi, u);

    cout << phi << endl
         << theta << endl
         << psi << endl
         << endl;
    ***/

    /***
    rotate3D(rot, phi, theta, psi);

    cout << rot << endl;
    ***/

    /***
    angle(phi, theta, psi, rot);

    cout << phi << endl
         << theta << endl
         << psi << endl
         << endl;

    quaternoin(u, phi, theta, psi);
    cout << u << endl;
    ***/

    /***
    rotate3D(rot, phi, theta, psi);

    cout << rot << endl;
    ***/

    return 0;
}

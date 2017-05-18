#include <iostream>

#include "Particle.h"

INITIALIZE_EASYLOGGINGPP

#define N 10000

//#define RAND_QUATERNION

//#define PARTICLE_TEST_3D

#define TEST_PARTICLE_LOAD

using namespace std;

int main()
{
#ifdef RAND_QUATERNION
    vec4 quat;

    randQuaternion(quat);

    std::cout << quat << std::endl;
#endif

#ifdef PARTICLE_TEST_3D
    //Symmetry sym("C15");
    Symmetry sym("C2");

    Particle par(MODE_3D, 1, N, 5, 0.01, &sym);

    vec4 quat;
    for (int i = 0; i < N; i++)
    {
        par.quaternion(quat, i);

        /***
        printf("%15.6lf %15.6lf %15.6lf %15.6lf\n",
               quat(0),
               quat(1),
               quat(2),
               quat(3));
        ***/

        //vec4 r(0, 1, 0, 0);
        
        vec4 r;

        quaternion_mul(r, quat, ANCHOR_POINT);
        quaternion_mul(r, r, quaternion_conj(quat));

        printf("%15.6lf %15.6lf %15.6lf\n",
               r(1),
               r(2),
               r(3));
    }

#endif

#ifdef TEST_PARTICLE_LOAD

    Symmetry sym("C1");

    Particle par(MODE_3D, 1, N, 5, 0.01, &sym);

    double stdR = 0.01;

    double stdTX = 1;

    double stdTY = 1;

    double stdD = 0;

    par.load(N,
             0,
             vec4(1, 0, 0, 0),
             stdR,
             vec2(0, 0),
             stdTX,
             stdTY,
             1,
             stdD);

    save("TEST_PARTICLE_LOAD.par", par);

    par.calVari();

    par.vari(stdR, stdTX, stdTY, stdD);

    printf("stdR = %lf, stdTX = %lf, stdTY = %lf, stdD = %lf\n",
           stdR,
           stdTX,
           stdTY,
           stdD);

#endif

    return 0;
}

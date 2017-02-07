#include <iostream>

#include "Particle.h"

INITIALIZE_EASYLOGGINGPP

#define N 100

using namespace std;

int main()
{
    /***
    Particle par(MODE_2D, 1, 100, 5);

    display(par);
    ***/

    /***
    mat4 r(N, 4);

    sampleVMS(r, vec4(1, 0, 0, 0), 0, N);

    cout << r << endl;
    ***/

    double x, y, z;

    gsl_rng* engine = get_random_engine();

    //gsl_ran_dir_3d(engine, &x, &y, &z);

    cout << x << ", " << y << ", " << z << endl;

    return 0;
}

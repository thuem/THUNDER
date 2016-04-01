/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

#define N 10000

int main(int argc, const char* argv[])
{
    Symmetry sym("C5");

    Particle particle;
    particle.init(N, 30, 30, &sym);
    // Particle particle(N, 30, 30, &sym);

    vec4 u;
    double v[4];

    double nt = N / 3;
    /***
    bingham_t B;
    bingham_new_S3(&B, e0, e1, e2, -30, -30, 0);
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        particle.normW();
        // cout << "neff = " << particle.neff() << endl;
        if (particle.neff() < nt)
            particle.resample();
        else particle.perturb();
        for (int j = 0; j < N; j++)
        {
            particle.quaternion(u, j);
            v[0] = u(0);
            v[1] = u(1);
            v[2] = u(2);
            v[3] = u(3);
            particle.setW(particle.w(j) * bingham_pdf(v, &B), j);
        }
    }

    bingham_free(&B);
    ***/

    display(particle);
}

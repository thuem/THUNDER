/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

#define M 10000

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    Symmetry sym("C2");

    Particle particle;
    particle.init(M, 30, 30, &sym);

    save("Initial.par", particle);

    vec4 u;
    double v[4];

    double nt = M / 3;
    /***
    bingham_t B;
    bingham_new_S3(&B, e0, e1, e2, -30, -30, 0);
    ***/

    char filename[FILE_NAME_LENGTH];
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        cout << "neff = " << particle.neff() << endl;
        if (particle.neff() < nt)
        {
            particle.resample(0.7);
            //particle.resample(GSL_MAX_INT(100, particle.n() / 2));
        }
        else particle.perturb();

        for (int j = 0; j < M; j++)
        {
            /***
            particle.mulW(1, j);
            ***/
            particle.quaternion(u, j);
            v[0] = u(0);
            v[1] = u(1);
            v[2] = u(2);
            v[3] = u(3);
            /***
            particle.mulW(bingham_pdf(v, &B), j);
            ***/
        }

        particle.normW();

        sprintf(filename, "Round_%04d.par", i);
        save(filename, particle);
    }

    /***
    bingham_free(&B);
    ***/
}

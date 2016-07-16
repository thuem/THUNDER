/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

#define M 5000

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    Symmetry sym("C2");

    Particle particle;
    particle.init(M, 2, 0.01, &sym);

    save("Initial.par", particle);

    vec4 u;

    double nt = M / 3;

    char filename[FILE_NAME_LENGTH];
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        cout << "neff = " << particle.neff() << endl;

        if (particle.neff() < nt)
        {
            cout << "Resampling" << endl;
            particle.resample();
        }
        
        particle.perturb();

        for (int j = 0; j < M; j++)
        {
            particle.quaternion(u, j);
            particle.mulW(pdfACG(u, 25, 1), j);
        }

        particle.normW();

        sprintf(filename, "Round_%04d.par", i);
        save(filename, particle);
    }
}

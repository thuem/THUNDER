/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Projector.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Experiment.h"
#include "Spectrum.h"

#define PF 1

#define N 380
#define M 40000
#define MAX_X 2
#define MAX_Y 2

#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS_U 2e4
#define DEFOCUS_V 2e4
#define THETA 0
#define CS 0

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    FFT fft;

    cout << "Read-in Ref" << endl;
    Volume ref;
    ImageFile imf("ref.mrc", "r");
    imf.readMetaData();
    imf.readVolume(ref);

    cout << "Checkout Size" << endl;
    if ((ref.nColRL() != N) ||
        (ref.nRowRL() != N))
        cout << "Wrong Size!" << endl;

    cout << "Padding Head" << endl;
    Volume padRef;
    VOL_PAD_RL(padRef, ref, PF);
    normalise(padRef);

    cout << "Writing padRef" << endl;
    imf.readMetaData(padRef);
    imf.writeVolume("padRef.mrc", padRef);

    /***
    cout << "Reading from Hard-disk" << endl;
    ImageFile imf2("padHead.mrc", "rb");
    imf2.readMetaData();
    imf2.readVolume(padHead);
    ***/
    
    cout << "Fourier Transforming Ref" << endl;
    fft.fw(padRef);
    fft.fw(ref);

    cout << "Power Spectrum" << endl;
    vec ps = vec::Zero(N);
    powerSpectrum(ps, ref, N / 2 - 1);

    cout << "Setting Projectee" << endl;
    Projector projector;
    projector.setPf(PF);
    projector.setProjectee(padRef.copyVolume());

    cout << "Setting CTF" << endl;
    Image ctf(N, N, FT_SPACE);
    CTF(ctf,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS_U,
        DEFOCUS_V,
        THETA,
        CS);

    cout << "Initialising Experiment" << endl;
    Experiment exp("C15.db");
    exp.createTables();
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U, DEFOCUS_V, THETA, CS);
    exp.appendGroup("");

    char name[256];

    Image image(N, N, FT_SPACE);
    
    cout << "Initialising Random Sampling Points" << endl;
    Symmetry sym("C15");
    Particle par(M, MAX_X, MAX_Y, &sym);
    cout << "Saving Sampling Points" << endl;
    save("Sampling_Points.par", par);

    Coordinate5D coord;
    auto engine = get_random_engine();
    for (int i = 0; i < M; i++)
    {
        SET_0_FT(image);

        sprintf(name, "%05d.mrc", i + 1);
        printf("%s\n", name);

        par.coord(coord, i);
        projector.project(image, coord);

        FOR_EACH_PIXEL_FT(image)
            image[i] *= REAL(ctf[i]);

        Image noise(N, N, FT_SPACE);
        SET_0_FT(noise);
        IMAGE_FOR_EACH_PIXEL_FT(noise)
            if (QUAD(i, j) < pow(N / 2 - 1, 2))
                noise.setFT(COMPLEX(gsl_ran_gaussian(engine, 10 * sqrt(ps(AROUND(NORM(i, j))))),
                                    gsl_ran_gaussian(engine, 10 * sqrt(ps(AROUND(NORM(i, j)))))),
                            i,
                            j);

        ADD_FT(image, noise);

        fft.bw(image);

        printf("image: mean = %f, stddev = %f, maxValue = %f\n",
               gsl_stats_mean(&image(0), 1, image.sizeRL()),
               gsl_stats_sd(&image(0), 1, image.sizeRL()),
               image(cblas_idamax(image.sizeRL(), &image(0), 1)));
        
        exp.appendParticle(name, 1, 1);

        imf.readMetaData(image);
        imf.writeImage(name, image);

        fft.fw(image);
    }
    
    return 0;
}

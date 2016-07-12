/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <omp.h>

#include <gsl/gsl_statistics.h>

#include "Projector.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Experiment.h"
#include "Spectrum.h"

#define PF 2

#define N 380
//#define M 40000
#define M 5000
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

    Volume cylinder(N, N, N, RL_SPACE);
    SET_0_RL(cylinder);

    CLOG(INFO, "LOGGER_SYS") << "Generate Cylinder" << endl;
    VOLUME_FOR_EACH_PIXEL_RL(cylinder)
        if ((NORM(i, j) < 75.0 / PIXEL_SIZE) &&
            (abs(k) < 100))
            cylinder.setRL(1, i, j, k);

    ImageFile imfCylinder;

    CLOG(INFO, "LOGGER_SYS") << "Write Cylinder" << endl;
    imfCylinder.readMetaData(cylinder);
    imfCylinder.writeVolume("cylinder.mrc", cylinder);

    CLOG(INFO, "LOGGER_SYS") << "Pad Cylinder" << endl;
    Volume padCylinder;
    VOL_PAD_RL(padCylinder, cylinder, PF);

    CLOG(INFO, "LOGGER_SYS") << "Write padCylinder" << endl;
    imfCylinder.readMetaData(padCylinder);
    imfCylinder.writeVolume("padCylinder.mrc", padCylinder);

    CLOG(INFO, "LOGGER_SYS") << "Read-in Ref" << endl;

    Volume ref;
    ImageFile imf("ref.mrc", "r");
    imf.readMetaData();
    imf.readVolume(ref);

    CLOG(INFO, "LOGGER_SYS") << "Max = " << gsl_stats_max(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Min = " << gsl_stats_min(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Mean = " << gsl_stats_mean(&ref(0), 1, ref.sizeRL());

    /***
    FOR_EACH_PIXEL_RL(ref)
        if (ref(i) < 0) ref(i) = 0;

    CLOG(INFO, "LOGGER_SYS") << "Max = " << gsl_stats_max(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Min = " << gsl_stats_min(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Mean = " << gsl_stats_mean(&ref(0), 1, ref.sizeRL());
    ***/

    CLOG(INFO, "LOGGER_SYS") << "Checkout Size" << endl;
    if ((ref.nColRL() != N) ||
        (ref.nRowRL() != N))
        CLOG(INFO, "LOGGER_SYS") << "Wrong Size!" << endl;

    CLOG(INFO, "LOGGER_SYS") << "Padding Head" << endl;
    Volume padRef;
    VOL_PAD_RL(padRef, ref, PF);
    //normalise(padRef);

    CLOG(INFO, "LOGGER_SYS") << "Writing padRef" << endl;
    imf.readMetaData(padRef);
    imf.writeVolume("padRef.mrc", padRef);

    /***
    CLOG(INFO, "LOGGER_SYS") << "Reading from Hard-disk" << endl;
    ImageFile imf2("padHead.mrc", "rb");
    imf2.readMetaData();
    imf2.readVolume(padHead);
    ***/
    
    CLOG(INFO, "LOGGER_SYS") << "Fourier Transforming Ref" << endl;
    fft.fw(padRef);
    fft.fw(ref);

    CLOG(INFO, "LOGGER_SYS") << "Sum of ref = " << REAL(ref[0]) << endl;
    CLOG(INFO, "LOGGER_SYS") << "Sum of padRef = " << REAL(padRef[0]) << endl;

    CLOG(INFO, "LOGGER_SYS") << "Power Spectrum" << endl;
    vec ps = vec::Zero(N);
    powerSpectrum(ps, ref, N / 2 - 1);

    CLOG(INFO, "LOGGER_SYS") << "Setting Projectee" << endl;
    Projector projector;
    projector.setPf(PF);
    projector.setProjectee(padRef.copyVolume());

    CLOG(INFO, "LOGGER_SYS") << "Setting CTF" << endl;
    Image ctf(N, N, FT_SPACE);
    CTF(ctf,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS_U,
        DEFOCUS_V,
        THETA,
        CS);

    CLOG(INFO, "LOGGER_SYS") << "Initialising Experiment" << endl;
    Experiment exp("C15.db");
    exp.createTables();
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U, DEFOCUS_V, THETA, CS);
    exp.appendGroup("");

    CLOG(INFO, "LOGGER_SYS") << "Initialising Random Sampling Points" << endl;
    Symmetry sym("C15");
    Particle par(M, MAX_X, MAX_Y, &sym);
    CLOG(INFO, "LOGGER_SYS") << "Saving Sampling Points" << endl;
    save("Sampling_Points.par", par);

    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        auto engine = get_random_engine();

        char name[256];
        Coordinate5D coord;

        Image image(N, N, FT_SPACE);
        SET_0_FT(image);

        sprintf(name, "%05d.mrc", i + 1);
        printf("%s\n", name);

        par.coord(coord, i);
        projector.project(image, coord);

        FOR_EACH_PIXEL_FT(image)
            image[i] *= REAL(ctf[i]);

        /***
        Image noise(N, N, FT_SPACE);
        SET_0_FT(noise);
        IMAGE_FOR_EACH_PIXEL_FT(noise)
            if (QUAD(i, j) < pow(N / 2 - 1, 2))
                noise.setFT(COMPLEX(gsl_ran_gaussian(engine, 10 * sqrt(ps(AROUND(NORM(i, j))))),
                                    gsl_ran_gaussian(engine, 10 * sqrt(ps(AROUND(NORM(i, j)))))),
                            i,
                            j);

        ADD_FT(image, noise);
        ***/

        fft.bw(image);

        printf("image: mean = %f, stddev = %f, maxValue = %f\n",
               gsl_stats_mean(&image(0), 1, image.sizeRL()),
               gsl_stats_sd(&image(0), 1, image.sizeRL()),
               image(cblas_idamax(image.sizeRL(), &image(0), 1)));
        
        #pragma omp critical
        exp.appendParticle(name, 1, 1);

        imf.readMetaData(image);
        imf.writeImage(name, image);

        fft.fw(image);
    }

    
    return 0;
}

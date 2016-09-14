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
#include "Mask.h"

#define PF 2

#define N 380
//#define M 40000
#define M 5000
#define TRANS_S 2

#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS_U_1 2e4
#define DEFOCUS_V_1 2e4
#define DEFOCUS_U_2 1.8e4
#define DEFOCUS_V_2 1.8e4
#define DEFOCUS_U_3 1.7e4
#define DEFOCUS_V_3 1.7e4
#define DEFOCUS_U_4 1.5e4
#define DEFOCUS_V_4 1.5e4
#define THETA 0
#define CS 0

#define NOISE_FACTOR 10

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    FFT fft;

    Volume cylinder(N, N, N, RL_SPACE);
    SET_0_RL(cylinder);

    CLOG(INFO, "LOGGER_SYS") << "Generate Cylinder";
    VOLUME_FOR_EACH_PIXEL_RL(cylinder)
        if ((NORM(i, j) < 75.0 / PIXEL_SIZE) &&
            (abs(k) < 70))
            cylinder.setRL(1, i, j, k);

    ImageFile imfCylinder;

    CLOG(INFO, "LOGGER_SYS") << "Write Cylinder";
    imfCylinder.readMetaData(cylinder);
    imfCylinder.writeVolume("cylinder.mrc", cylinder);

    CLOG(INFO, "LOGGER_SYS") << "Pad Cylinder";
    Volume padCylinder;
    VOL_PAD_RL(padCylinder, cylinder, PF);

    CLOG(INFO, "LOGGER_SYS") << "Write padCylinder";
    imfCylinder.readMetaData(padCylinder);
    imfCylinder.writeVolume("padCylinder.mrc", padCylinder);

    CLOG(INFO, "LOGGER_SYS") << "Read-in Ref";

    Volume ref;
    ImageFile imf("ref.mrc", "r");
    imf.readMetaData();
    imf.readVolume(ref);

    CLOG(INFO, "LOGGER_SYS") << "Checkout Size";
    if ((ref.nColRL() != N) ||
        (ref.nRowRL() != N))
        CLOG(INFO, "LOGGER_SYS") << "Wrong Size!";

    CLOG(INFO, "LOGGER_SYS") << "Max = " << gsl_stats_max(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Min = " << gsl_stats_min(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Mean = " << gsl_stats_mean(&ref(0), 1, ref.sizeRL());

    double bg = background(ref, N * 1.2 / 2, EDGE_WIDTH_RL);

    CLOG(INFO, "LOGGER_SYS") << "Background of Reference = " << bg;

    CLOG(INFO, "LOGGER_SYS") << "Truncated Ref";

    FOR_EACH_PIXEL_RL(ref)
        if (ref(i) < 0) ref(i) = 0;

    CLOG(INFO, "LOGGER_SYS") << "Max = " << gsl_stats_max(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Min = " << gsl_stats_min(&ref(0), 1, ref.sizeRL());
    CLOG(INFO, "LOGGER_SYS") << "Mean = " << gsl_stats_mean(&ref(0), 1, ref.sizeRL());

    imf.readMetaData(ref);
    imf.writeVolume("truncRef.mrc", ref);

    CLOG(INFO, "LOGGER_SYS") << "Padding Head";
    Volume padRef;
    VOL_PAD_RL(padRef, ref, PF);
    //normalise(padRef);

    CLOG(INFO, "LOGGER_SYS") << "Writing padRef";
    imf.readMetaData(padRef);
    imf.writeVolume("padRef.mrc", padRef);

    /***
    CLOG(INFO, "LOGGER_SYS") << "Reading from Hard-disk";
    ImageFile imf2("padHead.mrc", "rb");
    imf2.readMetaData();
    imf2.readVolume(padHead);
    ***/
    
    CLOG(INFO, "LOGGER_SYS") << "Fourier Transforming Ref";
    fft.fw(padRef);
    fft.fw(ref);

    CLOG(INFO, "LOGGER_SYS") << "Sum of ref = " << REAL(ref[0]);
    CLOG(INFO, "LOGGER_SYS") << "Sum of padRef = " << REAL(padRef[0]);

    CLOG(INFO, "LOGGER_SYS") << "Setting Projectee";
    Projector projector;
    projector.setPf(PF);
    projector.setProjectee(padRef.copyVolume());

    CLOG(INFO, "LOGGER_SYS") << "Setting CTF";
    Image ctf_1(N, N, FT_SPACE);
    CTF(ctf_1,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS_U_1,
        DEFOCUS_V_1,
        THETA,
        CS);
    Image ctf_2(N, N, FT_SPACE);
    CTF(ctf_2,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS_U_2,
        DEFOCUS_V_2,
        THETA,
        CS);
    Image ctf_3(N, N, FT_SPACE);
    CTF(ctf_3,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS_U_3,
        DEFOCUS_V_3,
        THETA,
        CS);
    Image ctf_4(N, N, FT_SPACE);
    CTF(ctf_4,
        PIXEL_SIZE,
        VOLTAGE,
        DEFOCUS_U_4,
        DEFOCUS_V_4,
        THETA,
        CS);

    CLOG(INFO, "LOGGER_SYS") << "Initialising Experiment";
    Experiment exp("C15.db");
    exp.createTables();
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U_1, DEFOCUS_V_1, THETA, CS);
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U_2, DEFOCUS_V_2, THETA, CS);
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U_3, DEFOCUS_V_3, THETA, CS);
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U_4, DEFOCUS_V_4, THETA, CS);
    exp.appendGroup("");
    exp.appendGroup("");
    exp.appendGroup("");
    exp.appendGroup("");

    CLOG(INFO, "LOGGER_SYS") << "Initialising Random Sampling Points";
    Symmetry sym("C15");
    Particle par(M, TRANS_S, 0.01, &sym);
    cout << "Saving Sampling Points" << endl;
    save("Sampling_Points.par", par);

    Image image(N, N, FT_SPACE);
    SET_0_FT(image);
    
    Coordinate5D coord;
    par.coord(coord, 0);
    
    projector.project(image, coord);
    fft.bw(image);

    double std = gsl_stats_sd(&image(0), 1, image.sizeRL());

    #pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        // CLOG(INFO, "LOGGER_SYS") << "Power Spectrum";

        FFT fftThread;

        auto engine = get_random_engine();

        char name[256];
        Coordinate5D coord;

        Image image(N, N, FT_SPACE);
        SET_0_FT(image);

        sprintf(name, "%05d.mrc", i + 1);
        printf("%s\n", name);

        par.coord(coord, i);
        projector.project(image, coord);

        /***
        vec ps = vec::Zero(N);
        powerSpectrum(ps, image, N / 2 - 1);
        ***/

        if (i % 4 == 0)
        {
            FOR_EACH_PIXEL_FT(image)
                image[i] *= REAL(ctf_1[i]);
        }
        else if (i % 4 == 1)
        {
            FOR_EACH_PIXEL_FT(image)
                image[i] *= REAL(ctf_2[i]);
        }
        else if (i % 4 == 2)
        {
            FOR_EACH_PIXEL_FT(image)
                image[i] *= REAL(ctf_3[i]);
        }
        else
        {
            FOR_EACH_PIXEL_FT(image)
                image[i] *= REAL(ctf_4[i]);
        }

        /***
        Image noise(N, N, FT_SPACE);
        SET_0_FT(noise);
        IMAGE_FOR_EACH_PIXEL_FT(noise)
            if (QUAD(i, j) < pow(N / 2 - 1, 2))
                noise.setFT(COMPLEX(gsl_ran_gaussian(engine, 1 * sqrt(ps(AROUND(NORM(i, j))))),
                                    gsl_ran_gaussian(engine, 1 * sqrt(ps(AROUND(NORM(i, j)))))),
                            i,
                            j);
                            ***/

        //ADD_FT(image, noise);

        fftThread.bw(image);

        Image noise(N, N, RL_SPACE);
        FOR_EACH_PIXEL_RL(noise)
            noise(i) = gsl_ran_gaussian(engine, NOISE_FACTOR * std);

        ADD_RL(image, noise);

        /***
        printf("image: mean = %f, stddev = %f, maxValue = %f\n",
               gsl_stats_mean(&image(0), 1, image.sizeRL()),
               gsl_stats_sd(&image(0), 1, image.sizeRL()),
               image(cblas_idamax(image.sizeRL(), &image(0), 1)));
        ***/
        
        if (i % 4 == 0)
        {
            #pragma omp critical
            exp.appendParticle(name, 1, 1);
        }
        else if (i % 4 == 1)
        {
            #pragma omp critical
            exp.appendParticle(name, 2, 2);
        }
        else if (i % 4 == 2)
        {
            #pragma omp critical
            exp.appendParticle(name, 3, 3);
        }
        else
        {
            #pragma omp critical
            exp.appendParticle(name, 4, 4);
        }

        ImageFile imfThread;

        imfThread.readMetaData(image);
        imfThread.writeImage(name, image);

        fftThread.fw(image);
    }
    
    return 0;
}

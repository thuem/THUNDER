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
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Experiment.h"
#include "Spectrum.h"
#include "Mask.h"

#define PF 2

#define N 380
#define M 5000
#define TRANS_S 2

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

    MPI_Init(&argc, &argv);

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    FFT fft;

    CLOG(INFO, "LOGGER_SYS") << "Initialising Random Sampling Points";
    Symmetry sym("C15");
    Particle par(M, TRANS_S, 0.01, &sym);

    if (commRank == MASTER_ID)
    {
        CLOG(INFO, "LOGGER_SYS") << "Read-in Ref";

        Volume ref;
        ImageFile imf("ref.mrc", "r");
        imf.readMetaData();
        imf.readVolume(ref);

        CLOG(INFO, "LOGGER_SYS") << "Checkout Size";

        if ((ref.nColRL() != N) ||
            (ref.nRowRL() != N))
            CLOG(INFO, "LOGGER_SYS") << "Wrong Size!";

        FOR_EACH_PIXEL_RL(ref)
            if (ref(i) < 0) ref(i) = 0;

        imf.readMetaData(ref);
        imf.writeVolume("truncRef.mrc", ref);

        CLOG(INFO, "LOGGER_SYS") << "Padding Head";

        Volume padRef;
        VOL_PAD_RL(padRef, ref, PF);

        CLOG(INFO, "LOGGER_SYS") << "Writing padRef";

        imf.readMetaData(padRef);
        imf.writeVolume("padRef.mrc", padRef);

        CLOG(INFO, "LOGGER_SYS") << "Fourier Transforming Ref";
        fft.fw(padRef);
        fft.fw(ref);

        CLOG(INFO, "LOGGER_SYS") << "Setting Projectee";
        Projector projector;
        projector.setPf(PF);
        projector.setProjectee(padRef.copyVolume());

        CLOG(INFO, "LOGGER_SYS") << "Setting CTF";
        Image ctf(N, N, FT_SPACE);

        CTF(ctf,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_U,
            DEFOCUS_V,
            THETA,
            CS);


    /***
    par.coord(coord, 0);
    
    projector.project(image, coord);
    fft.bw(image);

    double std = gsl_stats_sd(&image(0), 1, image.sizeRL());
    ***/

        #pragma omp parallel for
        for (int i = 0; i < M; i++)
        {
            FFT fftThread;

            auto engine = get_random_engine();

            char name[256];
            Coordinate5D coord;

            Image image(N, N, FT_SPACE);
            SET_0_FT(image);

            sprintf(name, "%05d.mrc", i + 1);

            par.coord(coord, i);
            projector.project(image, coord);

            /***
            FOR_EACH_PIXEL_FT(image)
                image[i] *= REAL(ctf[i]);
            ***/

            fftThread.bw(image);

        /***
        Image noise(N, N, RL_SPACE);
        FOR_EACH_PIXEL_RL(noise)
            noise(i) = gsl_ran_gaussian(engine, std);

        ADD_RL(image, noise);
        ***/

            ImageFile imfThread;

            imfThread.readMetaData(image);
            imfThread.writeImage(name, image);

            fftThread.fw(image);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (commRank == MASTER_ID)
        CLOG(INFO, "LOGGER_SYS") << "Projection Done!";

    Reconstructor reco(N, 2, &sym);
    reco.setMPIEnv();

    if (commRank == MASTER_ID)
        CLOG(INFO, "LOGGER_SYS") << "Reconstructor Set!";

    MPI_Barrier(MPI_COMM_WORLD);

    if (commRank == HEMI_A_LEAD)
    {
        char nameInsert[256];

        Coordinate5D coord;

        for (int i = 0; i < M; i++)
        {
            sprintf(nameInsert, "%05d.mrc", i + 1);

            CLOG(INFO, "LOGGER_SYS") << "Inserting " << nameInsert;

            Image insert(N, N, RL_SPACE);

            ImageFile imfInsert(nameInsert, "rb");
            imfInsert.readMetaData();
            imfInsert.readImage(insert);

            CLOG(INFO, "LOGGER_SYS") << nameInsert << "Read!";

            fft.fw(insert);

            CLOG(INFO, "LOGGER_SYS") << nameInsert << "Fourier transformed";

            par.coord(coord, i);

            reco.insert(insert, coord, 1);

            CLOG(INFO, "LOGGER_SYS") << nameInsert << "Inserted!";
        }

        CLOG(INFO, "LOGGER_SYS") << "Reconstructing!";

        Volume result;
        reco.reconstruct(result);

        CLOG(INFO, "LOGGER_SYS") << "Saving Result!";

        ImageFile imf;

        imf.readMetaData(result);
        imf.writeVolume("result.mrc", result);
    }
    
    return 0;
}

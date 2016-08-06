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

    Symmetry sym("C15");

    Projector projector;

    if (commRank == MASTER_ID)
    {
        CLOG(INFO, "LOGGER_SYS") << "Initialising Random Sampling Points";
        //Particle par(M, TRANS_S, 0.01, &sym);
        Particle par(M, TRANS_S, 0.01, NULL);
        save("SamplingPoints.par", par);

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
        fft.fwMT(padRef);
        fft.fwMT(ref);

        CLOG(INFO, "LOGGER_SYS") << "padRef[0]" << padRef[0];

        CLOG(INFO, "LOGGER_SYS") << "Setting Projectee";
        projector.setPf(PF);
        projector.setProjectee(padRef.copyVolume());

        /***
        CLOG(INFO, "LOGGER_SYS") << "Setting CTF";
        Image ctf(N, N, FT_SPACE);

        CTF(ctf,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_U,
            DEFOCUS_V,
            THETA,
            CS);
        ***/

        Coordinate5D coord;
        Image image(N, N, FT_SPACE);
        SET_0_FT(image);

        par.coord(coord, 0);
    
        projector.project(image, coord);
        fft.bw(image);

        double std = gsl_stats_sd(&image(0), 1, image.sizeRL());

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

            Image noise(N, N, RL_SPACE);
            FOR_EACH_PIXEL_RL(noise)
                noise(i) = gsl_ran_gaussian(engine, std);

            ADD_RL(image, noise);

            ImageFile imfThread;

            imfThread.readMetaData(image);
            imfThread.writeImage(name, image);

            fftThread.fw(image);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (commRank == MASTER_ID)
        CLOG(INFO, "LOGGER_SYS") << "Projection Done!";

    //Reconstructor reco(N, 2, &sym);
    Reconstructor reco(N, 2, NULL);
    reco.setMPIEnv();

    if (commRank == MASTER_ID)
        CLOG(INFO, "LOGGER_SYS") << "Reconstructor Set!";

    MPI_Barrier(MPI_COMM_WORLD);

    if (commRank != MASTER_ID)
    {
        CLOG(INFO, "LOGGER_SYS") << "Loading Sampling Points";
        Particle par(M, TRANS_S, 0.01, &sym);
        load(par, "SamplingPoints.par");

        char nameInsert[256];

        Coordinate5D coord;

        for (int i = M / (commSize - 1) * (commRank - 1);
                 i < M / (commSize - 1) * commRank;
                 i++)
        {
            FFT fftThread;

            sprintf(nameInsert, "%05d.mrc", i + 1);

            CLOG(INFO, "LOGGER_SYS") << "Inserting " << nameInsert;

            Image insert(N, N, RL_SPACE);

            ImageFile imfInsert(nameInsert, "rb");
            imfInsert.readMetaData();
            imfInsert.readImage(insert);

            CLOG(INFO, "LOGGER_SYS") << nameInsert << " Read!";

            fftThread.fw(insert);

            CLOG(INFO, "LOGGER_SYS") << nameInsert << " Fourier transformed";

            par.coord(coord, i);

            reco.insert(insert, coord, 1);

            CLOG(INFO, "LOGGER_SYS") << nameInsert << " Inserted!";
        }

        CLOG(INFO, "LOGGER_SYS") << "Reconstructing!";

        Volume result;
        reco.reconstruct(result);

        if (commRank == HEMI_A_LEAD)
        {
            CLOG(INFO, "LOGGER_SYS") << "Saving Result!";

            Volume resultA;
            VOL_EXTRACT_RL(resultA, result, 0.5);

            ImageFile imf;

            imf.readMetaData(resultA);
            imf.writeVolume("resultA.mrc", resultA);
        }

        if (commRank == HEMI_B_LEAD)
        {
            CLOG(INFO, "LOGGER_SYS") << "Saving Result!";

            Volume resultB;
            VOL_EXTRACT_RL(resultB, result, 0.5);

            ImageFile imf;

            imf.readMetaData(resultB);
            imf.writeVolume("resultB.mrc", resultB);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (commRank == MASTER_ID)
        CLOG(INFO, "LOGGER_SYS") << "Compare!";

    if (commRank == MASTER_ID)
    {
        Volume newRef;

        CLOG(INFO, "LOGGER_SYS") << "Loading Constructed Reference";

        ImageFile imfNew("resultA.mrc", "rb");
        imfNew.readMetaData();
        imfNew.readVolume(newRef);

        CLOG(INFO, "LOGGER_SYS") << "Padding Constructed Reference";

        Volume padNewRef;
        VOL_PAD_RL(padNewRef, newRef, PF);

        CLOG(INFO, "LOGGER_SYS") << "Fourier Transforming Constructed Reference";

        fft.fwMT(padNewRef);

        CLOG(INFO, "LOGGER_SYS") << "padNewRef[0]" << padNewRef[0];

        CLOG(INFO, "LOGGER_SYS") << "Setting Projector on Constructed Reference";
        
        Projector projectorNew;
        projectorNew.setPf(PF);
        projectorNew.setProjectee(padNewRef.copyVolume());

        CLOG(INFO, "LOGGER_SYS") << "Loading Sampling Points";
        //Particle par(M, TRANS_S, 0.01, &sym);
        Particle par(M, TRANS_S, 0.01, NULL);
        load(par, "SamplingPoints.par");

        #pragma omp parallel for
        for (int i = 0; i < 100; i++)
        {
            FFT fftThread;

            char name[256];

            Coordinate5D coord;

            Image image(N, N, RL_SPACE);
            sprintf(name, "%05d.mrc", i + 1);
            ImageFile imfOri(name, "rb");
            imfOri.readMetaData();
            imfOri.readImage(image);
            fftThread.fw(image);

            Image imageNew(N, N, FT_SPACE);
            SET_0_FT(imageNew);

            par.coord(coord, i);

            projectorNew.project(imageNew, coord);

            Image diff(N, N, FT_SPACE);
            FOR_EACH_PIXEL_FT(diff)
                diff[i] = image[i] - imageNew[i];

            CLOG(INFO, "LOGGER_SYS") << "ORI[0] = "
                                     << REAL(image[0])
                                     << ", NEW[0] = "
                                     << REAL(imageNew[0])
                                     << ", DIFF[0] = "
                                     << REAL(diff[0]);

            fftThread.bw(diff);

            sprintf(name, "DIFF_%05d.bmp", i + 1);
            diff.saveRLToBMP(name);

            fftThread.bw(image);

            sprintf(name, "ORI_%05d.bmp", i + 1);
            image.saveRLToBMP(name);

            fftThread.bw(imageNew);

            sprintf(name, "NEW_%05d.bmp", i + 1);
            imageNew.saveRLToBMP(name);
        }
    }
    
    return 0;
}

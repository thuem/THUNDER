/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <omp_compat.h>

#include <gsl/gsl_statistics.h>

#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Spectrum.h"
#include "Mask.h"

#define PF 2

#define N 160
#define M 5000
//#define M 40000
//#define M 10
#define TRANS_S 2

#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS_1 1.3e4
#define DEFOCUS_2 1.4e4
#define DEFOCUS_3 1.5e4
#define DEFOCUS_4 1.6e4
#define DEFOCUS_5 1.8e4
#define DEFOCUS_6 2.0e4
#define DEFOCUS_7 2.1e4
#define DEFOCUS_8 2.2e4
#define THETA 0
#define CS 0

#define BLOB_A 1.9

#define NOISE_FACTOR 1

#define TEST_2D
//#define TEST_3D

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

#ifdef TEST_2D

    MPI_Init(&argc, &argv);

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

#endif

#ifdef TEST_3D

    MPI_Init(&argc, &argv);

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    FFT fft;

    Symmetry sym("C4");

        Image ctf1(N, N, FT_SPACE);
        Image ctf2(N, N, FT_SPACE);
        Image ctf3(N, N, FT_SPACE);
        Image ctf4(N, N, FT_SPACE);
        Image ctf5(N, N, FT_SPACE);
        Image ctf6(N, N, FT_SPACE);
        Image ctf7(N, N, FT_SPACE);
        Image ctf8(N, N, FT_SPACE);

        CTF(ctf1,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_1,
            DEFOCUS_1,
            THETA,
            CS);

        CTF(ctf2,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_2,
            DEFOCUS_2,
            THETA,
            CS);

        CTF(ctf3,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_3,
            DEFOCUS_3,
            THETA,
            CS);

        CTF(ctf4,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_4,
            DEFOCUS_4,
            THETA,
            CS);

        CTF(ctf5,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_5,
            DEFOCUS_5,
            THETA,
            CS);

        CTF(ctf6,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_6,
            DEFOCUS_6,
            THETA,
            CS);

        CTF(ctf7,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_7,
            DEFOCUS_7,
            THETA,
            CS);

        CTF(ctf8,
            PIXEL_SIZE,
            VOLTAGE,
            DEFOCUS_8,
            DEFOCUS_8,
            THETA,
            CS);

    Projector projector;

    if (commRank == MASTER_ID)
    {
        CLOG(INFO, "LOGGER_SYS") << "Initialising Random Sampling Points";
        Particle par(M, 1, TRANS_S, 0.01, &sym);
        //Particle par(M, TRANS_S, 0.01, NULL);
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

        CLOG(INFO, "LOGGER_SYS") << "padRef[0]" << REAL(padRef[0]);

        CLOG(INFO, "LOGGER_SYS") << "Setting Projectee";
        projector.setPf(PF);
        projector.setProjectee(padRef.copyVolume());

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

            gsl_rng* engine = get_random_engine();

            char name[256];
            Coordinate5D coord;

            Image image(N, N, FT_SPACE);
            SET_0_FT(image);

            sprintf(name, "%05d.mrc", i + 1);

            par.coord(coord, i);
            projector.project(image, coord);

            if (i % 8 == 0)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf1[i]);
            }
            else if (i % 8 == 1)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf2[i]);
            }
            else if (i % 8 == 2)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf3[i]);
            }
            else if (i % 8 == 3)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf4[i]);
            }
            else if (i % 8 == 4)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf5[i]);
            }
            else if (i % 8 == 5)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf6[i]);
            }
            else if (i % 8 == 6)
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf7[i]);
            }
            else
            {
                FOR_EACH_PIXEL_FT(image)
                    image[i] *= REAL(ctf8[i]);
            }

            fftThread.bw(image);

            Image noise(N, N, RL_SPACE);
            FOR_EACH_PIXEL_RL(noise)
                noise(i) = gsl_ran_gaussian(engine, NOISE_FACTOR * std);

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

    Reconstructor reco(N, 2, &sym, BLOB_A);
    //Reconstructor reco(N, 2, NULL, 0.95);
    //Reconstructor reco(N, 2, &sym);
    //Reconstructor reco(N, 2, NULL, 0.95);
    //Reconstructor reco(N, 2, &sym, 0.95);
    //Reconstructor reco(N, 2, NULL);
    reco.setMPIEnv();
    //reco.setMaxRadius(33);

    if (commRank == MASTER_ID)
        CLOG(INFO, "LOGGER_SYS") << "Reconstructor Set!";

    MPI_Barrier(MPI_COMM_WORLD);

    if (commRank != MASTER_ID)
    {
        CLOG(INFO, "LOGGER_SYS") << "Loading Sampling Points";
        Particle par(M, 1, TRANS_S, 0.01, &sym);
        load(par, "SamplingPoints.par");

        char nameInsert[256];

        Coordinate5D coord;

        CLOG(INFO, "LOGGER_SYS") << "Setting CTF";

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

            /***
            if (i % 8 == 0)
                reco.insert(insert, ctf1, coord, 1);
            else if (i % 8 == 1)
                reco.insert(insert, ctf2, coord, 1);
            else if (i % 8 == 2)
                reco.insert(insert, ctf3, coord, 1);
            else if (i % 8 == 3)
                reco.insert(insert, ctf4, coord, 1);
            else if (i % 8 == 4)
                reco.insert(insert, ctf5, coord, 1);
            else if (i % 8 == 5)
                reco.insert(insert, ctf6, coord, 1);
            else if (i % 8 == 6)
                reco.insert(insert, ctf7, coord, 1);
            else
                reco.insert(insert, ctf8, coord, 1);
            ***/

            CLOG(INFO, "LOGGER_SYS") << nameInsert << " Inserted!";
        }

        CLOG(INFO, "LOGGER_SYS") << "Reconstructing!";

        Volume result;
        reco.reconstruct(result);

        if (commRank == HEMI_A_LEAD)
        {
            CLOG(INFO, "LOGGER_SYS") << "Saving Result!";

            Volume resultA;
            //VOL_EXTRACT_RL(resultA, result, 0.5);
            VOL_EXTRACT_RL(resultA, result, 1);

            ImageFile imf;

            imf.readMetaData(resultA);
            imf.writeVolume("resultA.mrc", resultA);
        }

        if (commRank == HEMI_B_LEAD)
        {
            CLOG(INFO, "LOGGER_SYS") << "Saving Result!";

            Volume resultB;
            //VOL_EXTRACT_RL(resultB, result, 0.5);
            VOL_EXTRACT_RL(resultB, result, 1);

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

        CLOG(INFO, "LOGGER_SYS") << "padNewRef[0]" << REAL(padNewRef[0]);

        CLOG(INFO, "LOGGER_SYS") << "Setting Projector on Constructed Reference";
        
        Projector projectorNew;
        projectorNew.setPf(PF);
        projectorNew.setProjectee(padNewRef.copyVolume());

        CLOG(INFO, "LOGGER_SYS") << "Loading Sampling Points";
        Particle par(1, M, TRANS_S, 0.01, &sym);
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

            if (i % 8 == 0)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf1[i]);
            }
            else if (i % 8 == 1)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf2[i]);
            }
            else if (i % 8 == 2)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf3[i]);
            }
            else if (i % 8 == 3)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf4[i]);
            }
            else if (i % 8 == 4)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf5[i]);
            }
            else if (i % 8 == 5)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf6[i]);
            }
            else if (i % 8 == 6)
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf7[i]);
            }
            else
            {
                FOR_EACH_PIXEL_FT(image)
                    imageNew[i] *= REAL(ctf8[i]);
            }

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

#endif
    
    return 0;
}

/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Experiment.h"

#define PF 2

#define N 128
#define M 1000
#define MAX_X 10
#define MAX_Y 10

#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS_U 2e4
#define DEFOCUS_V 2e4
#define THETA 0
#define CS 0

using namespace std;

int main(int argc, char* argv[])
{
    FFT fft;

    cout << "Defining Head" << endl;
    Volume head(N, N, N, RL_SPACE);
    VOLUME_FOR_EACH_PIXEL_RL(head)
    {
        if ((NORM_3(i, j, k) < N / 8) ||
            (NORM_3(i - N / 8, j, k - N / 8) < N / 16) ||
            (NORM_3(i + N / 8, j, k - N / 8) < N / 16) ||
            ((NORM(i, j) < N / 16) &&
             (k + N / 16 < 0) &&
             (k + 3 * N / 16 > 0)))
            head.setRL(1, i, j, k);
        else
            head.setRL(0, i, j, k);
    }
    normalise(head);

    /***
    cout << "Adding Noise" << endl;
    Volume noise(N, N, N, RL_SPACE);
    FOR_EACH_PIXEL_RL(noise)
        noise(i) = gsl_ran_gaussian(RANDR, 5);
    ADD_RL(head, noise);
    normalise(head);
    ***/

    printf("head: mean = %f, stddev = %f, maxValue = %f\n",
           gsl_stats_mean(&head(0), 1, head.sizeRL()),
           gsl_stats_sd(&head(0), 1, head.sizeRL()),
           head(cblas_idamax(head.sizeRL(), &head(0), 1)));

    /***
    fft.fw(head);
    fft.bw(head);

    printf("head: mean = %f, stddev = %f\n",
           gsl_stats_mean(&head(0), 1, head.sizeRL()),
           gsl_stats_sd(&head(0), 1, head.sizeRL()));
    ***/
           
    cout << "Padding Head" << endl;
    Volume padHead;
    VOL_PAD_RL(padHead, head, PF);
    printf("padHead: mean = %f, stddev = %f, maxValue = %f\n",
           gsl_stats_mean(&padHead(0), 1, padHead.sizeRL()),
           gsl_stats_sd(&padHead(0), 1, padHead.sizeRL()),
           padHead(cblas_idamax(padHead.sizeRL(), &padHead(0), 1)));

    cout << "Fourier Transforming Head" << endl;
    fft.fw(padHead);

    cout << "Setting Projectee" << endl;
    Projector projector;
    projector.setPf(PF);
    projector.setProjectee(padHead);

    cout << "Setting CTF" << endl;
    Image ctf(N, N, FT_SPACE);
    CTF(ctf,
        VOLTAGE,
        PIXEL_SIZE,
        DEFOCUS_U,
        DEFOCUS_V,
        THETA,
        CS);

    cout << "Initialising Experiment" << endl;
    Experiment exp("MickeyMouse.db");
    exp.createTables();
    exp.appendMicrograph("", VOLTAGE, DEFOCUS_U, DEFOCUS_V, THETA, CS, 1);
    exp.appendGroup("");

    char name[256];

    Image image(N, N, FT_SPACE);
    // Image image(N, N, RL_SPACE);
    
    ImageFile imf;

    cout << "Initialising Random Sampling Points" << endl;
    Particle par(M, MAX_X, MAX_Y);

    Coordinate5D coord;
    for (int i = 0; i < M; i++)
    {
        SET_0_FT(image);

        sprintf(name, "%06d.mrc", i);
        printf("%s\n", name);

        par.coord(coord, i);
        projector.project(image, coord);

        MUL_FT(image, ctf);

        Image noise(N, N, RL_SPACE);
        FOR_EACH_PIXEL_RL(noise)
            noise(i) = gsl_ran_gaussian(RANDR, 2);

        fft.bw(image);

        ADD_RL(image, noise);

        printf("image: mean = %f, stddev = %f, maxValue = %f\n",
               gsl_stats_mean(&image(0), 1, image.sizeRL()),
               gsl_stats_sd(&image(0), 1, image.sizeRL()),
               image(cblas_idamax(image.sizeRL(), &image(0), 1)));
        
        exp.appendParticle(name, 1, 1);

        imf.readMetaData(image);
        imf.writeImage(name, image);

        sprintf(name, "%06d.bmp", i);
        image.saveRLToBMP(name);

        fft.fw(image);
    }
    
    return 0;
}

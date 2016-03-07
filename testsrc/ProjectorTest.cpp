/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>


#include "Projector.h"
#include "ImageFile.h"
#include "FFT.h"

#define N 256
#define M 8

int main(int argc, const char* argv[])
{
    std::cout << "Define a head." << std::endl;

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

    ImageFile imf;
    imf.readMetaData(head);
    imf.writeVolume("head.mrc", head);
    
    FFT fft;
    fft.fw(head);

    /***
    VOLUME_FOR_EACH_PIXEL_FT(head)
        printf("%f %f\n", REAL(head.getFT(i, j, k)),
                          IMAG(head.getFT(i, j, k)));
    ***/

    Projector projector;
    projector.setProjectee(head);

    char name[256];
    int counter = 0;

    // Image image(N, N, fourierSpace);
    Image image(N, N, RL_SPACE);

    for (int k = 0; k < M; k++)
        for (int j = 0; j < M; j++)
            for (int i = 0; i < M; i++)
            {
                printf("%02d %02d %02d\n", i, j, k);
                sprintf(name, "%02d%02d%02d.bmp", i, j, k);
                /***
                projector.project(image,
                                                2 * M_PI * i / M,
                                                M_PI * j / M,
                                                2 * M_PI * k / M);
                                          ***/
                /***
                FFT fft;
                fft.fw(image);
                fft.bw(image);
                ***/
                /***
                R2R_FT(image, sin(2));
                ***/
                R2R_FT(image, image, projector.project(image,
                                                2 * M_PI * i / M,
                                                M_PI * j / M,
                                                2 * M_PI * k / M));

                image.saveRLToBMP(name);
                // image.saveFTToBMP(name, 0.1);
            }

    return 0;
}

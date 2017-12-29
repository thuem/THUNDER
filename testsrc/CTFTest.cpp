//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "CTF.h"
#include "Spectrum.h"

#define N 380



/***
#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS_U 2e4
#define DEFOCUS_V 2e4
#define THETA 0
#define CS 0
***/



INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    RFLOAT pixelSize = 1.32;
    RFLOAT voltage = 3e5;
    RFLOAT defocusU = atof(argv[1]);
    /*
     *RFLOAT theta = 0;
     */
    RFLOAT Cs = 0;

    // std::cout << "200 Angstrom = " << resA2P(1.0 / 200, N, pixelSize) << std::endl;

    /***
    Image img(N, N, FT_SPACE);

    CTF(img, pixelSize, voltage, defocusU, defocusV, theta, Cs);

    img.saveFTToBMP("CTF.bmp", 0.1);

    for (int i = 1; i < img.nColRL() / 2; i++)
    {
        std::cout << "Pixel = "
             << i - 1
             << std::endl
             << "Resolution = "
             << 1.0 / resP2A(i, img.nColRL(), pixelSize)
             << std::endl
             << "CTF = "
             << ringAverage(i, img, [](const Complex x){ return REAL(x); })
             << std::endl;
    }
    ***/

    for (RFLOAT i = 0.01; i < N / 2; i += 0.01)
    {
        RFLOAT f = i / (pixelSize * N);

        printf("%12.6f    %12.6f    %12.6f\n",
               i,
               pixelSize * N / i,
               CTF(f, voltage, defocusU, Cs));
    }

    return 0;
}

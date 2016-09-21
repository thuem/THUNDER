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

using namespace std;

/***
#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS_U 2e4
#define DEFOCUS_V 2e4
#define THETA 0
#define CS 0
***/

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    double pixelSize = 1.32;
    double voltage = 3e5;
    double defocusU = atof(argv[1]);
    double theta = 0;
    double Cs = 0;

    /***
    Image img(N, N, FT_SPACE);

    CTF(img, pixelSize, voltage, defocusU, defocusV, theta, Cs);

    img.saveFTToBMP("CTF.bmp", 0.1);

    for (int i = 1; i < img.nColRL() / 2; i++)
    {
        cout << "Pixel = "
             << i - 1
             << endl
             << "Resolution = "
             << 1.0 / resP2A(i, img.nColRL(), pixelSize)
             << endl
             << "CTF = "
             << ringAverage(i, img, [](const Complex x){ return REAL(x); })
             << endl;
    }
    ***/

    for (int i = 0; i < N / 2; i++)
    {
        double f = i / (pixelSize * N);

        printf("%04d    %12.6f\n",
               i,
               CTF(f, voltage, defocusU, Cs));
    }

    return 0;
}

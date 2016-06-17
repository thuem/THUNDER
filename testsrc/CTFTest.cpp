/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "CTF.h"

#define N 1024

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    double pixelSize = 1.32;
    double voltage = 3e5;
    double defocusU = 20000;
    double defocusV = 20000;
    double theta = 0;
    double Cs = 0;

    Image img(N, N, FT_SPACE);

    CTF(img, pixelSize, voltage, defocusU, defocusV, theta, Cs);

    img.saveFTToBMP("CTF.bmp", 0.1);

    return 0;
}

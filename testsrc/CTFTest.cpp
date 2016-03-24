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

int main(int argc, const char* argv[])
{
    double pixelSize = 1.32;
    double voltage = 3e5;
    double defocusU = 20000;
    double defocusV = 20000;
    double theta = 0;
    double Cs = 0;

    Image img(N, N, FT_SPACE);

    CTF(img, pixelSize, voltage, defocusU, defocusV, theta, Cs);

    Image realPart(N, N, FT_SPACE);
    IMAGE_FOR_EACH_PIXEL_FT(img)
        realPart.setFT(COMPLEX(REAL(img.getFT(i, j)), 0), i, j);

    realPart.saveFTToBMP("realPart.bmp", 0.1);

    return 0;
}

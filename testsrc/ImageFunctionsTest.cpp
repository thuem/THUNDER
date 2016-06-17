/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "ImageFile.h"
#include "FFT.h"

#include "ImageFunctions.h"

#define N 380

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    Image image(N, N, RL_SPACE);

    IMAGE_FOR_EACH_PIXEL_RL(image)
    {
        if (QUAD(i, j) < N * N / 16)
            image.setRL(1, i, j);
        else
            image.setRL(0, i, j);
    }

    cout << "****** transalte ******" << endl;

    R2R_FT(image, image, translate(image, image, 0, 0));
    image.saveRLToBMP("trans0.bmp");

    R2R_FT(image, image, translate(image, image, N / 8, N / 8));
    image.saveRLToBMP("trans1.bmp");

    R2R_FT(image, image, translate(image, image, N / 4, N / 4));
    image.saveRLToBMP("trans2.bmp");

    cout << "****** translate ******" << endl;

    FFT fft;
    fft.fw(image);

    Image trans(N, N, FT_SPACE);
    translate(trans, image, N / 8, N / 8);

    int nTransCol, nTransRow;

    cout << "Translating, " << N / 8 << ", " << N / 8 << endl;

    translate(nTransCol, nTransRow, image, trans, N / 2, N / 4, N / 4);

    cout << "nTransCol = " << nTransCol << ", nTranRow = " << nTransRow << endl;
}

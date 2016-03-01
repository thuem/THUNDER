/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "ImageFunctions.h"

#include "FFT.h"

#define N 128

using namespace std;

int main(int argc, const char* argv[])
{
    Image image(N, N, realSpace);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            if ((i > N / 4) &&
                (i < N / 4 * 3) &&
                (j > N / 4) &&
                (j < N / 4 * 3))
                image.setRL(1, i, j);
            else
                image.setRL(0, i, j);
        }

    image.saveRLToBMP("ori.bmp");

    cout << "****** NEG_RL ******" << endl;
    NEG_RL(image);
    image.saveRLToBMP("NEG_RL.bmp");

    /***
    cout << "****** normalise ******" << endl;

    normalise(image);
    ***/

    cout << "****** SCALE_RL ******" << endl;

    SCALE_RL(image, 10);
    image.saveRLToBMP("SCALE_RL.bmp");

    cout << "****** transalte ******" << endl;

    R2R_FT(image, image, translate(image, image, 0, 0));
    image.saveRLToBMP("trans0.bmp");

    R2R_FT(image, image, translate(image, image, N / 8, N / 8));
    image.saveRLToBMP("trans1.bmp");

    R2R_FT(image, image, translate(image, image, N / 4, N / 4));
    image.saveRLToBMP("trans2.bmp");
}

/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "ImageFunctions.h"

#include "FFT.h"

#define N 128

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

    R2R_FT(image, image, translate(image, image, 0, 0));
    image.saveRLToBMP("trans0.bmp");

    R2R_FT(image, image, translate(image, image, N / 8, N / 8));
    image.saveRLToBMP("trans1.bmp");

    R2R_FT(image, image, translate(image, image, N / 4, N / 4));
    image.saveRLToBMP("trans2.bmp");
}

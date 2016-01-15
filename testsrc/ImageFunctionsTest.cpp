/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "ImageFunctions.h"

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

    image.saveRLToBMP("image.bmp");
}

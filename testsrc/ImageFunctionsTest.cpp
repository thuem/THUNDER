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

#define N 3710

using namespace std;

int main(int argc, const char* argv[])
{
    Image image(N, N, RL_SPACE);

    try
    {
        IMAGE_FOR_EACH_PIXEL_RL(image)
        {
            image.setRL(1, i, j);
        }
    }
    catch (Error& err)
    {
        cout << err;
    }

    /***
    image.saveRLToBMP("ori.bmp");

    cout << "****** NEG_RL ******" << endl;
    NEG_RL(image);
    image.saveRLToBMP("NEG_RL.bmp");

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
    ***/

    cout << "****** extract ******" << endl;
    Image particle(100, 100, RL_SPACE);
    try
    {
        // extract(particle, image, 1251, 122);
        extract(particle, image, -554, -1683);

        ImageFile imf;
        imf.writeImage("image.mrc", particle);
    }
    catch (Error& err)
    {
        cout << err;
    }
}

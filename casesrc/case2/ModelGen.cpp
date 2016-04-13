/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "ImageFile.h"
#include "Volume.h"

#define N 380

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Read-in Volume" << endl;
    Volume vol;
    ImageFile imf("ref.mrc", "r");
    imf.readVolume(vol);

    return 0;
}

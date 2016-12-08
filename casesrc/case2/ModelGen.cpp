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

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    std::cout << "Read-in Volume" << std::endl;
    Volume vol;
    ImageFile imf("ref.mrc", "r");
    imf.readVolume(vol);

    return 0;
}

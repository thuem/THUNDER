//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <fstream>
#include <iostream>

#include <json/json.h>

#include "FFT.h"
#include "ImageFile.h"
#include "Volume.h"
#include "Euler.h"
#include "Transformation.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    ImageFile imfSrc(argv[2], "rb");

    Volume src;

    imfSrc.readMetaData();
    imfSrc.readVolume(src);

    dvec3 v;
    v << atof(argv[3]), atof(argv[4]), atof(argv[5]);

    dmat33 rot;
    alignZ(rot, v);

    // cout << rot << endl;

    Volume dst(src.nColRL(), src.nRowRL(), src.nSlcRL(), RL_SPACE);

    VOL_TRANSFORM_MAT_RL(dst, src, rot, src.nColRL() / 2 - 1, LINEAR_INTERP);

    ImageFile imfDst;

    imfDst.readMetaData(dst);
    imfDst.writeVolume(argv[1], dst, atof(argv[6]));

    return 0;
}

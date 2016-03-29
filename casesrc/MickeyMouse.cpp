/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Experiment.h"

#define PF 2

#define N 256
#define MAX_X 10
#define MAX_Y 10

#define PIXEL_SIZE 1.32

#define M 6000

using namespace std;

int main(int argc, char* argv[])
{
    cout << "Initialising Parameters" << endl;
    MLOptimiserPara para;
    para.iterMax = 30;
    para.pf = PF;
    para.a = 1.9;
    para.alpha = 10;
    para.pixelSize = PIXEL_SIZE;
    para.M = M;
    para.maxX = MAX_X;
    para.maxY = MAX_Y;
    sprintf(para.sym, "C2V");
    sprintf(para.initModel, "initMode.mrc");

    cout << "Setting Parameters" << endl;
    MLOptimiser opt;
    opt.setPara(para);
}

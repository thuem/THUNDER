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
#include "MLOptimiser.h"

#define PF 1

#define N 128
#define MAX_X 4
#define MAX_Y 4

#define PIXEL_SIZE 1.32

#define M 1000
#define MF 10

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    InitGoogleLogging(argv[0]);

    cout << "Initialising Parameters" << endl;
    MLOptimiserPara para;
    para.iterMax = atoi(argv[1]);
    para.k = 1;
    para.size = N;
    para.pf = PF;
    para.a = 1.9;
    para.alpha = 10;
    para.pixelSize = PIXEL_SIZE;
    para.m = M;
    para.mf = MF;
    para.maxX = MAX_X;
    para.maxY = MAX_Y;
    sprintf(para.sym, "C2");
    sprintf(para.initModel, "sphere.mrc");
    sprintf(para.db, "MickeyMouse.db");

    cout << "Setting Parameters" << endl;
    MLOptimiser opt;
    opt.setPara(para);

    cout << "MPISetting" << endl;
    opt.setMPIEnv();

    cout << "Run" << endl;
    opt.run();

    MPI_Finalize();
}

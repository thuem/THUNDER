/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include <gsl/gsl_randist.h>

#include "Projector.h"
#include "ImageFile.h"
#include "FFT.h"
#include "Random.h"
#include "CTF.h"

#define N 256
#define M 10
#define PF 2

#define PIXEL_SIZE 1.32
#define VOLTAGE 3e5
#define DEFOCUS 2e4
#define THETA 0
#define CS 0

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);
}

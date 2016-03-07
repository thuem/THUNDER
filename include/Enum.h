/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ENUM_H
#define ENUM_H 

enum ConjugateFlag
{
    conjugateUnknown = -1,
    conjugateYes,
    conjugateNo
};

/***
enum Space
{
    realSpace = 0,
    fourierSpace
};
***/

/***
enum Interpolation2DStyle
{
    nearest2D = 0,
    linear2D,
    sinc2D
};

enum Interpolation3DStyle
{
    nearest3D = 0,
    linear3D,
    sinc3D
};
***/

enum VerboseFlag
{
    verboseYes = 0,
    verboseNo
};

enum SamplingResolution
{
    coarse = 0,
    fine
};

#endif // ENUM_H 

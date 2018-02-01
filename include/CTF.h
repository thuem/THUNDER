//This header file is add by huabin
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef CTF_H
#define CTF_H

#include "huabin.h"
#include "Complex.h"
#include "Functions.h"
#include "Image.h"

//#define CTF_A 0.1

//#define CTF_TAU 0.01

//const RFLOAT w1 = sqrt(1 - CTF_A * CTF_A);
//const RFLOAT w2 = CTF_A;

RFLOAT CTF(const RFLOAT f,
           const RFLOAT voltage,
           const RFLOAT defocus,
           const RFLOAT CS,
           const RFLOAT amplitudeContrast,
           const RFLOAT phaseShift);

/**
 * This function generates a CTF using parameters given.
 *
 * @param dst the destination image
 * @param pixelSize pixel size (Angstrom)
 * @param voltage voltage (Volt)
 * @param defocusU the first defocus parameter (Angstrom)
 * @param defocusV the second defocus parameter (Angstrom)
 * @param theta the defocus angle (rad)
 * @param Cs Cs
 */
void CTF(Image& dst,
         const RFLOAT pixelSize,
         const RFLOAT voltage,
         const RFLOAT defocusU,
         const RFLOAT defocusV,
         const RFLOAT theta,
         const RFLOAT Cs,
         const RFLOAT amplitudeContrast,
         const RFLOAT phaseShift);

void CTF(RFLOAT* dst,
         const RFLOAT pixelSize,
         const RFLOAT voltage,
         const RFLOAT defocusU,
         const RFLOAT defocusV,
         const RFLOAT theta,
         const RFLOAT Cs,
         const RFLOAT amplitudeContrast,
         const RFLOAT phaseShift,
         const int nCol,
         const int nRow,
         const int* iCol,
         const int* iRow,
         const int _nPxl);

#endif // CTF_H

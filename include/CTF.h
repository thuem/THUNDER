//This header file is add by huabin
#include "huabin.h"
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

#include "Complex.h"
#include "Functions.h"
#include "Image.h"

#define CTF_A 0.1

#define CTF_TAU 0.01

const RFLOAT w1 = sqrt(1 - CTF_A * CTF_A);
const RFLOAT w2 = CTF_A;

RFLOAT CTF(const RFLOAT f,
           const RFLOAT voltage,
           const RFLOAT defocus,
           const RFLOAT CS);

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
         const RFLOAT Cs);

#endif // CTF_H

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

void CTF(Image& dst,
         const double pixelSize,
         const double voltage,
         const double defocusU,
         const double defocusV,
         const double theta,
         const double Cs);
/* pixelSize : Angstrom
 * voltage : V 
 * defocusU : Angstrom
 * defocusV : Angstrom
 * theta : rad
 * Cs */

void reduceCTF(Image& dst,
               const Image& src,
               const Image& ctf);

void reduceCTF(Image& dst,
               const Image& src,
               const Image& ctf,
               const double r);

#endif // CTF_H

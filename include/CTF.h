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

#define CTF_TAU 0.1

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
               const int r);

/**
 * This function reduce the CTF affect using Wiener filter. Sigma stands for the
 * power spectrum of the noise, meanwhile tau stands for the power spectrum of
 * the signal. It is worth noticed that tau and sigma should be in the same
 * scale.
 *
 * @param dst the destination image
 * @param src the source image
 * @param ctf CTF
 * @param sigma the power spectrum of the noise
 * @param tau the power spectrum of the signal
 * @param pf padding factor
 * @param r frequency threshold
 */
void reduceCTF(Image& dst,
               const Image& src,
               const Image& ctf,
               const vec& sigma,
               const vec& tau,
               const int pf,
               const int r);

#endif // CTF_H

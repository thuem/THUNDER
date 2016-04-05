/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <functional>

#include <armadillo>

#include "Error.h"
#include "Typedef.h"

#include "Image.h"
#include "Volume.h"

using namespace std;
using namespace arma;

double nyquist(const double pixelSize);
/* Nyquist resolution limit in Angstrom(-1) */

double resP2A(const double resP,
              const int imageSize,
              const double pixelSize);
/* convert the resolution from pixel to 1/Angstrom
 * imageSize -> the size of image in pixles
 * pixelSize -> the size of one pixel in Angstrom */

double resA2P(const double resA,
              const int imageSize,
              const double pixelSize);
/* convert the resolution from 1/Angstrom to pixel */

void resP2A(vec& res,
            const int imageSize,
            const double pixelSize);
/* convert the resolution from pixel to 1/Angstrom */

void resA2P(vec& res,
            const int imageSize,
            const double pixelSize);
/* convert the resolution from 1/Angstrom to pixel */

double ringAverage(const int resP,
                   const Image& img,
                   const function<double(const Complex)> func); 
/* calculate ring average at given resolution */

double shellAverage(const int resP,
                    const Volume& vol,
                    const function<double(const Complex)> func);
/* calculate shell average at given resolution */

void powerSpectrum(vec& dst,
                   const Image& src,
                   const int r);
/* calculate the power spectrum of the Fourier image */

void powerSpectrum(vec& dst,
                   const Volume& src,
                   const int r);
/* calculate the power spectrum of the Fourier volume */

void FRC(vec& dst,
         const Image& A,
         const Image& B);
/* calculate the Fourier ring coefficient */

void FSC(vec& dst,
         const Volume& A,
         const Volume& B);
/* calculate the Fourier shell coefficient */

/***
void wilsonPlot(std::map<double, double>& dst,
                const int imageSize,
                const double pixelSize,
                const double upperResA,
                const double lowerResA,
                const Vector<double>& ps,
                const int step = 1);

void wilsonPlot(std::map<double, double>& dst,
                const Volume& volume,
                const double pixelSize,
                const double upperResA,
                const double lowerResA);
                ***/

#endif // SPECTRUM_H

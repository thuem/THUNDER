/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Spectrum.h"

double nyquist(const double pixelSize)
{
    return 2 / pixelSize;
}

double resP2A(const double resP,
              const int imageSize,
              const double pixelSize)
{
    return resP / imageSize / pixelSize;
}

double resA2P(const double resA,
              const int imageSize,
              const double pixelSize)
{
    return resA * imageSize * pixelSize;
}

void resP2A(vec& res,
            const int imageSize,
            const double pixelSize)
{
    res.transform([&](double val){ return val / imageSize / pixelSize; });
}

void resA2P(vec& res,
            const int imageSize,
            const double pixelSize)
{
    res.transform([&](double val){ return val * imageSize * pixelSize; });
}

double ringAverage(const int resP,
                   const Image& img)
{
    double result = 0;
    int counter = 0;

    IMAGE_FOR_EACH_PIXEL_FT(img)
        if (AROUND(NORM(i, j)) == resP)
        {
            result += ABS(img.getFT(i, j));
            counter++;
        }

    return result / counter;
}

double ringAverage(const int resP,
                   const Image& img,
                   const function<double(const Complex)> func)
{
    double result = 0;
    int counter = 0;

    IMAGE_FOR_EACH_PIXEL_FT(img)
        if (AROUND(NORM(i, j)) == resP)
        {
            result += func(img.getFT(i, j));
            counter++;
        }

    return result / counter;
}

double shellAverage(const int resP,
                    const Volume& vol,
                    const function<double(const Complex)> func)
{
    double result = 0;
    int counter = 0;

    VOLUME_FOR_EACH_PIXEL_FT(vol)
        if (AROUND(NORM_3(i, j, k)) == resP)
        {
            result += func(vol.getFT(i, j, k));
            counter++;
        }

    return result / counter;
}

void powerSpectrum(vec& dst,
                   const Image& src,
                   const int r)
{
    for (int i = 0; i < r; i++)
        dst(i) = ringAverage(i, src, [](const Complex x){ return ABS2(x); });
}

void powerSpectrum(vec& dst,
                   const Volume& src,
                   const int r)
{
    for (int i = 0; i < r; i++)
        dst(i) = shellAverage(i, src, [](const Complex x){ return ABS2(x); });
}

void FRC(vec& dst,
         const Image& A,
         const Image& B,
         const int r)
{
    dst.resize(r);

    vec vecS = zeros<vec>(r);
    vec vecA = zeros<vec>(r);
    vec vecB = zeros<vec>(r);

    IMAGE_FOR_EACH_PIXEL_FT(A)
    {
        int u = AROUND(NORM(i, j));
        vecS[u] += REAL(A.getFT(i, j) * CONJUGATE(B.getFT(i, j)));
        vecA[u] += ABS2(A.getFT(i, j));
        vecB[u] += ABS2(B.getFT(i, j));
    }

    dst = vecS / sqrt(vecA % vecB);
}

void FSC(vec& dst,
         const Volume& A,
         const Volume& B,
         const int r)
{
    dst.resize(r);

    vec vecS = zeros<vec>(r);
    vec vecA = zeros<vec>(r);
    vec vecB = zeros<vec>(r);

    VOLUME_FOR_EACH_PIXEL_FT(A)
    {
        int u = AROUND(NORM_3(i, j, k));
        vecS[u] += REAL(A.getFT(i, j, k) * CONJUGATE(B.getFT(i, j, k)));
        vecA[u] += ABS2(A.getFT(i, j, k));
        vecB[u] += ABS2(B.getFT(i, j, k));
    }

    dst = vecS / sqrt(vecA % vecB);
}

/***
void wilsonPlot(std::map<double, double>& dst,
                const int imageSize,
                const double pixelSize,
                const double upperResA,
                const double lowerResA,
                const Vector<double>& ps,
                const int step)
{
    for (int i = 0; i < ps.length(); i++)
    {
        double resA;
        int resP = i * step;

        // calculate the resolution in Angstrom(-1)
        resP2A(resA, resP, imageSize, pixelSize);

        if ((resA <= upperResA) &&
            (resA >= lowerResA) &&
            (resA < nyquist(pixelSize)))
            dst.insert(std::pair<double, double>(resA * resA, ps.get(i)));
    }
}

void wilsonPlot(std::map<double, double>& dst,
                const Volume& volume,
                const double pixelSize,
                const double upperResA,
                const double lowerResA)
{
    // calculate power spectrum
    Vector<double> ps(volume.nColumn() / 2);
    powerSpectrum(ps, volume);

    // get Wilson Plot
    wilsonPlot(dst, volume.nColumn(), pixelSize, upperResA, lowerResA, ps);
}
***/

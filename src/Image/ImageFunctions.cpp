/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "ImageFunctions.h"

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        double phase = 2 * M_PI * (i * rCol + j * rRow);
        dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
    }
}

void translate(Image& dst,
               const Image& src,
               const double r,
               const double nTransCol,
               const double nTransRow)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD(i, j) < r * r)
        {
            double phase = 2 * M_PI * (i * rCol + j * rRow);
            dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
        }
    }
}

void crossCorrelation(Image& dst,
                      const Image& a,
                      const Image& b,
                      const double r)
{
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        if (QUAD(i, j) < r * r)
            dst.setFT(CONJUGATE(a.getFT(i, j)) * b.getFT(i, j), i, j);
}

void translate(int& nTransCol,
               int& nTransRow,
               const Image& a,
               const Image& b,
               const double r,
               const int maxX,
               const int maxY)
{
    Image cc(a.nColRL(),
             a.nRowRL(),
             FT_SPACE);

    SET_0_FT(cc);

    // calculate the cross correlation between A and B
    crossCorrelation(cc, a, b, r);

    FFT fft;

    //#pragma omp critical
    fft.bw(cc);

    double max = 0;

    nTransCol = 0;
    nTransRow = 0;

    for (int j = -maxY; j <= maxY; j++)
        for (int i = -maxX; i <= maxX; i++)
        {
            if (cc.getRL(i, j) > max)
            {            
                max = cc.getRL(i, j);
                nTransCol = i;
                nTransRow = j;
            }
        }
}

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Image& src,
                  const double r)
{
    vector<double> bg;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (NORM(i, j) > r)
            bg.push_back(src.getRL(i, j));

    mean = gsl_stats_mean(&bg[0], 1, bg.size());
    stddev = gsl_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void removeDust(Image& img,
                const double wDust,
                const double bDust,
                const double mean,
                const double stddev)
{
    auto engine = get_random_engine();

    IMAGE_FOR_EACH_PIXEL_RL(img)
        if ((img.getRL(i, j) > mean + wDust * stddev) ||
            (img.getRL(i, j) < mean - bDust * stddev))
            img.setRL(mean + gsl_ran_gaussian(engine, stddev), i, j);
}

void normalise(Image& img,
               const double wDust,
               const double bDust,
               const double r)
{
    double mean;
    double stddev;

    bgMeanStddev(mean, stddev, img, r);

    removeDust(img, wDust, bDust, mean, stddev);

    bgMeanStddev(mean, stddev, img, r);

    FOR_EACH_PIXEL_RL(img)
        img(i) -= mean;

    SCALE_RL(img, 1.0 / stddev);
}

void extract(Image& dst,
             const Image& src,
             const int xOff,
             const int yOff)
{
    IMAGE_FOR_EACH_PIXEL_RL(dst)
        dst.setRL(src.getRL(i + xOff, j + yOff), i, j);
}

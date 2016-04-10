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

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Image& src,
                  const double r)
{
    vector<double> bg;
    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (NORM(i, j) > r)
            bg.push_back(src.getRL(i, j));

    vec bv(bg);
    mean = arma::mean(bv);
    stddev = arma::stddev(bv);
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

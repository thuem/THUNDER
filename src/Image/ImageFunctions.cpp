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

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int r)
{
    IMAGE_FOR_PIXEL_R_FT(r)
    {
        if (QUAD(i, j) < gsl_pow_2(r))
        {
            int index = dst.iFTHalf(i, j);

            dst[index] = a.iGetFT(index) * b.iGetFT(index);
        }
    }
}

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int* iPxl,
         const int nPxl)
{
    for (int i = 0; i < nPxl; i++)
    {
        int index = iPxl[i];

        dst[index] = a.iGetFT(index) * b.iGetFT(index);
    }
}

void translate(Image& dst,
               const double nTransCol,
               const double nTransRow)
{
    double rCol = nTransCol / dst.nColRL();
    double rRow = nTransRow / dst.nRowRL();

    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        double phase = 2 * M_PI * (i * rCol + j * rRow);
        dst.setFT(COMPLEX_POLAR(-phase), i, j);
    }
}

void translateMT(Image& dst,
                 const double nTransCol,
                 const double nTransRow)
{
    double rCol = nTransCol / dst.nColRL();
    double rRow = nTransRow / dst.nRowRL();

    #pragma omp parallel for
    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        double phase = 2 * M_PI * (i * rCol + j * rRow);
        dst.setFT(COMPLEX_POLAR(-phase), i, j);
    }
}

void translate(Image& dst,
               const double r,
               const double nTransCol,
               const double nTransRow)
{
    double rCol = nTransCol / dst.nColRL();
    double rRow = nTransRow / dst.nRowRL();

    IMAGE_FOR_PIXEL_R_FT(r)
        if (QUAD(i, j) < gsl_pow_2(r))
        {
            double phase = 2 * M_PI * (i * rCol + j * rRow);
            dst.setFT(COMPLEX_POLAR(-phase), i, j);
        }
}

void translateMT(Image& dst,
                 const double r,
                 const double nTransCol,
                 const double nTransRow)
{
    double rCol = nTransCol / dst.nColRL();
    double rRow = nTransRow / dst.nRowRL();

    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        if (QUAD(i, j) < gsl_pow_2(r))
        {
            double phase = 2 * M_PI * (i * rCol + j * rRow);
            dst.setFT(COMPLEX_POLAR(-phase), i, j);
        }
}

void translate(Image& dst,
               const double nTransCol,
               const double nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl)
{
    double rCol = nTransCol / dst.nColRL();
    double rRow = nTransRow / dst.nRowRL();

    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[iPxl[i]] = COMPLEX_POLAR(-phase);
    }
}

void translateMT(Image& dst,
                 const double nTransCol,
                 const double nTransRow,
                 const int* iCol,
                 const int* iRow,
                 const int* iPxl,
                 const int nPxl)
{
    double rCol = nTransCol / dst.nColRL();
    double rRow = nTransRow / dst.nRowRL();

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[iPxl[i]] = COMPLEX_POLAR(-phase);
    }
}

void translate(Complex* dst,
               const double nTransCol,
               const double nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl)
{
    double rCol = nTransCol / nCol;
    double rRow = nTransRow / nRow;

    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[i] = COMPLEX_POLAR(-phase);
    }
}

void translateMT(Complex* dst,
                 const double nTransCol,
                 const double nTransRow,
                 const int nCol,
                 const int nRow,
                 const int* iCol,
                 const int* iRow,
                 const int nPxl)
{
    double rCol = nTransCol / nCol;
    double rRow = nTransRow / nRow;

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[i] = COMPLEX_POLAR(-phase);
    }
}

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

void translateMT(Image& dst,
                 const Image& src,
                 const double nTransCol,
                 const double nTransRow)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    #pragma omp parallel for
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

    IMAGE_FOR_PIXEL_R_FT(r)
        if (QUAD(i, j) < gsl_pow_2(r))
        {
            double phase = 2 * M_PI * (i * rCol + j * rRow);
            dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
        }
}

void translateMT(Image& dst,
                 const Image& src,
                 const double r,
                 const double nTransCol,
                 const double nTransRow)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_EACH_PIXEL_FT(src)
        if (QUAD(i, j) < gsl_pow_2(r))
        {
            double phase = 2 * M_PI * (i * rCol + j * rRow);
            dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
        }
}

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[iPxl[i]] = src.iGetFT(iPxl[i]) * COMPLEX_POLAR(-phase);
    }
}

void translateMT(Image& dst,
                 const Image& src,
                 const double nTransCol,
                 const double nTransRow,
                 const int* iCol,
                 const int* iRow,
                 const int* iPxl,
                 const int nPxl)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[iPxl[i]] = src.iGetFT(iPxl[i]) * COMPLEX_POLAR(-phase);
    }
}

void translate(Complex* dst,
               const Complex* src,
               const double nTransCol,
               const double nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl)
{
    double rCol = nTransCol / nCol;
    double rRow = nTransRow / nRow;

    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[i] = src[i] * COMPLEX_POLAR(-phase);
    }
}

void translateMT(Complex* dst,
                 const Complex* src,
                 const double nTransCol,
                 const double nTransRow,
                 const int nCol,
                 const int nRow,
                 const int* iCol,
                 const int* iRow,
                 const int nPxl)
{
    double rCol = nTransCol / nCol;
    double rRow = nTransRow / nRow;

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        double phase = 2 * M_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[i] = src[i] * COMPLEX_POLAR(-phase);
    }
}

void crossCorrelation(Image& dst,
                      const Image& a,
                      const Image& b,
                      const double r)
{
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        if (QUAD(i, j) < gsl_pow_2(r))
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

double stddev(const double mean,
              const Image& src)
{
    return gsl_stats_sd_m(&src.iGetRL(0), 1, src.sizeRL(), mean);
}

void meanStddev(double& mean,
                double& stddev,
                const Image& src)
{
    mean = gsl_stats_mean(&src.iGetRL(0), 1, src.sizeRL());
    stddev = gsl_stats_sd_m(&src.iGetRL(0), 1, src.sizeRL(), mean);
}

double centreStddev(const double mean,
                    const Image& src,
                    const double r)
{
    vector<double> centre;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) < gsl_pow_2(r))
            centre.push_back(src.getRL(i, j));

    return gsl_stats_sd_m(&centre[0], 1, centre.size(), mean);
}

void centreMeanStddev(double& mean,
                      double& stddev,
                      const Image& src,
                      const double r)
{
    vector<double> centre;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) < gsl_pow_2(r))
            centre.push_back(src.getRL(i, j));

    mean = gsl_stats_mean(&centre[0], 1, centre.size());
    stddev = gsl_stats_sd_m(&centre[0], 1, centre.size(), mean);
}

double bgStddev(const double mean,
                const Image& src,
                const double r)
{
    vector<double> bg;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) > gsl_pow_2(r))
            bg.push_back(src.getRL(i, j));

    return gsl_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

double bgStddev(const double mean,
                const Volume& src,
                const double r)
{
    // TODO
    //
    return 0;
}

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Image& src,
                  const double r)
{
    vector<double> bg;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) > gsl_pow_2(r))
            bg.push_back(src.getRL(i, j));

    mean = gsl_stats_mean(&bg[0], 1, bg.size());
    stddev = gsl_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Volume& src,
                  const double r)
{
    vector<double> bg;

    VOLUME_FOR_EACH_PIXEL_RL(src)
        if (QUAD_3(i, j, k) > gsl_pow_2(r))
            bg.push_back(src.getRL(i, j, k));

    mean = gsl_stats_mean(&bg[0], 1, bg.size());
    stddev = gsl_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Volume& src,
                  const double rU,
                  const double rL)
{
    vector<double> bg;

    VOLUME_FOR_EACH_PIXEL_RL(src)
        if ((QUAD_3(i, j, k) >= gsl_pow_2(rL)) &&
            (QUAD_3(i, j, k) < gsl_pow_2(rU)))
            bg.push_back(src.getRL(i, j, k));

    mean = gsl_stats_mean(&bg[0], 1, bg.size());
    stddev = gsl_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void removeDust(Image& img,
                const double wDust,
                const double bDust,
                const double mean,
                const double stddev)
{
    gsl_rng* engine = get_random_engine();

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

void binning(Image& dst,
             const Image& src,
             const int bf)
{
    int nCol = src.nColRL() / bf;
    int nRow = src.nRowRL() / bf;

    dst.alloc(nCol, nRow, RL_SPACE);

    IMAGE_FOR_EACH_PIXEL_RL(dst)
    {
        double sum = 0;

        for (int y = 0; y < bf; y++)
            for (int x = 0; x < bf; x++)
                sum += src.getRL(bf * i + x,
                                 bf * j + y);

        dst.setRL(sum / gsl_pow_2(bf),
                  i,
                  j);
    }
}

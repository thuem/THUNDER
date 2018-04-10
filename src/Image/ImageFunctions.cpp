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

#include "ImageFunctions.h"

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int r)
{
    IMAGE_FOR_PIXEL_R_FT(r)
    {
        if (QUAD(i, j) < TSGSL_pow_2(r))
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
               const RFLOAT nTransCol,
               const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();

    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
        dst.setFT(COMPLEX_POLAR(-phase), i, j);
    }
}

void translateMT(Image& dst,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();

    #pragma omp parallel for
    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
        dst.setFT(COMPLEX_POLAR(-phase), i, j);
    }
}

void translateMT(Volume& dst,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow,
                 const RFLOAT nTransSlc)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();
    RFLOAT rSlc = nTransSlc / dst.nSlcRL();

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(dst)
    {
        RFLOAT phase = M_2X_PI * (i * rCol + j * rRow + k * rSlc);
        dst.setFT(COMPLEX_POLAR(-phase), i, j, k);
    }
}

void translate(Image& dst,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();

    IMAGE_FOR_PIXEL_R_FT(r)
        if (QUAD(i, j) < TSGSL_pow_2(r))
        {
            RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
            dst.setFT(COMPLEX_POLAR(-phase), i, j);
        }
}

void translateMT(Image& dst,
                 const RFLOAT r,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();

    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        if (QUAD(i, j) < TSGSL_pow_2(r))
        {
            RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
            dst.setFT(COMPLEX_POLAR(-phase), i, j);
        }
}

void translate(Image& dst,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();

    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[iPxl[i]] = COMPLEX_POLAR(-phase);
    }
}

void translateMT(Image& dst,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow,
                 const int* iCol,
                 const int* iRow,
                 const int* iPxl,
                 const int nPxl)
{
    RFLOAT rCol = nTransCol / dst.nColRL();
    RFLOAT rRow = nTransRow / dst.nRowRL();

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[iPxl[i]] = COMPLEX_POLAR(-phase);
    }
}

void translate(Complex* dst,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl)
{
    RFLOAT rCol = nTransCol / nCol;
    RFLOAT rRow = nTransRow / nRow;

    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[i] = COMPLEX_POLAR(-phase);
    }
}

void translateMT(Complex* dst,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow,
                 const int nCol,
                 const int nRow,
                 const int* iCol,
                 const int* iRow,
                 const int nPxl)
{
    RFLOAT rCol = nTransCol / nCol;
    RFLOAT rRow = nTransRow / nRow;

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);
        dst[i] = COMPLEX_POLAR(-phase);
    }
}

void translate(Image& dst,
               const Image& src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
        dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
    }
}

void translateMT(Image& dst,
                 const Image& src,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();

    #pragma omp parallel for
    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
        dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
    }
}

void translateMT(Volume& dst,
                 const Volume& src,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow,
                 const RFLOAT nTransSlc)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();
    RFLOAT rSlc = nTransSlc / src.nSlcRL();

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT phase = M_2X_PI * (i * rCol + j * rRow + k * rSlc);
        dst.setFTHalf(src.getFTHalf(i, j, k) * COMPLEX_POLAR(-phase), i, j, k);
    }
}

void translate(Image& dst,
               const Image& src,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();

    IMAGE_FOR_PIXEL_R_FT(r)
        if (QUAD(i, j) < TSGSL_pow_2(r))
        {
            RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
            dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
        }
}

void translateMT(Image& dst,
                 const Image& src,
                 const RFLOAT r,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();

    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_EACH_PIXEL_FT(src)
        if (QUAD(i, j) < TSGSL_pow_2(r))
        {
            RFLOAT phase = M_2X_PI * (i * rCol + j * rRow);
            dst.setFT(src.getFT(i, j) * COMPLEX_POLAR(-phase), i, j);
        }
}

void translate(Image& dst,
               const Image& src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();

    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[iPxl[i]] = src.iGetFT(iPxl[i]) * COMPLEX_POLAR(-phase);
    }
}

void translateMT(Image& dst,
                 const Image& src,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow,
                 const int* iCol,
                 const int* iRow,
                 const int* iPxl,
                 const int nPxl)
{
    RFLOAT rCol = nTransCol / src.nColRL();
    RFLOAT rRow = nTransRow / src.nRowRL();

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[iPxl[i]] = src.iGetFT(iPxl[i]) * COMPLEX_POLAR(-phase);
    }
}

void translate(Complex* dst,
               const Complex* src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl)
{
    RFLOAT rCol = nTransCol / nCol;
    RFLOAT rRow = nTransRow / nRow;

    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[i] = src[i] * COMPLEX_POLAR(-phase);
    }
}

void translateMT(Complex* dst,
                 const Complex* src,
                 const RFLOAT nTransCol,
                 const RFLOAT nTransRow,
                 const int nCol,
                 const int nRow,
                 const int* iCol,
                 const int* iRow,
                 const int nPxl)
{
    RFLOAT rCol = nTransCol / nCol;
    RFLOAT rRow = nTransRow / nRow;

    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        RFLOAT phase = M_2X_PI * (iCol[i] * rCol + iRow[i] * rRow);

        dst[i] = src[i] * COMPLEX_POLAR(-phase);
    }
}

void crossCorrelation(Image& dst,
                      const Image& a,
                      const Image& b,
                      const RFLOAT r)
{
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        if (QUAD(i, j) < TSGSL_pow_2(r))
            dst.setFT(CONJUGATE(a.getFT(i, j)) * b.getFT(i, j), i, j);
}

void translate(int& nTransCol,
               int& nTransRow,
               const Image& a,
               const Image& b,
               const RFLOAT r,
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

    RFLOAT max = 0;

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

RFLOAT stddev(const RFLOAT mean,
              const Image& src)
{
    return TSGSL_stats_sd_m(&src.iGetRL(0), 1, src.sizeRL(), mean);
}

void meanStddev(RFLOAT& mean,
                RFLOAT& stddev,
                const Image& src)
{
    mean = TSGSL_stats_mean(&src.iGetRL(0), 1, src.sizeRL());
    stddev = TSGSL_stats_sd_m(&src.iGetRL(0), 1, src.sizeRL(), mean);
}

RFLOAT centreStddev(const RFLOAT mean,
                    const Image& src,
                    const RFLOAT r)
{
    vector<RFLOAT> centre;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) < TSGSL_pow_2(r))
            centre.push_back(src.getRL(i, j));

    return TSGSL_stats_sd_m(&centre[0], 1, centre.size(), mean);
}

void centreMeanStddev(RFLOAT& mean,
                      RFLOAT& stddev,
                      const Image& src,
                      const RFLOAT r)
{
    vector<RFLOAT> centre;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) < TSGSL_pow_2(r))
            centre.push_back(src.getRL(i, j));

    mean = TSGSL_stats_mean(&centre[0], 1, centre.size());
    stddev = TSGSL_stats_sd_m(&centre[0], 1, centre.size(), mean);
}

RFLOAT bgStddev(const RFLOAT mean,
                const Image& src,
                const RFLOAT r)
{
    vector<RFLOAT> bg;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) > TSGSL_pow_2(r))
            bg.push_back(src.getRL(i, j));

    return TSGSL_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

RFLOAT bgStddev(const RFLOAT mean,
                const Volume& src,
                const RFLOAT r)
{
    // TODO
    //
    return 0;
}

void bgMeanStddev(RFLOAT& mean,
                  RFLOAT& stddev,
                  const Image& src,
                  const RFLOAT r)
{
    vector<RFLOAT> bg;

    IMAGE_FOR_EACH_PIXEL_RL(src)
        if (QUAD(i, j) > TSGSL_pow_2(r))
            bg.push_back(src.getRL(i, j));

    mean = TSGSL_stats_mean(&bg[0], 1, bg.size());
    stddev = TSGSL_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void bgMeanStddev(RFLOAT& mean,
                  RFLOAT& stddev,
                  const Volume& src,
                  const RFLOAT r)
{
    vector<RFLOAT> bg;

    VOLUME_FOR_EACH_PIXEL_RL(src)
        if (QUAD_3(i, j, k) > TSGSL_pow_2(r))
            bg.push_back(src.getRL(i, j, k));

    mean = TSGSL_stats_mean(&bg[0], 1, bg.size());
    stddev = TSGSL_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void bgMeanStddev(RFLOAT& mean,
                  RFLOAT& stddev,
                  const Volume& src,
                  const RFLOAT rU,
                  const RFLOAT rL)
{
    vector<RFLOAT> bg;

    VOLUME_FOR_EACH_PIXEL_RL(src)
        if ((QUAD_3(i, j, k) >= TSGSL_pow_2(rL)) &&
            (QUAD_3(i, j, k) < TSGSL_pow_2(rU)))
            bg.push_back(src.getRL(i, j, k));

    mean = TSGSL_stats_mean(&bg[0], 1, bg.size());
    stddev = TSGSL_stats_sd_m(&bg[0], 1, bg.size(), mean);
}

void removeDust(Image& img,
                const RFLOAT wDust,
                const RFLOAT bDust,
                const RFLOAT mean,
                const RFLOAT stddev)
{
    gsl_rng* engine = get_random_engine();

    IMAGE_FOR_EACH_PIXEL_RL(img)
        if ((img.getRL(i, j) > mean + wDust * stddev) ||
            (img.getRL(i, j) < mean - bDust * stddev))
            img.setRL(mean + TSGSL_ran_gaussian(engine, stddev), i, j);
}

void normalise(Image& img,
               const RFLOAT wDust,
               const RFLOAT bDust,
               const RFLOAT r)
{
    RFLOAT mean;
    RFLOAT stddev;

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
        RFLOAT sum = 0;

        for (int y = 0; y < bf; y++)
            for (int x = 0; x < bf; x++)
                sum += src.getRL(bf * i + x,
                                 bf * j + y);

        dst.setRL(sum / TSGSL_pow_2(bf),
                  i,
                  j);
    }
}

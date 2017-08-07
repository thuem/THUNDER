/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Mask.h"

double nPixel(const double r,
              const double ew)
{
    double sum = 0;

    for (int j = FLOOR(-(r + ew)); j <= CEIL(r + ew); j++)
        for (int i = FLOOR(-(r + ew)); i <= CEIL(r + ew); i++)
        {
            double u = NORM(i, j);

            if (u <= r)
                sum += 1;
            else if (u <= r + ew)
                sum += 0.5 + 0.5 * cos((u - r) / ew * M_PI);
        }

    return sum;
}

double nVoxel(const double r,
              const double ew)
{
    double sum = 0;

    for (int k = FLOOR(-(r + ew)); k <= CEIL(r + ew); k++)
        for (int j = FLOOR(-(r + ew)); j <= CEIL(r + ew); j++)
            for (int i = FLOOR(-(r + ew)); i <= CEIL(r + ew); i++)
            {
                double u = NORM_3(i, j, k);

                if (u <= r)
                    sum += 1;
                else if (u <= r + ew)
                    sum += 0.5 + 0.5 * cos((u - r) / ew * M_PI);
            }

    return sum;
}

double regionMean(const Image& img,
                  const int r)
{
    double weightSum = 0;
    double sum = 0;

    IMAGE_FOR_EACH_PIXEL_RL(img)
    {
        int u = AROUND(NORM(i, j));

        if (u == r)
        {
            weightSum += 1;
            sum += img.getRL(i, j);
        }
    }

    return sum / weightSum;
}

double regionMean(const Volume& vol,
                  const int r)
{
    double weightSum = 0;
    double sum = 0;

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        int u = AROUND(NORM_3(i, j, k));

        if (u == r)
        {
            #pragma omp atomic
            weightSum += 1;

            #pragma omp atomic
            sum += vol.getRL(i, j, k);
        }
    }

    return sum / weightSum;
}

double regionMean(const Image& img,
                  const double rU,
                  const double rL)
{
    double weightSum = 0;
    double sum = 0;

    IMAGE_FOR_EACH_PIXEL_RL(img)
    {
        double u = NORM(i, j);

        if ((u < rU) &&
            (u >= rL))
        {
            weightSum += 1;
            sum += img.getRL(i, j);
        }
    }

    return sum / weightSum;
}

double regionMean(const Volume& vol,
                  const double rU,
                  const double rL)
{
    double weightSum = 0;
    double sum = 0;

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        double u = NORM_3(i, j, k);

        if ((u < rU) &&
            (u >= rL))
        {
            #pragma omp atomic
            weightSum += 1;

            #pragma omp atomic
            sum += vol.getRL(i, j, k);
        }
    }

    return sum / weightSum;
}

double background(const Image& img,
                  const double r,
                  const double ew)
{
    double weightSum = 0;
    double sum = 0;

    IMAGE_FOR_EACH_PIXEL_RL(img)
    {
        double u = NORM(i, j);

        if (u > r + ew)
        {
            weightSum += 1;
            sum += img.getRL(i, j);
        }
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI); // portion of background
            weightSum += w;
            sum += img.getRL(i, j) * w;
        }
    }

    return sum / weightSum;
}

double background(const Image& img,
                  const Image& alpha)
{
    double weightSum = 0;
    double sum = 0;

    IMAGE_FOR_EACH_PIXEL_RL(img)
    {
        double w = 1 - alpha.getRL(i, j); // portion of background
        weightSum += w;
        sum += img.getRL(i, j) * w;
    }

    return sum / weightSum;
}

double background(const Volume& vol,
                  const double r,
                  const double ew)
{
    double weightSum = 0;
    double sum = 0;
    
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        double u = NORM_3(i, j, k);

        if (u > r + ew)
        {
            #pragma omp atomic
            weightSum += 1;

            #pragma omp atomic
            sum += vol.getRL(i, j, k);
        }
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI); // portion of background

            #pragma omp atomic
            weightSum += w;

            #pragma omp atomic
            sum += vol.getRL(i, j, k) * w;
        }
    }

    return sum / weightSum;
}

double background(const Volume& vol,
                  const Volume& alpha)
{
    double weightSum = 0;
    double sum = 0;

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        double w = 1 - alpha.getRL(i, j, k); // portion of background

        #pragma omp atomic
        weightSum += w;

        #pragma omp atomic
        sum += vol.getRL(i, j, k) * w;
    }

    return sum / weightSum;
}

double background(const Volume& vol,
                  const double rU,
                  const double rL,
                  const double ew)
{
    double weightSum = 0;
    double sum = 0;
    
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        double u = NORM_3(i, j, k);

        if ((u > rL + ew) &&
            (u < rU))
        {
            #pragma omp atomic
            weightSum += 1;

            #pragma omp atomic
            sum += vol.getRL(i, j, k);
        }
        else if ((u >= rL) &&
                 (u < rU))
        {
            double w = 0.5 - 0.5 * cos((u - rL) / ew * M_PI); // portion of background

            #pragma omp atomic
            weightSum += w;

            #pragma omp atomic
            sum += vol.getRL(i, j, k) * w;
        }
    }

    return sum / weightSum;
}

/***
void directSoftMask(Image& dst,
                    const Image& src,
                    const double r,
                    const double ew)
{
    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double u = NORM(i, j);

        if (u > r + ew)
            dst.setRL(bg, i, j);
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI); // portion of background
            dst.setRL(bg * w + src.getRL(i, j) * (1 - w), i, j);
        }
        else
            dst.setRL(src.getRL(i, j), i, j);
    }
}
***/

void softMask(Image& mask,
              const double r,
              const double ew)
{
    IMAGE_FOR_EACH_PIXEL_RL(mask)
    {
        double u = NORM(i, j);

        if (u > r + ew)
            mask.setRL(0, i, j);
        else if (u >= r)
            mask.setRL(0.5 + 0.5 * cos((u - r) / ew * M_PI), i, j);
        else
            mask.setRL(1, i, j);
    }
}

void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew)
{
    double bg = background(src, r, ew);

    softMask(dst, src, r, ew, bg);
}

void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew,
              const double bg)
{
    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double u = NORM(i, j);

        if (u > r + ew)
            dst.setRL(bg, i, j);
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI); // portion of background
            dst.setRL(bg * w + src.getRL(i, j) * (1 - w), i, j);
        }
        else
            dst.setRL(src.getRL(i, j), i, j);
    }
}

void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew,
              const double bgMean,
              const double bgStd)
{
    gsl_rng* engine = get_random_engine();

    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double u = NORM(i, j);

        if (u < r)
            dst.setRL(src.getRL(i, j), i, j);
        else
        {
            double bg = bgMean + gsl_ran_gaussian(engine, bgStd);

            if (u > r + ew)
                dst.setRL(bg, i, j);
            else
            {
                double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI); // portion of background
                dst.setRL(bg * w + src.getRL(i, j) * (1 - w), i, j);
            }
        }
    }
}

void softMask(Image& dst,
              const Image& src,
              const Image& alpha)
{
    double bg = background(src, alpha);

    softMask(dst, src, alpha, bg);
}

void softMask(Image& dst,
              const Image& src,
              const Image& alpha,
              const double bg)
{
    FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.iGetRL(i);
        dst(i) = bg * w + src.iGetRL(i) * (1 - w);
    }
    /***
    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.getRL(i, j); // portion of background
        dst.setRL(bg * w + src.getRL(i, j) * (1 - w), i, j);
    }
    ***/
}

void softMask(Image& dst,
              const Image& src,
              const Image& alpha,
              const double bgMean,
              const double bgStd)
{
    gsl_rng* engine = get_random_engine();

    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.getRL(i, j); // portion of background

        double bg = bgMean + gsl_ran_gaussian(engine, bgStd);

        dst.setRL(bg * w + src.getRL(i, j) * (1 - w), i, j);
    }
}

void softMask(Volume& mask,
              const double r,
              const double ew)
{
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(mask)
    {
        double u = NORM_3(i, j, k);

        if (u > r + ew)
            mask.setRL(0, i, j, k);
        else if (u >= r)
            mask.setRL(0.5 + 0.5 * cos((u - r) / ew * M_PI), i, j, k);
        else
            mask.setRL(1, i, j, k);
    }
}
void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew)
{
    double bg = background(src, r, ew);

    softMask(dst, src, r, ew, bg);
}

void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew,
              const double bg)
{
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src)
    {
        double u = NORM_3(i, j, k);

        if (u > r + ew)
            dst.setRL(bg, i, j, k);
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI); // portion of background
            dst.setRL(bg * w + src.getRL(i, j, k) * (1 - w), i, j, k);
        }
        else
            dst.setRL(src.getRL(i, j, k), i, j, k);
    }
}

void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha)
{
    double bg = background(src, alpha);

    softMask(dst, src, alpha, bg);
}

void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha,
              const double bg)
{
    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.iGetRL(i); // portion of background
        dst(i) = bg * w + src.iGetRL(i) * (1 - w);
    }
    /***
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.getRL(i, j, k); // portion of background
        dst.setRL(bg * w + src.getRL(i, j, k) * (1 - w), i, j, k);
    }
    ***/
}

void regionBgSoftMask(Image& dst,
                      const Image& src,
                      const double r,
                      const double ew,
                      const double rU,
                      const double rL)
{
    double bg = regionMean(src, rU, rL);

    softMask(dst, src, r, ew, bg);
}

void regionBgSoftMask(Volume& dst,
                      const Volume& src,
                      const double r,
                      const double ew,
                      const double rU,
                      const double rL)
{
    double bg = regionMean(src, rU, rL);

    softMask(dst, src, r, ew, bg);
}

void removeIsolatedPoint(Volume& vol)
{
    Volume volTmp = vol.copyVolume();

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(vol)
        if (vol.getRL(i, j, k) == 1)
        {
            bool isolated = true;

            if (((i - 1 >= -vol.nColRL() / 2) &&
                 (vol.getRL(i - 1, j, k) == 1)) ||
                ((j - 1 >= -vol.nRowRL() / 2) &&
                 (vol.getRL(i, j - 1, k) == 1)) ||
                ((k - 1 >= -vol.nSlcRL() / 2) &&
                 (vol.getRL(i, j, k - 1) == 1)) ||
                ((i + 1 < vol.nColRL() / 2) &&
                 (vol.getRL(i + 1, j, k) == 1)) ||
                ((j + 1 < vol.nRowRL() / 2) &&
                 (vol.getRL(i, j + 1, k) == 1)) ||
                ((k + 1 < vol.nSlcRL() / 2) &&
                 (vol.getRL(i, j, k + 1) == 1)))
                isolated = false;

            if (isolated) volTmp.setRL(0, i, j, k);
        }

    vol.swap(volTmp);
}

void extMask(Volume& vol,
             const double ext)
{
    Volume volTmp = vol.copyVolume();

    int a = CEIL(std::abs(ext));

    if (ext > 0)
    {
        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(vol)
            if (vol.getRL(i, j, k) == 1)
                VOLUME_FOR_EACH_PIXEL_IN_GRID(a)
                    if (QUAD_3(x, y, z) < gsl_pow_2(ext))
                        volTmp.setRL(1, i + x, j + y, k + z);
    }
    else if (ext < 0)
    {
        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(vol)
            if (vol.getRL(i, j, k) == 0)
                VOLUME_FOR_EACH_PIXEL_IN_GRID(a)
                    if (QUAD_3(x, y, z) < gsl_pow_2(ext))
                        volTmp.setRL(0, i + x, j + y, k + z);
    }

    vol.swap(volTmp);
}

void softEdge(Volume& vol,
              const double ew)
{
    int a = CEIL(ew);

    Volume distance(vol.nColRL(),
                    vol.nRowRL(),
                    vol.nSlcRL(),
                    RL_SPACE);

    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(distance)
        distance(i) = FLT_MAX;

    omp_lock_t* mtx = new omp_lock_t[vol.sizeRL()];

    #pragma omp parallel for
    for (int i = 0; i < (int)vol.sizeRL(); i++)
        omp_init_lock(&mtx[i]);

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(vol)
        if (vol.getRL(i, j, k) == 1)
            VOLUME_FOR_EACH_PIXEL_IN_GRID(a)
            {
                double d = NORM_3(x, y, z);

                if (d < ew)
                {
                    int index = distance.iRL(i + x, j + y, k + z);
                    omp_set_lock(&mtx[index]);
                    if (distance(index) > d) distance(index) = d;
                    omp_unset_lock(&mtx[index]);
                }
            }

    #pragma omp parallel for
    for (int i = 0; i < (int)vol.sizeRL(); i++)
        omp_destroy_lock(&mtx[i]);

    delete[] mtx;

    #pragma omp parallel for schedule(dynamic)
    FOR_EACH_PIXEL_RL(vol)
    {
        double d = distance(i);
        if ((d != 0) && (d < ew))
            vol(i) = 0.5 + 0.5 * cos(d / ew * M_PI);
    }
}

void genMask(Volume& dst,
             const Volume& src,
             const double thres)
{
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src)
        if (src.getRL(i, j, k) > thres)
            dst.setRL(1, i, j, k);
        else
            dst.setRL(0, i, j, k);

    // remove isolated point
    removeIsolatedPoint(dst);
}

void genMask(Volume& dst,
             const Volume& src,
             const double thres,
             const double ext)
{
    genMask(dst, src, thres);

    extMask(dst, ext);
}

void genMask(Volume& dst,
             const Volume& src,
             const double thres,
             const double ext,
             const double ew)
{
    genMask(dst, src, thres, ext);

    softEdge(dst, ew);
}

void autoMask(Volume& dst,
              const Volume& src,
              const double r)
{
    vector<double> data;
    VOLUME_FOR_EACH_PIXEL_RL(src)
        if (QUAD_3(i, j, k) < gsl_pow_2(r))
            data.push_back(GSL_MAX_DBL(0, src.getRL(i, j, k)));

    size_t n = data.size();

    sort(&data[0],
         &data[0] + n,
         std::greater<double>());

    vector<double> partialSum(n);
    std::partial_sum(&data[0], &data[0] + n, &partialSum[0]);

    double totalSum = partialSum[n - 1];

    size_t start;
    for (start = 0; start < n; start++)
        if (partialSum[start + 1] > totalSum * GEN_MASK_INIT_STEP)
            break;

    double thres = 0;

    double step = GEN_MASK_INIT_STEP + GEN_MASK_GAP;
    double gap = GEN_MASK_GAP;
    int nPrevBin = 0;
    int prev = 0;
    int bin = 0;

    for (size_t i = start; i < n; i++)
    {
        if (partialSum[i] < totalSum * step)
            bin += 1;
        else
        {
            if ((nPrevBin != 0) &&
                (prev * 2 < bin * nPrevBin))
                break;
            else
            {
                step += gap;

                nPrevBin += 1;
                prev += bin;

                bin = 0;

                thres = data[i];
            }
        }
    }

    genMask(dst, src, thres);
}

void autoMask(Volume& dst,
              const Volume& src,
              const double ext,
              const double r)
{
    autoMask(dst, src, r);

    extMask(dst, ext);
}

void autoMask(Volume& dst,
              const Volume& src,
              const double ext,
              const double ew,
              const double r)
{
    autoMask(dst, src, ext, r);

    softEdge(dst, ew);
}

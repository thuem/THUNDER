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
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);
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
        double w = 1 - alpha.getRL(i, j);
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
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);

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
        double w = 1 - alpha.getRL(i, j, k);

        #pragma omp atomic
        weightSum += w;

        #pragma omp atomic
        sum += vol.getRL(i, j, k) * w;
    }

    return sum / weightSum;
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
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);
            dst.setRL(bg * (1 - w) + src.getRL(i, j) * w, i, j);
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
    auto engine = get_random_engine();

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
                double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);
                dst.setRL(bg * (1 - w) + src.getRL(i, j) * w, i, j);
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
    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double w = alpha.getRL(i, j);
        dst.setRL(bg * (1 - w) + w * src.getRL(i, j), i, j);
    }
}

void softMask(Image& dst,
              const Image& src,
              const Image& alpha,
              const double bgMean,
              const double bgStd)
{
    auto engine = get_random_engine();

    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double w = alpha.getRL(i, j);

        double bg = bgMean + gsl_ran_gaussian(engine, bgStd);

        dst.setRL(bg * (1 - w) + w * src.getRL(i, j), i, j);
    }
}

void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew)
{
    double bg = background(src, r, ew);

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src)
    {
        double u = NORM_3(i, j, k);

        if (u > r + ew)
            dst.setRL(bg, i, j, k);
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);
            dst.setRL(bg * (1 - w) + w * src.getRL(i, j, k), i, j, k);
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
    VOLUME_FOR_EACH_PIXEL_RL(src)
    {
        double w = alpha.getRL(i, j, k);
        dst.setRL(bg * (1 - w) + w * src.getRL(i, j, k), i, j, k);
    }
}

void genMask(Volume& dst,
             const Volume& src,
             const double dt,
             const double r)
{
    //double thres = gsl_stats_max(&src.iGetRL(0), 1, src.sizeRL()) * dt;
    vector<double> bg;
    VOLUME_FOR_EACH_PIXEL_RL(src)
        if ((QUAD_3(i, j, k) < gsl_pow_2(r)) &&
            (QUAD_3(i, j, k) > gsl_pow_2(r * 0.85)))
            bg.push_back(src.getRL(i, j, k));

    //double mean = gsl_stats_mean(&src.iGetRL(0), 1, src.sizeRL());
    //double std = gsl_stats_sd_m(&src.iGetRL(0), 1, src.sizeRL(), mean);

    double mean = gsl_stats_mean(&bg[0], 1, bg.size());
    double std = gsl_stats_sd_m(&bg[0], 1, bg.size(), mean);

    cout << "mean = " << mean << endl;
    cout << "std = " << std << endl;

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src)
        if (src.getRL(i, j, k) > mean + dt * std)
            dst.setRL(1, i, j, k);
        else
            dst.setRL(0, i, j, k);

    // remove isolated point

    Volume dstTmp = dst.copyVolume();

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
        if (dst.getRL(i, j, k) == 1)
        {
            bool isolated = true;

            if (((i - 1 >= -dst.nColRL() / 2) &&
                 (dst.getRL(i - 1, j, k) == 1)) ||
                ((j - 1 >= -dst.nRowRL() / 2) &&
                 (dst.getRL(i, j - 1, k) == 1)) ||
                ((k - 1 >= -dst.nSlcRL() / 2) &&
                 (dst.getRL(i, j, k - 1) == 1)) ||
                ((i + 1 < dst.nColRL() / 2) &&
                 (dst.getRL(i + 1, j, k) == 1)) ||
                ((j + 1 < dst.nRowRL() / 2) &&
                 (dst.getRL(i, j + 1, k) == 1)) ||
                ((k + 1 < dst.nSlcRL() / 2) &&
                 (dst.getRL(i, j, k + 1) == 1)))
                isolated = false;

            if (isolated) dstTmp.setRL(0, i, j, k);
        }

    dst = move(dstTmp);
}

void genMask(Volume& dst,
             const Volume& src,
             const double dt,
             const double ext,
             const double r)
{
    genMask(dst, src, dt, r);

    Volume dstTmp = dst.copyVolume();

    int a = CEIL(abs(ext));

    if (ext > 0)
    {
        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(dst)
            if (dst.getRL(i, j, k) == 1)
                VOLUME_FOR_EACH_PIXEL_IN_GRID(a)
                    if (QUAD_3(x, y, z) < gsl_pow_2(ext))
                        dstTmp.setRL(1, i + x, j + y, k + z);
    }
    else if (ext < 0)
    {
        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(dst)
            if (dst.getRL(i, j, k) == 0)
                VOLUME_FOR_EACH_PIXEL_IN_GRID(a)
                    if (QUAD_3(x, y, z) < gsl_pow_2(ext))
                        dstTmp.setRL(0, i + x, j + y, k + z);
    }

    dst = move(dstTmp);
}

void genMask(Volume& dst,
             const Volume& src,
             const double dt,
             const double ext,
             const double ew,
             const double r)
{
    genMask(dst, src, dt, ext, r);

    int a = CEIL(ew);

    Volume distance(dst.nColRL(),
                    dst.nRowRL(),
                    dst.nSlcRL(),
                    RL_SPACE);

    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(distance)
        distance(i) = FLT_MAX;

    omp_lock_t* mtx = new omp_lock_t[dst.sizeRL()];

    #pragma omp parallel for
    for (int i = 0; i < (int)dst.sizeRL(); i++)
        omp_init_lock(&mtx[i]);

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
        if (dst.getRL(i, j, k) == 1)
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
    for (int i = 0; i < (int)dst.sizeRL(); i++)
        omp_destroy_lock(&mtx[i]);

    delete[] mtx;

    #pragma omp parallel for schedule(dynamic)
    FOR_EACH_PIXEL_RL(dst)
    {
        double d = distance(i);
        if ((d != 0) && (d < ew))
            dst(i) = 0.5 + 0.5 * cos(d / ew * M_PI);
    }
}

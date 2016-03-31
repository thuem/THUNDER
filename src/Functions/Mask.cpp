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

    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        double u = NORM_3(i, j, k);

        if (u > r + ew)
        {
            weightSum += 1;
            sum += vol.getRL(i, j, k);
        }
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);
            weightSum += w;
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

    VOLUME_FOR_EACH_PIXEL_RL(vol)
    {
        double w = 1 - alpha.getRL(i, j, k);
        weightSum += w;
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
              const Image& alpha)
{
    double bg = background(src, alpha);

    IMAGE_FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.getRL(i, j);
        dst.setRL(bg * (1 - w) + w * src.getRL(i, j), i, j);
    }
}

void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew)
{
    double bg = background(src, r, ew);

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

    VOLUME_FOR_EACH_PIXEL_RL(src)
    {
        double w = 1 - alpha.getRL(i, j, k);
        dst.setRL(bg * (1 - w) + w * src.getRL(i, j, k), i, j, k);
    }
}

/***
void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold)
{
    VOLUME_FOR_EACH_PIXEL_RL(src)
        if (src.get(i, j, k) > densityThreshold)
            dst.set(1, i, j, k);
        else
            dst.set(0, i, j, k);
}

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend)
{
    generateMask(dst, src, densityThreshold);

    Volume dstTmp = dst;

    int ext = ceil(abs(extend));

    if (extend > 0)
        VOLUME_FOR_EACH_PIXEL(dst)
            if (dst.get(i, j, k) == 1)
                for (int z = -ext; z < ext; z++)
                    for (int y = -ext; y < ext; y++)
                        for (int x = -ext; x < ext; x++)
                            if ((x * x + y * y + z * z) < extend * extend)
                                dstTmp.setRL(1, i + x, j + y, k + z);
    else if (extend < 0)
        VOLUME_FOR_EACH_PIXEL(dst)
            if (dst.get(i, j, k) == 0)
                for (int z = -ext; z < ext; z++)
                    for (int y = -ext; y < ext; y++)
                        for (int x = -ext; x < ext; x++)
                            if ((x * x + y * y + z * z) < extend * extend)
                                dstTmp.setRL(0, i + x, j + y, k + z);

    dst = dstTmp;
}

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend,
                  const double ew)
{
    generateMask(dst, src, densityThreshold, extend);

    int ew = ceil(ew);

    auto distance = [&dst, ew](const double i,
                               const double j,
                               const double k)
    {
        double result = FLT_MAX;

        for (int z = -ew; z < ew; z++)
            for (int y = -ew; y < ew; y++)
                for (int x = - ew; x < ew; x++)
                    if (((x * x + y * y + z * z) < result * result) &&
                        (dst.get(i + x, j + y, k + z) == 1))
                        result = sqrt(x * x + y * y + z * z);

        return result;
    };

    VOLUME_FOR_EACH_PIXEL(dst)
    {
        double d = distance(i, j, k);
        if ((dst.get(i, j, k) != 1) && (d < ew))
            dst.set(0.5 + 0.5 * cos(d / ew * M_PI), i, j, k);
    }
}
***/

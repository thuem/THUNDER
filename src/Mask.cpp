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

double background(const Volume& vol,
                  const double r,
                  const double ew)
{
    double weightSum = 0;
    double sum = 0;

    VOLUME_FOR_EACH_PIXEL_RL(volume)
    {
        double u = NORM_3(abs(i - vol.nCol() / 2),
                          abs(j - vol.nRow() / 2),
                          abs(k - vol.nSlc() / 2));

        if (u > r + ew)
        {
            weightSum += 1;
            sum += volume.getRL(i, j, k);
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

double background(const Volume& volume,
                  const Volume& alpha)
{
    double weightSum = 0;
    double sum = 0;

    VOLUME_FOR_EACH_PIXEL_RL(volume)
    {
        double w = 1 - alpha.getRL(i, j, k);
        weightSum += w;
        sum += volume.getRL(i, j, k) * w;
    }

    return sum / weightSum;
}

void softMask(Image& img,
              const double r,
              const double ew)
{
    IMAGE_FOR_EACH_PIXEL_RL(img)
    {
        double u = NORM(abs(i - vol.nCol() / 2),
                        abs(j - vol.nRow() / 2));
        if (u > r + ew)
            img.setRL(0, i, j);
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * M_PI);
            img.setRL(img.getRL(i, j) * w, i, j);
        }
    }
}

void softMask(Volume& dst,
              const Volume& src,
              const double radius,
              const double edgeWidth)
{
    double bg = background(src, radius, edgeWidth);

    VOLUME_FOR_EACH_PIXEL(src)
    {
        double r = NORM_3(i, j, k);

        if (r >= radius + edgeWidth)
            dst.setRL(bg, i, j, k);
        else if (r >= radius)
        {
            double w = 0.5 - 0.5 * cos((r - radius) / edgeWidth * M_PI);
            dst.setRL(linear(bg, src.get(i, j, k), w), i, j, k);
        }
    }
}

void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha)
{
    double bg = background(src, alpha);

    VOLUME_FOR_EACH_PIXEL(src)
    {
        double w = 1 - alpha.get(i, j, k);
        dst.set(linear(bg, src.get(i, j, k), w), i, j, k);
    }
}

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold)
{
    VOLUME_FOR_EACH_PIXEL(src)
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
                                dstTmp.set(1, i + x, j + y, k + z);
    else if (extend < 0)
        VOLUME_FOR_EACH_PIXEL(dst)
            if (dst.get(i, j, k) == 0)
                for (int z = -ext; z < ext; z++)
                    for (int y = -ext; y < ext; y++)
                        for (int x = -ext; x < ext; x++)
                            if ((x * x + y * y + z * z) < extend * extend)
                                dstTmp.set(0, i + x, j + y, k + z);

    dst = dstTmp;
}

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend,
                  const double edgeWidth)
{
    generateMask(dst, src, densityThreshold, extend);

    int ew = ceil(edgeWidth);

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
        if ((dst.get(i, j, k) != 1) && (d < edgeWidth))
            dst.set(0.5 + 0.5 * cos(d / edgeWidth * M_PI), i, j, k);
    }
}

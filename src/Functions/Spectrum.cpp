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
#include "Functions/Random.h"

RFLOAT nyquist(const RFLOAT pixelSize)
{
    return 2 / pixelSize;
}

RFLOAT resP2A(const RFLOAT resP,
              const int imageSize,
              const RFLOAT pixelSize)
{
    return resP / imageSize / pixelSize;
}

RFLOAT resA2P(const RFLOAT resA,
              const int imageSize,
              const RFLOAT pixelSize)
{
    return resA * imageSize * pixelSize;
}

void resP2A(vec& res,
            const int imageSize,
            const RFLOAT pixelSize)
{
    res /= imageSize * pixelSize;
}

void resA2P(vec& res,
            const int imageSize,
            const RFLOAT pixelSize)
{
    res *= imageSize * pixelSize;
}

RFLOAT ringAverage(const int resP,
                   const Image& img,
                   const function<RFLOAT(const Complex)> func)
{
    RFLOAT result = 0;
    int counter = 0;

    IMAGE_FOR_EACH_PIXEL_FT(img)
        if (AROUND(NORM(i, j)) == resP)
        {
            result += func(img.getFT(i, j));
            counter++;
        }

    return result / counter;
}

Complex ringAverage(const int resP,
                    const Image& img,
                    const function<Complex(const Complex)> func)
{
    Complex result = COMPLEX(0, 0);
    RFLOAT counter = 0;

    IMAGE_FOR_EACH_PIXEL_FT(img)
        if (AROUND(NORM(i, j)) == resP)
        {
            result += func(img.getFT(i, j));
            counter += 1;
        }

    return result / counter;
}

void ringAverage(vec& dst,
                 const Image& src,
                 const function<RFLOAT(const Complex)> func,
                 const int r)
{
    dst.setZero();

    uvec counter = uvec::Zero(dst.size());

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD(i, j) < TSGSL_pow_2(r))
        {
            int u = AROUND(NORM(i, j));

            if (u < r)
            {
                dst(u) += func(src.getFTHalf(i, j));
                counter(u) += 1;
            }
        }
    }

    for (int i = 0; i < r; i++)
        dst(i) /= counter(i);
}

RFLOAT shellAverage(const int resP,
                    const Volume& vol,
                    const function<RFLOAT(const Complex)> func,
                    const unsigned int nThread)
{
    RFLOAT result = 0;
    int counter = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(vol)
        if (AROUND(NORM_3(i, j, k)) == resP)
        {
            #pragma omp atomic
            result += func(vol.getFT(i, j, k));
            #pragma omp atomic
            counter++;
        }

    return result / counter;
}

void shellAverage(vec& dst,
                  const Volume& src,
                  const function<RFLOAT(const Complex)> func,
                  const int r,
                  const unsigned int nThread)
{
    dst.setZero();

    uvec counter = uvec::Zero(dst.size());

    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD_3(i, j, k) < TSGSL_pow_2(r))
        {
            int u = AROUND(NORM_3(i, j, k));

            if (u < r)
            {
                #pragma omp atomic
                dst(u) += func(src.getFTHalf(i, j, k));
                #pragma omp atomic
                counter(u) += 1;
            }
        }
    }

    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < r; i++)
        dst(i) /= counter(i);
}

void powerSpectrum(vec& dst,
                   const Image& src,
                   const int r,
                   const unsigned int nThread)
{
    dst.setZero();

    uvec counter = uvec::Zero(dst.size());

    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD(i, j) < TSGSL_pow_2(r))
        {
            int u = AROUND(NORM(i, j));

            if (u < r)
            {
                #pragma omp atomic
                dst(u) += ABS2(src.getFT(i, j));
                #pragma omp atomic
                counter(u) += 1;
            }
        }
    }

    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < r; i++)
        dst(i) /= counter(i);
}

void powerSpectrum(vec& dst,
                   const Volume& src,
                   const int r,
                   const unsigned int nThread)
{
    dst.setZero();

    uvec counter = uvec::Zero(dst.size());

    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD_3(i, j, k) < TSGSL_pow_2(r))
        {
            int u = AROUND(NORM_3(i, j, k));

            if (u < r)
            {
                #pragma omp atomic
                dst(u) += ABS2(src.getFT(i, j, k));
                #pragma omp atomic
                counter(u) += 1;
            }
        }
    }

    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < r; i++)
        dst(i) /= counter(i);
}

void FRC(vec& dst,
         const Image& A,
         const Image& B)
{
#ifndef NAN_NO_CHECK
    SEGMENT_NAN_CHECK_COMPLEX(A.dataFT(), A.sizeFT());
    SEGMENT_NAN_CHECK_COMPLEX(B.dataFT(), B.sizeFT());
#endif

    vec vecS = vec::Zero(dst.size());
    vec vecA = vec::Zero(dst.size());
    vec vecB = vec::Zero(dst.size());

    IMAGE_FOR_EACH_PIXEL_FT(A)
    {
        int u = AROUND(NORM(i, j));
        if (u < dst.size())
        {
            vecS[u] += REAL(A.getFT(i, j) * CONJUGATE(B.getFT(i, j)));
            vecA[u] += ABS2(A.getFT(i, j));
            vecB[u] += ABS2(B.getFT(i, j));
        }
    }

    for (int i = 0; i < dst.size(); i++)
    {
        RFLOAT AB = sqrt(vecA(i) * vecB(i));
        if (AB == 0)
            dst(i) = 0;
        else
            dst(i) = vecS(i) / AB;
    }
}

void FRC(vec& dst,
         const Volume& A,
         const Volume& B,
         const int k)
{
#ifndef NAN_NO_CHECK
    SEGMENT_NAN_CHECK_COMPLEX(A.dataFT(), A.sizeFT());
    SEGMENT_NAN_CHECK_COMPLEX(B.dataFT(), B.sizeFT());
#endif

    vec vecS = vec::Zero(dst.size());
    vec vecA = vec::Zero(dst.size());
    vec vecB = vec::Zero(dst.size());

    /***
    for (int j = -A.nRowRL() / 2; j < A.nRowRL() / 2; j++)
        for (int i = 0; i <= A.nColRL() / 2; i++)
    ***/
    IMAGE_FOR_EACH_PIXEL_FT(A)
        {
            int u = AROUND(NORM(i, j));
            if (u < dst.size())
            {
                vecS[u] += REAL(A.getFT(i, j, k) * CONJUGATE(B.getFT(i, j, k)));
                vecA[u] += ABS2(A.getFT(i, j, k));
                vecB[u] += ABS2(B.getFT(i, j, k));
            }
        }

    /***
    std::cout << "vecS = " << vecS << std::endl;
    std::cout << "vecA = " << vecA << std::endl;
    std::cout << "vecB = " << vecB << std::endl;
    ***/

    for (int i = 0; i < dst.size(); i++)
    {
        RFLOAT AB = sqrt(vecA(i) * vecB(i));
        if (AB == 0)
            dst(i) = 0;
        else
            dst(i) = vecS(i) / AB;
    }
}

void FSC(vec& dst,
         const Volume& A,
         const Volume& B)
{
    vec vecS = vec::Zero(dst.size());
    vec vecA = vec::Zero(dst.size());
    vec vecB = vec::Zero(dst.size());

    VOLUME_FOR_EACH_PIXEL_FT(A)
    {
        int u = AROUND(NORM_3(i, j, k));
        if (u < dst.size())
        {
            vecS[u] += REAL(A.getFT(i, j, k) * CONJUGATE(B.getFT(i, j, k)));
            vecA[u] += ABS2(A.getFT(i, j, k));
            vecB[u] += ABS2(B.getFT(i, j, k));
        }
    }

    for (int i = 0; i < dst.size(); i++)
    {
        RFLOAT AB = sqrt(vecA(i) * vecB(i));
        if (AB == 0)
            dst(i) = 0;
        else
            dst(i) = vecS(i) / AB;
    }
}

int resP(const vec& fsc,
         const RFLOAT thres,
         const int pf,
         const int rL,
         const bool inverse)
{
    int result;

    if (inverse)
    {
        for (result = fsc.size() - 1; result >= rL; result--)
            if (fsc(result) > thres) break;
    }
    else
    {
        for (result = rL; result < fsc.size(); result++)
        {
            if (fsc(result) < thres) break;
        }

        result--;
    }

    return result / pf;
}

void randomPhase(Volume& dst,
                 const Volume& src,
                 const int r,
                 const unsigned int nThread)
{
    gsl_rng* engine = get_random_engine();

    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(dst)
    {
        int u = AROUND(NORM_3(i, j, k));

        if (u > r)
            dst.setFT(src.getFT(i, j, k)
                    * COMPLEX_POLAR(TSGSL_ran_flat(engine, 0, 2 * M_PI)),
                      i,
                      j,
                      k);
        else
            dst.setFT(src.getFT(i, j, k), i, j, k);
    }
}

void sharpen(Volume& dst,
             const Volume& src,
             const RFLOAT thres,
             const RFLOAT ew,
             const int rU,
             const int rL,
             const unsigned int nThread)
{
    RFLOAT bFactor;
    bFactorEst(bFactor, src, rU, rL);

    sharpen(dst, src, thres, ew, bFactor, nThread);
}

void sharpen(Volume& dst,
             const Volume& src,
             const RFLOAT thres,
             const RFLOAT ew,
             const RFLOAT bFactor,
             const unsigned int nThread)
{
    bFactorFilter(dst, src, bFactor, nThread);

    lowPassFilter(dst, dst, thres, ew, nThread);
}

void bFactorEst(RFLOAT& bFactor,
                const Volume& vol,
                const int rU,
                const int rL)
{
    vec I = vec::Zero(rU - rL);
    vec C = vec::Zero(rU - rL);

    VOLUME_FOR_EACH_PIXEL_FT(vol)
    {
        int u = AROUND(NORM_3(i, j, k));
        if ((u < rU) && (u >= rL))
        {
            I[u - rL] += ABS(vol.getFT(i, j, k));
            C[u - rL] += 1;
        }
    }
    
    for (int i = 0; i < rU - rL; i++)
    {
        I[i] = log(I[i] / C[i]);
        C[i] = TSGSL_pow_2((RFLOAT)(i + rL) / vol.nColRL());
    }

    RFLOAT c0, c1, cov00, cov01, cov11, sumsq;

    TSGSL_fit_linear(C.data(),
                   1,
                   I.data(),
                   1,
                   rU - rL, 
                   &c0,
                   &c1,
                   &cov00,
                   &cov01,
                   &cov11,
                   &sumsq);

    bFactor = 2 * c1;
}

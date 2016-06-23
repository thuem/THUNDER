/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "CTF.h"

void CTF(Image& dst,
         const double pixelSize,
         const double voltage,
         const double defocusU,
         const double defocusV,
         const double theta,
         const double Cs)
{
    double lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    //cout << "lambda = " << lambda << endl;

    double K1 = M_PI * lambda;
    double K2 = M_PI / 2 * Cs * gsl_pow_3(lambda);

    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        double u = NORM(i / (pixelSize * dst.nColRL()),
                        j / (pixelSize * dst.nRowRL()));

        double angle = atan2(j, i) - theta;
        double defocus = -(defocusU + defocusV
                         + (defocusU - defocusV) * cos(2 * angle)) / 2;

        double ki = K1 * defocus * gsl_pow_2(u) + K2 * gsl_pow_4(u);

        constexpr double w1 = sqrt(1 - CTF_A * CTF_A);
        constexpr double w2 = CTF_A;

        //dst.setFT(COMPLEX(w1 * sin(ki) + w2 * cos(ki), 0),
        dst.setFT(COMPLEX(w1 * sin(ki) - w2 * cos(ki), 0), // CORRECT_ONE
                  i,
                  j);
        /***
        dst.setFT(COMPLEX(cos(K1 * defocus * gsl_pow_2(u)
                            + K2 * gsl_pow_4(u)),
                          0),
                  i,
                  j);
                  ***/
        //dst.setFT(COMPLEX(1, 0), i, j); // for debug
    }
}

void reduceCTF(Image& dst,
               const Image& src,
               const Image& ctf)
{
    FOR_EACH_PIXEL_FT(src)
    {
        //dst[i] = src.iGetFT(i) / (CTF_TAU + REAL(ctf.iGetFT(i)));
        double v = REAL(ctf.iGetFT(i));

        dst[i] = v * src.iGetFT(i) / (gsl_pow_2(v) + CTF_TAU);

        /***
        if (abs(v) > CTF_TAU)
            dst[i] = src.iGetFT(i) / v;
        else
            dst[i] = src.iGetFT(i);
        ***/
    }
}

void reduceCTF(Image& dst,
               const Image& src,
               const Image& ctf,
               const int r)
{
    IMAGE_FOR_PIXEL_R_FT(r + 1)
        if (QUAD(i, j) < r * r)
        {
            double v = REAL(ctf.getFT(i, j));

            dst.setFT(v * src.getFT(i, j) / (gsl_pow_2(v) + CTF_TAU), i, j);

            /***
            if (abs(v) > CTF_TAU)
                dst.setFT(src.getFT(i, j) / v,
                          i,
                          j);
            else
                dst.setFT(src.getFT(i, j), i, j);
            ***/
        }
}

void reduceCTF(Image& dst,
               const Image& src,
               const Image& ctf,
               const vec& sigma,
               const vec& tau,
               const int pf,
               const int r)
{
    IMAGE_FOR_PIXEL_R_FT(r + 1)
    {
        int u = AROUND(NORM(i, j));

        if (u < r)
        {
            double v = REAL(ctf.getFT(i, j));

            /***
            CLOG(INFO, "LOGGER_SYS") << "sigma = " << sigma(u) << endl;
            CLOG(INFO, "LOGGER_SYS") << "tau = " << tau(pf * u) << endl;
            CLOG(INFO, "LOGGER_SYS") << "sigma / tau" << sigma(u) / tau(pf * u) << endl;
            ***/

            dst.setFT(v * src.getFT(i, j)
                    / (gsl_pow_2(v) + sigma(u) / tau(pf * u)),
                      i,
                      j);

            /***
            dst.setFT(v * src.getFT(i, j)
                    / (gsl_pow_2(v) + 0.1),
                      i,
                      j); // debug
            ***/
        }
    }
}

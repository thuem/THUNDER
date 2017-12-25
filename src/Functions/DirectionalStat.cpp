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

#include "DirectionalStat.h"

RFLOAT pdfACG(const vec4& x,
              const mat44& sig)
{
    return pow(sig.determinant(), -0.5)
         * pow(x.transpose() * sig.inverse() * x, -2);
}

RFLOAT pdfACG(const vec4& x,
              const RFLOAT k0,
              const RFLOAT k1)
{
    mat44 sig;
    sig << k0, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k1, 0,
           0, 0, 0, k1;

    return pdfACG(x, sig);
}

void sampleACG(mat4& dst,
               const mat44& src,
               const int n)
{
    // assuming src is a positive definite matrix
    // perform a L*LT decomposition
    LLT<mat44> llt(src);
    mat44 L = llt.matrixL();

    gsl_rng* engine = get_random_engine();

    for (int i = 0; i < n; i++)
    {
        // sample from a standard Gaussian distribution
        vec4 v;
        for (int j = 0; j < 4; j++)
            v(j) = TSGSL_ran_gaussian(engine, 1);

        v = L * v;
        v /= v.norm();

        dst.row(i) = v.transpose();
    }
}

void sampleACG(mat4& dst,
               const RFLOAT k0,
               const RFLOAT k1,
               const int n)
{
    mat44 src;
    src << k0, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k1, 0,
           0, 0, 0, k1;

    sampleACG(dst, src, n);
}

void sampleACG(mat4& dst,
               const RFLOAT k1,
               const RFLOAT k2,
               const RFLOAT k3,
               const int n)
{
    mat44 src;
    src << 1, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k2, 0,
           0, 0, 0, k3;

    sampleACG(dst, src, n);
}

void inferACG(mat44& dst,
              const mat4& src)
{
    mat44 A;
    mat44 B = mat44::Identity();

    do
    {
        A = B;

        B = mat44::Zero();
        RFLOAT nf = 0;

        for (int i = 0; i < src.rows(); i++)
        {
            // get the tensor product of the i-th quaternion and itself
            mat44 tensor;
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    tensor(j, k) = src(i, j) * src(i, k);

            // get the factor
            RFLOAT u = src.row(i) * A.inverse() * src.row(i).transpose();

            B += tensor / u;

            nf += 1.0 / u;
        }

        B *= 4.0 / nf;

        /***
        // make it self-adjoint
        for (int i = 1; i < 4; i++)
            for (int j = 0; j < i; j++)
                B(i, j) = B(j, i);
        ***/
    } while ((abs((A - B).array())).sum() > 1e-3);

    dst = A;

#ifndef NAN_NO_CHECK

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (TSGSL_isnan(dst(i, j)))
            {
                REPORT_ERROR("NAN DETECTED");
                abort();
            }

#endif
}

void inferACG(RFLOAT& k0,
              RFLOAT& k1,
              const mat4& src)
{
    mat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<mat44> eigenSolver(A);

    //vec4 ev = eigenSolver.eigenvalues();

    //ev = ev.cwiseAbs();
    
    k0 = eigenSolver.eigenvalues().maxCoeff();
    k1 = eigenSolver.eigenvalues().minCoeff();
}

void inferACG(RFLOAT& k1,
              RFLOAT& k2,
              RFLOAT& k3,
              const mat4& src)
{
    mat44 A;
    inferACG(A, src);

    k1 = A(1, 1) / A(0, 0);
    k2 = A(2, 2) / A(0, 0);
    k3 = A(3, 3) / A(0, 0);

    /***
    SelfAdjointEigenSolver<mat44> eigenSolver(A);

    vec4 ev = eigenSolver.eigenvalues();

    // sort eigenvalues in ascending sort
    TSGSL_sort(ev.data(), 1, 4);

    k1 = ev(2) / ev(3);
    k2 = ev(1) / ev(3);
    k3 = ev(0) / ev(3);
    ***/
}

void inferACG(vec4& mean,
              const mat4& src)
{
    mat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<mat44> eigenSolver(A);

    int i;

    eigenSolver.eigenvalues().maxCoeff(&i);

    mean = eigenSolver.eigenvectors().col(i);

    mean /= mean.norm();

#ifndef NAN_NO_CHECK
    
    for (int i = 0; i < 4; i++)
        if (TSGSL_isnan(mean(i)))
        {
            REPORT_ERROR("NAN DETECTED");
            abort();
        }

#endif
}

RFLOAT pdfVMS(const vec2& x,
              const vec2& mu,
              const RFLOAT k)
{
    RFLOAT kappa = (1 - k) * (1 + 2 * k - TSGSL_pow_2(k)) / k / (2 - k);

    return exp(kappa * x.dot(mu)) / (2 * M_PI * TSGSL_sf_bessel_I0(kappa));
}

void sampleVMS(mat2& dst,
               const vec2& mu,
               const RFLOAT k,
               const RFLOAT n)
{
    RFLOAT kappa = (1 - k) * (1 + 2 * k - TSGSL_pow_2(k)) / k / (2 - k);

    gsl_rng* engine = get_random_engine();

    if (kappa < 1e-1)
    {
        for (int i = 0; i < n; i++)
            TSGSL_ran_dir_2d(engine, &dst(i, 0), &dst(i, 1));
    }
    else
    {
        RFLOAT a = 1 + sqrt(1 + 4 * TSGSL_pow_2(kappa));
        RFLOAT b = (a - sqrt(2 * a)) / (2 * kappa);
        RFLOAT r = (1 + TSGSL_pow_2(b)) / (2 * b);

        for (int i = 0; i < n; i++)
        {
            RFLOAT f;

            while (true)
            {
                RFLOAT z = cos(M_PI * TSGSL_rng_uniform(engine));

                f = (1 + r * z) / (r + z);

                RFLOAT c = kappa * (r - f);

                RFLOAT u2 = TSGSL_rng_uniform(engine);

                if (c * (2 - c) > u2) break;

                if (log(c / u2) + 1 - c >= 0) break;
            }

            RFLOAT delta0 = sqrt((1 - f) * (f + 1)) * mu(1);
            RFLOAT delta1 = sqrt((1 - f) * (f + 1)) * mu(0);

            if (TSGSL_rng_uniform(engine) > 0.5)
            {
                dst(i, 0) = mu(0) * f + delta0;
                dst(i, 1) = mu(1) * f - delta1;
            }
            else
            {
                dst(i, 0) = mu(0) * f - delta0;
                dst(i, 1) = mu(1) * f + delta1;
            }
        }
    }
}

void sampleVMS(mat4& dst,
               const vec4& mu,
               const RFLOAT k,
               const RFLOAT n)
{
    dst = mat4::Zero(dst.rows(), 4);

    mat2 dst2D = dst.leftCols<2>();

    sampleVMS(dst2D, vec2(mu(0), mu(1)), k, n);

    dst.leftCols<2>() = dst2D;
}

void inferVMS(vec2& mu,
              RFLOAT& k,
              const mat2& src)
{
    mu = vec2::Zero();

    for (int i = 0; i < src.rows(); i++)
    {
        mu(0) += src(i, 0);
        mu(1) += src(i, 1);
    }

    RFLOAT R = mu.norm() / src.rows();

    mu /= mu.norm();

    /***
    R = GSL_MIN_DBL(R, 1 - 1e-3); // for the purpose of avoiding extreme value

    kappa = R * (2 - TSGSL_pow_2(R)) / (1 - TSGSL_pow_2(R));
    ***/

    k = 1 - R;
}

void inferVMS(RFLOAT& k,
              const mat2& src)
{
    vec2 mu;

    inferVMS(mu, k, src);
}

void inferVMS(vec4& mu,
              RFLOAT& k,
              const mat4& src)
{
    vec2 mu2D;

    inferVMS(mu2D, k, src.leftCols<2>());

    mu = vec4(mu2D(0), mu2D(1), 0, 0);
}

void inferVMS(RFLOAT& k,
              const mat4& src)
{
    vec4 mu;

    inferVMS(mu, k, src);
}

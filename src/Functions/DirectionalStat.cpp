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

double pdfACG(const vec4& x,
              const mat44& sig)
{
    return pow(sig.determinant(), -0.5)
         * pow(x.transpose() * sig.inverse() * x, -2);
}

double pdfACG(const vec4& x,
              const double k0,
              const double k1)
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
            v(j) = gsl_ran_gaussian(engine, 1);

        v = L * v;
        v /= v.norm();

        dst.row(i) = v.transpose();
    }
}

void sampleACG(mat4& dst,
               const double k0,
               const double k1,
               const int n)
{
    mat44 src;
    src << k0, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k1, 0,
           0, 0, 0, k1;

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
        double nf = 0;

        for (int i = 0; i < src.rows(); i++)
        {
            // get the tensor product of the i-th quaternion and itself
            mat44 tensor;
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    tensor(j, k) = src(i, j) * src(i, k);

            // get the factor
            double u = src.row(i) * A.inverse() * src.row(i).transpose();

            B += tensor / u;

            nf += 1.0 / u;
        }

        B *= 4.0 / nf;

        // make it self-adjoint
        for (int i = 1; i < 4; i++)
            for (int j = 0; j < i; j++)
                B(i, j) = B(j, i);
    } while ((abs((A - B).array())).sum() > 1e-5);

    dst = A;
}

void inferACG(double& k0,
              double& k1,
              const mat4& src)
{
    mat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<mat44> eigenSolver(A);

    //vec4 ev = eigenSolver.eigenvalues();

    //ev = ev.cwiseAbs();
    
    k0 = eigenSolver.eigenvalues().cwiseAbs().maxCoeff();
    k1 = eigenSolver.eigenvalues().cwiseAbs().minCoeff();
}

void inferACG(vec4& mean,
              const mat4& src)
{
    mat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<mat44> eigenSolver(A);

    int i;

    eigenSolver.eigenvalues().cwiseAbs().maxCoeff(&i);

    mean = eigenSolver.eigenvectors().col(i);

    mean /= mean.norm();
}

double pdfVMS(const vec2& x,
              const vec2& mu,
              const double kappa)
{
    return exp(kappa * x.dot(mu)) / (2 * M_PI * gsl_sf_bessel_I0(kappa));
}

void sampleVMS(mat2& dst,
               const vec2& mu,
               const double kappa,
               const double n)
{
    gsl_rng* engine = get_random_engine();

    if (kappa < 1e-1)
    {
        for (int i = 0; i < n; i++)
            gsl_ran_dir_2d(engine, &dst(i, 0), &dst(i, 1));
    }
    else
    {
        double a = 1 + sqrt(1 + 4 * gsl_pow_2(kappa));
        double b = (a - sqrt(2 * a)) / (2 * kappa);
        double r = (1 + gsl_pow_2(b)) / (2 * b);

        for (int i = 0; i < n; i++)
        {
            double f;

            while (true)
            {
                double z = cos(M_PI * gsl_rng_uniform(engine));

                f = (1 + r * z) / (r + z);

                double c = kappa * (r - f);

                double u2 = gsl_rng_uniform(engine);

                if (c * (2 - c) > u2) break;

                if (log(c / u2) + 1 - c >= 0) break;
            }

            double delta0 = sqrt((1 - f) * (f + 1)) * mu(1);
            double delta1 = sqrt((1 - f) * (f + 1)) * mu(0);

            if (gsl_rng_uniform(engine) > 0.5)
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
               const double kappa,
               const double n)
{
    dst = mat4::Zero(dst.rows(), 4);

    mat2 dst2D = dst.leftCols<2>();

    sampleVMS(dst2D, vec2(mu(0), mu(1)), kappa, n);

    dst.leftCols<2>() = dst2D;
}

void inferVMS(vec2& mu,
              double& kappa,
              const mat2& src)
{
    mu = vec2::Zero();

    for (int i = 0; i < src.rows(); i++)
    {
        mu(0) += src(i, 0);
        mu(1) += src(i, 1);
    }

    double R = mu.norm() / src.rows();

    mu /= mu.norm();

    kappa = R * (2 - gsl_pow_2(R)) / (1 - gsl_pow_2(R));
}

void inferVMS(double& kappa,
              const mat2& src)
{
    vec2 mu;

    inferVMS(mu, kappa, src);
}

void inferVMS(vec4& mu,
              double& kappa,
              const mat4& src)
{
    vec2 mu2D;

    inferVMS(mu2D, kappa, src.leftCols<2>());

    mu = vec4(mu2D(0), mu2D(1), 0, 0);
}

void inferVMS(double& kappa,
              const mat4& src)
{
    vec4 mu;

    inferVMS(mu, kappa, src);
}

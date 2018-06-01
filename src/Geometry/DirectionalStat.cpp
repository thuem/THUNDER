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

double pdfACG(const dvec4& x,
              const dmat44& sig)
{
    return pow(sig.determinant(), -0.5)
         * pow(x.transpose() * sig.inverse() * x, -2);
}

double pdfACG(const dvec4& x,
              const double k0,
              const double k1)
{
    dmat44 sig;
    sig << k0, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k1, 0,
           0, 0, 0, k1;

    return pdfACG(x, sig);
}

void sampleACG(dmat4& dst,
               const dmat44& src,
               const int n)
{
    // assuming src is a positive definite matrix
    // perform a L*LT decomposition
    LLT<dmat44> llt(src);
    dmat44 L = llt.matrixL();

    gsl_rng* engine = get_random_engine();

    for (int i = 0; i < n; i++)
    {
        // sample from a standard Gaussian distribution
        dvec4 v;
        for (int j = 0; j < 4; j++)
            v(j) = gsl_ran_gaussian(engine, 1);

        v = L * v;
        v /= v.norm();

        dst.row(i) = v.transpose();
    }
}

void sampleACG(dmat4& dst,
               const double k0,
               const double k1,
               const int n)
{
    dmat44 src;
    src << k0, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k1, 0,
           0, 0, 0, k1;

    sampleACG(dst, src, n);
}

void sampleACG(dmat4& dst,
               const double k1,
               const double k2,
               const double k3,
               const int n)
{
    dmat44 src;
    src << 1, 0, 0, 0,
           0, k1, 0, 0,
           0, 0, k2, 0,
           0, 0, 0, k3;

    sampleACG(dst, src, n);
}

void inferACG(dmat44& dst,
              const dmat4& src)
{
    dmat44 A;
    dmat44 B = dmat44::Identity();

    do
    {
        A = B;

        B = dmat44::Zero();
        double nf = 0;

        for (int i = 0; i < src.rows(); i++)
        {
            // get the tensor product of the i-th quaternion and itself
            dmat44 tensor;
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    tensor(j, k) = src(i, j) * src(i, k);

            // get the factor
            double u = src.row(i) * A.inverse() * src.row(i).transpose();

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
            if (gsl_isnan(dst(i, j)))
            {
                REPORT_ERROR("NAN DETECTED");
                abort();
            }

#endif
}

void inferACG(double& k0,
              double& k1,
              const dmat4& src)
{
    dmat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<dmat44> eigenSolver(A);

    //dvec4 ev = eigenSolver.eigenvalues();

    //ev = ev.cwiseAbs();
    
    uvec s = d_index_sort_descend(eigenSolver.eigenvalues());
    k0 = eigenSolver.eigenvalues()(s[0]);
    k1 = eigenSolver.eigenvalues()(s[1]);

#ifndef NAN_NO_CHECK
    if (k0 < 0) { CLOG(FATAL, "LOGGER_SYS") << "K0 = " << k0; abort(); };
    if (k1 < 0) { CLOG(FATAL, "LOGGER_SYS") << "K1 = " << k1; abort(); };
#endif
}

void inferACG(double& k,
              const dmat4& src)
{
    double k0, k1;

    inferACG(k0, k1, src);

    k = k1 / k0;

#ifndef NAN_NO_CHECK
    if (gsl_isnan(k)) { REPORT_ERROR("NAN DETECTED"); abort(); };
#endif
}

void inferACG(double& k1,
              double& k2,
              double& k3,
              const dmat4& src)
{
    dmat44 A;
    inferACG(A, src);

    k1 = A(1, 1) / A(0, 0);
    k2 = A(2, 2) / A(0, 0);
    k3 = A(3, 3) / A(0, 0);

#ifndef NAN_NO_CHECK

    if (gsl_isnan(k1)) { REPORT_ERROR("NAN DETECTED"); abort(); };
    if (gsl_isnan(k2)) { REPORT_ERROR("NAN DETECTED"); abort(); };
    if (gsl_isnan(k3)) { REPORT_ERROR("NAN DETECTED"); abort(); };

#endif

#ifndef NAN_NO_CHECK
    if (k1 < 0) { CLOG(FATAL, "LOGGER_SYS") << "K1 = " << k1; abort(); };
    if (k2 < 0) { CLOG(FATAL, "LOGGER_SYS") << "K2 = " << k2; abort(); };
    if (k3 < 0) { CLOG(FATAL, "LOGGER_SYS") << "K3 = " << k3; abort(); };
#endif

    /***
    SelfAdjointEigenSolver<dmat44> eigenSolver(A);

    dvec4 ev = eigenSolver.eigenvalues();

    // sort eigenvalues in ascending sort
    gsl_sort(ev.data(), 1, 4);

    k1 = ev(2) / ev(3);
    k2 = ev(1) / ev(3);
    k3 = ev(0) / ev(3);
    ***/
}

void inferACG(dvec4& mean,
              const dmat4& src)
{
    dmat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<dmat44> eigenSolver(A);

    int i;

    eigenSolver.eigenvalues().maxCoeff(&i);

    mean = eigenSolver.eigenvectors().col(i);

    mean /= mean.norm();

#ifndef NAN_NO_CHECK
    
    for (int i = 0; i < 4; i++)
        if (gsl_isnan(mean(i)))
        {
            REPORT_ERROR("NAN DETECTED");
            abort();
        }

#endif
}

double pdfVMS(const dvec2& x,
              const dvec2& mu,
              const double k)
{
    double kappa = (1 - k) * (1 + 2 * k - gsl_pow_2(k)) / k / (2 - k);

    if (kappa < 5) // avoiding overflow
        return exp(kappa * x.dot(mu)) / (2 * M_PI * gsl_sf_bessel_I0(kappa));
    else
        return gsl_ran_gaussian_pdf((x - mu).norm(), sqrt(1.0 / kappa));
}

void sampleVMS(dmat2& dst,
               const vec2& mu,
               const double k,
               const double n)
{
    double kappa = (1 - k) * (1 + 2 * k - gsl_pow_2(k)) / k / (2 - k);

    gsl_rng* engine = get_random_engine();

    if (kappa < 1e-1) // avoiding overflow
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

void sampleVMS(dmat4& dst,
               const dvec4& mu,
               const double k,
               const double n)
{
    dst = dmat4::Zero(dst.rows(), 4);

    dmat2 dst2D = dst.leftCols<2>();

    sampleVMS(dst2D, vec2(mu(0), mu(1)), k, n);

    dst.leftCols<2>() = dst2D;
}

void inferVMS(dvec2& mu,
              double& k,
              const dmat2& src)
{
    mu = dvec2::Zero();

    for (int i = 0; i < src.rows(); i++)
    {
        mu(0) += src(i, 0);
        mu(1) += src(i, 1);
    }

    double R = mu.norm() / src.rows();

    mu /= mu.norm();

    /***
    R = gsl_MIN_double(R, 1 - 1e-3); // for the purpose of avoiding extreme value

    kappa = R * (2 - gsl_pow_2(R)) / (1 - gsl_pow_2(R));
    ***/

    k = 1 - R;
}

void inferVMS(double& k,
              const dmat2& src)
{
    dvec2 mu;

    inferVMS(mu, k, src);
}

void inferVMS(dvec4& mu,
              double& k,
              const dmat4& src)
{
    dvec2 mu2D;

    inferVMS(mu2D, k, src.leftCols<2>());

    mu = dvec4(mu2D(0), mu2D(1), 0, 0);
}

void inferVMS(double& k,
              const dmat4& src)
{
    dvec4 mu;

    inferVMS(mu, k, src);
}

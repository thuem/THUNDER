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
    /***
    cout << sig.determinant() << endl;
    cout << sig.inverse() << endl;
    ***/
    /***
    cout << pow(sig.determinant(), -0.5) << endl;
    cout << pow(x.transpose() * sig.inverse() * x, -2) << endl;
    ***/
    return pow(sig.determinant(), -0.5)
         * pow(x.transpose() * sig.inverse() * x, -2);
    /***
    return 1;
    ***/
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

    auto engine = get_random_engine();

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

            /***
            cout << tensor << endl << endl;
            cout << u << endl << endl;
            ***/

            B += tensor / u;

            nf += 1.0 / u;
        }

        B *= 4.0 / nf;

        // make it self-adjoint
        for (int i = 1; i < 4; i++)
            for (int j = 0; j < i; j++)
                B(i, j) = B(j, i);
    } while ((abs((A - B).array())).sum() > 0.001);

    dst = A;
}

void inferACG(double& k0,
              double& k1,
              const mat4& src)
{
    mat44 A;
    inferACG(A, src);

    SelfAdjointEigenSolver<mat44> eigenSolver(A);
    
    k0 = eigenSolver.eigenvalues().maxCoeff();
    k1 = eigenSolver.eigenvalues().minCoeff();
}

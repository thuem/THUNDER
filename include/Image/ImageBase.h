/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_BASE
#define IMAGE_BASE

#include <functional>
#include <cstring>
#include <cstdio>

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "Complex.h"
#include "Typedef.h"
#include "Functions.h"
#include "Utils.h"
#include "Logging.h"

using namespace std;

#define RL_SPACE 0
#define FT_SPACE 1

#define FOR_EACH_PIXEL_RL(base) \
    for (size_t i = 0; i < base.sizeRL(); i++)

#define FOR_EACH_PIXEL_FT(base) \
    for (size_t i = 0; i < base.sizeFT(); i++)

#define SET_0_RL(base) \
    memset(&base(0), 0, sizeof(double) * base.sizeRL());

#define SET_0_FT(base) \
    memset(&base[0], 0, sizeof(Complex) * base.sizeFT());

#define SET_1_RL(base) \
    FOR_EACH_PIXEL_RL(base) \
        base(i) = 1

#define SET_1_FT(base) \
    FOR_EACH_PIXEL_FT(base) \
        base[i] = COMPLEX(1, 0)

#define NEG_RL(base) \
    SCALE_RL(base, -1)

#define NEG_FT(base) \
    SCALE_FT(base, -1)

#define SCALE_RL(base, a) \
    cblas_dscal(base.sizeRL(), a, &base(0), 1)

#define SCALE_FT(base, a) \
    FOR_EACH_PIXEL_FT(base) \
        base[i] *= a

#define ADD_RL(a, b) \
    cblas_daxpy(a.sizeRL(), 1, &b(0), 1, &a(0), 1);

#define ADD_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] += (b)[i]

#define SUB_RL(a, b) \
    cblas_daxpy(a.sizeRL(), -1, &b(0), 1, &a(0), 1);

#define SUB_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] -= (b)[i]

#define MUL_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] *= (b)[i]

#define DIV_FT(a, b) \
    FOR_EACH_PIXEL_FT(a) \
        (a)[i] /= (b)[i]

class ImageBase
{
    MAKE_DEFAULT_MOVE(ImageBase)

    protected:

        unique_ptr<double[]> _dataRL;
        unique_ptr<Complex[]> _dataFT;

        size_t _sizeRL = 0;
        size_t _sizeFT = 0;

        ImageBase();
        ~ImageBase();

    public:

        const double& iGetRL(size_t i = 0) const;
        // return a const pointer which points to the i-th element

        const Complex& iGetFT(size_t i = 0) const;
        // return a const pointer which points to the i-th element

        double& operator()(const size_t i);

        Complex& operator[](const size_t i);

        bool isEmptyRL() const;
        // check whether _data is NULL or not

        bool isEmptyFT() const;
        // check whether _dataFT is NULL or not

        size_t sizeRL() const;
        // return the total size of this image

        size_t sizeFT() const;
        // return the total size of the Fourier transformed image
        
        void clear();
        // free the memory

        void clearRL();
        // free the memory storing real space image

        void clearFT();
        // free the memory storing Fourier Transform image

        void copyBase(ImageBase&) const;
        ImageBase copyBase() const
        {
            ImageBase res;
            copyBase(res);
            return res;
        }
};

double norm(ImageBase& base);

void normalise(ImageBase& base);

#endif // IMAGE_BASE

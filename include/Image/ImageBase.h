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

class ImageBase
{
    protected:

        double* _dataRL = NULL;
        Complex* _dataFT = NULL;

        size_t _sizeRL = 0;
        size_t _sizeFT = 0;

    public:

        ImageBase();

        ImageBase(const ImageBase& that);

        ImageBase& operator=(const ImageBase& that);

        const double& getRL(size_t i = 0) const;
        // return a const pointer which points to the i-th element

        const Complex& getFT(size_t i = 0) const;
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
        
    public:

        virtual void isSameSize(ImageBase& that) {};
};

void normalise(ImageBase& image);

#define FOR_EACH_PIXEL_RL(base) \
    for (size_t i = 0; i < base.sizeRL(); i++)

#define FOR_EACH_PIXEL_FT(base) \
    for (size_t i = 0; i < base.sizeFT(); i++)

#endif // IMAGE_BASE 

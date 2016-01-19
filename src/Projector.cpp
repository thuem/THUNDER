/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Projector.h"

Projector::Projector() {}

Projector::Projector(const Interpolation3DStyle interpolation)
{
    _interpolation = interpolation;
}

Projector::Projector(const Projector& that)
{
    *this = that;
}

Projector::~Projector() {}

Projector& Projector::operator=(const Projector& that)
{
    _maxRadius = that.maxRadius();
    _interpolation = that.interpolation();
    _projectee = that.projectee();

    return *this;
}

bool Projector::isEmpty() const
{
    return _projectee.isEmptyFT();
}

int Projector::maxRadius() const
{
    return _maxRadius;
}

void Projector::setMaxRadius(const int maxRadius)
{
    _maxRadius = maxRadius;
}

Interpolation3DStyle Projector::interpolation() const
{
    return _interpolation;
}

void Projector::setInterpolation(const Interpolation3DStyle interpolation)
{
    _interpolation = interpolation;
}

const Volume& Projector::projectee() const
{
    return _projectee;
}

void Projector::setProjectee(const Volume& src)
{
    _projectee = src;

    // meshReverse(_projectee);

    /***
    VOLUME_FOR_EACH_PIXEL_FT(_projectee)
        printf("%f %f\n", REAL(_projectee.getFT(i, j, k)),
                          IMAG(_projectee.getFT(i, j, k)));
    ***/

    // set _maxRadius
    _maxRadius = floor(MIN_3(_projectee.nColRL(),
                             _projectee.nRowRL(),
                             _projectee.nSlcRL()) / 2 - 1);
}

void Projector::project(Image& dst,
                        const mat33& mat) const
{
    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        /***
        size_t index;
        if (j < 0)
            index = (j + dst.nRow()) * (dst.nColumn() / 2 + 1) + i;
        else
            index = j * (dst.nColumn() / 2 + 1) + i;

        // set a vector points to the point on the image
        ***/
        vec3 newCor = {i, j, 0};
        // std::cout << newCor << std::endl;
        vec3 oldCor = mat * newCor;
        // std::cout << oldCor << std::endl;
        /***
        Vector<double> newCor(3);
        newCor(0) = i;
        newCor(1) = j;
        newCor(2) = 0;
        ***/

        // Vector<double> oldCor = rotateMatrix * newCor;

        if (norm(oldCor) < _maxRadius)
        {
            // dst.setFT(COMPLEX(0, 0), i, j);
            dst.setFT(_projectee.getByInterpolationFT(oldCor(0),
                                                      oldCor(1),
                                                      oldCor(2),
                                                      _interpolation),
                      i,
                      j);
            /***
            printf("%f %f\n",
                   REAL(dst.getFT(i, j)),
                   IMAG(dst.getFT(i, j)));
            ***/
        }
        else
            dst.setFT(COMPLEX(0, 0), i, j);
    }

    // meshReverse(dst);
}

void Projector::project(Image& dst,
                        const double phi,
                        const double theta,
                        const double psi) const
{
    mat33 mat;
    rotate3D(mat, phi, theta, psi);

    project(dst, mat);
}

void Projector::project(Image& dst,
                        const double phi,
                        const double theta,
                        const double psi,
                        const double x,
                        const double y) const
{
    project(dst, phi, theta, psi);
    translate(dst, dst, x, y);
}

void Projector::project(Image& dst,
                        const Coordinate5D& coordinate5D) const
{
    project(dst,
            coordinate5D.phi,
            coordinate5D.theta,
            coordinate5D.psi,
            coordinate5D.x,
            coordinate5D.y);
}

void Projector::gridCorrection()
{
    /***
    FFT fft;

    fft.complexToFloat(_projectee);

    VOLUME_FOR_EACH_PIXEL(_projectee)
    {
        if ((i != 0) || (j != 0) || (k != 0))
        {
            double u = sinc(M_PI * sqrt(i * i + j * j + k * k)
                         / _projectee.nColumn());

            // when sinc3D, no correction needed
            switch (_interpolation)
            {
                case nearest3D:
                    _projectee.set(_projectee.get(i, j, k) / u, i, j, k);
                    break;

                case linear3D:
                    _projectee.set(_projectee.get(i, j, k) / u / u, i, j, k);
                    break;
            }
        }
    }
    ***/
}

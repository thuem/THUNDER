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

Projector::Projector(const Projector& that)
{
    *this = that;
}

Projector::~Projector() {}

Projector& Projector::operator=(const Projector& that)
{
    _maxRadius = that.maxRadius();
    _interp = that.interp();
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

int Projector::interp() const
{
    return _interp;
}

void Projector::setInterp(const int interp)
{
    _interp= interp;
}

int Projector::pf() const
{
    return _pf;
}

void Projector::setPf(const int pf)
{
    _pf = pf;
}

const Volume& Projector::projectee() const
{
    return _projectee;
}

void Projector::setProjectee(const Volume& src)
{
    _projectee = src;

    // make sure the scale correct
    SCALE_FT(_projectee, 1.0 / _pf);
    SCALE_FT(_projectee, 1.0 / sqrt(_projectee.nColRL()));

    _maxRadius = floor(MIN_3(_projectee.nColRL(),
                             _projectee.nRowRL(),
                             _projectee.nSlcRL()) / _pf / 2 - 1);
}

void Projector::project(Image& dst,
                        const mat33& mat) const
{
    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        vec3 newCor = {(double)i, (double)j, 0};
        // std::cout << newCor << std::endl;
        vec3 oldCor = mat * newCor * _pf;
        // std::cout << oldCor << std::endl;

        if (norm(oldCor) < _maxRadius * _pf)
            dst.setFT(_projectee.getByInterpolationFT(oldCor(0),
                                                      oldCor(1),
                                                      oldCor(2),
                                                      _interp),
                      i,
                      j);
    }
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
    translate(dst, dst, _maxRadius, x, y);
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

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

Projector::~Projector() {}

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

void Projector::setProjectee(Volume src)
{
    _projectee = std::move(src);

    // make sure the scale correct
    // SCALE_FT(_projectee, 1.0 / _pf);
    // SCALE_FT(_projectee, 1.0 / _pf / sqrt(_pf * _projectee.nColRL()));
    // SCALE_FT(_projectee, 1.0 / _pf / sqrt(_projectee.nColRL()));

    _maxRadius = floor(MIN_3(_projectee.nColRL(),
                             _projectee.nRowRL(),
                             _projectee.nSlcRL()) / _pf / 2 - 1);

    /***
    // perform grid correction
    gridCorrection();
    ***/
}

void Projector::project(Image& dst,
                        const mat33& mat) const
{
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
    {
        if (QUAD(i, j) < _maxRadius * _maxRadius)
        {
            vec3 newCor = {(double)i, (double)j, 0};
            vec3 oldCor = mat * newCor * _pf;

            dst.setFT(_projectee.getByInterpolationFT(oldCor(0),
                                                      oldCor(1),
                                                      oldCor(2),
                                                      _interp),
                      i,
                      j);
        }
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

void Projector::project(Image& dst,
                        const mat33& rot,
                        const vec2& t) const
{
    project(dst, rot);
    translate(dst, dst, _maxRadius, t(0), t(1));
}

void Projector::gridCorrection()
{
    if ((_interp == LINEAR_INTERP) ||
        (_interp == NEAREST_INTERP))
    {
        FFT fft;

        fft.bw(_projectee);

        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(_projectee)
        {
            double r = NORM_3(i, j, k) / (_projectee.nColRL() * _pf);

            _projectee.setRL(_projectee.getRL(i, j, k) / TIK_RL(r), i, j, k);
        }

        fft.fw(_projectee);
    }
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

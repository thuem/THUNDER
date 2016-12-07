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
    _interp = interp;
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
    _projectee.swap(src);

    // make sure the scale correct
    // SCALE_FT(_projectee, 1.0 / _pf);
    // SCALE_FT(_projectee, 1.0 / _pf / sqrt(_pf * _projectee.nColRL()));
    // SCALE_FT(_projectee, 1.0 / _pf / sqrt(_projectee.nColRL()));

    _maxRadius = floor(MIN_3(_projectee.nColRL(),
                             _projectee.nRowRL(),
                             _projectee.nSlcRL()) / _pf / 2 - 1);

    // perform grid correction
    gridCorrection();
}

void Projector::project(Image& dst,
                        const mat33& mat) const
{
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
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

void Projector::project(Image& dst,
                        const mat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    for (int i = 0; i < nPxl; i++)
    {
        vec3 newCor = {(double)iCol[i], (double)iRow[i], 0};
        vec3 oldCor = mat * newCor * _pf;

        dst[iPxl[i]] = _projectee.getByInterpolationFT(oldCor(0),
                                                       oldCor(1),
                                                       oldCor(2),
                                                       _interp);
    }
}

void Projector::projectMT(Image& dst,
                          const mat33& mat) const
{
    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
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

void Projector::projectMT(Image& dst,
                          const mat33& mat,
                          const int* iCol,
                          const int* iRow,
                          const int* iPxl,
                          const int nPxl) const
{
    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        vec3 newCor = {(double)iCol[i], (double)iRow[i], 0};
        vec3 oldCor = mat * newCor * _pf;

        dst[iPxl[i]] = _projectee.getByInterpolationFT(oldCor(0),
                                                       oldCor(1),
                                                       oldCor(2),
                                                       _interp);
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

void Projector::projectMT(Image& dst,
                          const double phi,
                          const double theta,
                          const double psi) const
{
    mat33 mat;
    rotate3D(mat, phi, theta, psi);

    projectMT(dst, mat);
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
                        const double phi,
                        const double theta,
                        const double psi,
                        const double x,
                        const double y,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    mat33 mat;
    rotate3D(mat, phi, theta, psi);

    project(dst, mat, iCol, iRow, iPxl, nPxl);

    translate(dst, dst, x, y, iCol, iRow, iPxl, nPxl);
}

void Projector::projectMT(Image& dst,
                          const double phi,
                          const double theta,
                          const double psi,
                          const double x,
                          const double y) const
{
    projectMT(dst, phi, theta, psi);

    translateMT(dst, dst, _maxRadius, x, y);
}

void Projector::projectMT(Image& dst,
                          const double phi,
                          const double theta,
                          const double psi,
                          const double x,
                          const double y,
                          const int* iCol,
                          const int* iRow,
                          const int* iPxl,
                          const int nPxl) const
{
    mat33 mat;
    rotate3D(mat, phi, theta, psi);

    projectMT(dst, mat, iCol, iRow, iPxl, nPxl);

    translateMT(dst, dst, x, y, iCol, iRow, iPxl, nPxl);
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

void Projector::projectMT(Image& dst,
                          const Coordinate5D& coordinate5D) const
{
    projectMT(dst,
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

void Projector::project(Image& dst,
                        const mat33& rot,
                        const vec2& t,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    project(dst, rot, iCol, iRow, iPxl, nPxl);

    //translate(dst, dst, _maxRadius, t(0), t(1));
    translate(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
}

void Projector::projectMT(Image& dst,
                          const mat33& rot,
                          const vec2& t) const
{
    projectMT(dst, rot);

    translateMT(dst, dst, _maxRadius, t(0), t(1));
}

void Projector::projectMT(Image& dst,
                          const mat33& rot,
                          const vec2& t,
                          const int* iCol,
                          const int* iRow,
                          const int* iPxl,
                          const int nPxl) const
{
    projectMT(dst, rot, iCol, iRow, iPxl, nPxl);

    //translateMT(dst, dst, _maxRadius, t(0), t(1));
    translateMT(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
}

void Projector::gridCorrection()
{
    if ((_interp == LINEAR_INTERP) ||
        (_interp == NEAREST_INTERP))
    {
        FFT fft;

        fft.bwMT(_projectee);

        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(_projectee)
        {
            double r = NORM_3(i, j, k) / (_projectee.nColRL() * _pf);

            if (r < 0.25 / _pf * PROJ_LOOSE_FACTOR)
                _projectee.setRL(_projectee.getRL(i, j, k)
                               / TIK_RL(r),
                                 i,
                                 j,
                                 k);
            else
                _projectee.setRL(0, i, j, k);
        }

        fft.fwMT(_projectee);
    }
}

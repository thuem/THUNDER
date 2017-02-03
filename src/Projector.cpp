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

Projector::Projector()
{
    _mode = MODE_3D;

    _maxRadius = -1;

    _interp = LINEAR_INTERP;

    _pf = 2;
}

Projector::~Projector() {}

void Projector::swap(Projector& that)
{
    std::swap(_maxRadius, that._maxRadius);
    std::swap(_interp, that._interp);
    std::swap(_pf, that._pf);
    
    _projectee2D.swap(that._projectee2D);
    _projectee3D.swap(that._projectee3D);
}

bool Projector::isEmpty2D() const
{
    return _projectee2D.isEmptyFT();
}

bool Projector::isEmpty3D() const
{
    return _projectee3D.isEmptyFT();
}

int Projector::mode() const
{
    return _mode;
}

void Projector::setMode(const int mode)
{
    _mode = mode;
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

const Image& Projector::projectee2D() const
{
    return _projectee2D;
}

const Volume& Projector::projectee3D() const
{
    return _projectee3D;
}

void Projector::setProjectee(Image src)
{
    _projectee2D.swap(src);

    _maxRadius = floor(MIN(_projectee2D.nColRL(),
                           _projectee2D.nRowRL()) / _pf / 2 - 1);

    // perform grid correction
    gridCorrection();
}

void Projector::setProjectee(Volume src)
{
    _projectee3D.swap(src);

    _maxRadius = floor(MIN_3(_projectee3D.nColRL(),
                             _projectee3D.nRowRL(),
                             _projectee3D.nSlcRL()) / _pf / 2 - 1);

    // perform grid correction
    gridCorrection();
}

void Projector::project(Image& dst,
                        const mat22& mat) const
{
    // TODO
}

void Projector::project(Image& dst,
                        const mat33& mat) const
{
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            vec3 newCor((double)(i * _pf), (double)(j * _pf), 0);
            vec3 oldCor = mat * newCor;

            dst.setFT(_projectee3D.getByInterpolationFT(oldCor(0),
                                                        oldCor(1),
                                                        oldCor(2),
                                                        _interp),
                      i,
                      j);
        }
}

void Projector::project(Image& dst,
                        const mat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    //TODO
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
        vec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        vec3 oldCor = mat * newCor;

        dst[iPxl[i]] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                         oldCor(1),
                                                         oldCor(2),
                                                         _interp);
    }
}

void Projector::project(Complex* dst,
                        const mat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl) const
{
    //TODO
}

void Projector::project(Complex* dst,
                        const mat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl) const
{
    for (int i = 0; i < nPxl; i++)
    {
        vec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        vec3 oldCor = mat * newCor;

        dst[i] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                   oldCor(1),
                                                   oldCor(2),
                                                   _interp);
    }
}

void Projector::projectMT(Image& dst,
                          const mat22& mat) const
{
    //TODO
}

void Projector::projectMT(Image& dst,
                          const mat33& mat) const
{
    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            vec3 newCor((double)(i * _pf), (double)(j * _pf), 0);
            vec3 oldCor = mat * newCor;

            dst.setFT(_projectee3D.getByInterpolationFT(oldCor(0),
                                                        oldCor(1),
                                                        oldCor(2),
                                                        _interp),
                      i,
                      j);
        }
}

void Projector::projectMT(Image& dst,
                          const mat22& mat,
                          const int* iCol,
                          const int* iRow,
                          const int* iPxl,
                          const int nPxl) const
{
    //TODO
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
        vec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        vec3 oldCor = mat * newCor;

        dst[iPxl[i]] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                         oldCor(1),
                                                         oldCor(2),
                                                         _interp);
    }
}

void Projector::projectMT(Complex* dst,
                          const mat22& mat,
                          const int* iCol,
                          const int* iRow,
                          const int nPxl) const
{
    //TODO
}

void Projector::projectMT(Complex* dst,
                          const mat33& mat,
                          const int* iCol,
                          const int* iRow,
                          const int nPxl) const
{
    #pragma omp parallel for
    for (int i = 0; i < nPxl; i++)
    {
        vec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        vec3 oldCor = mat * newCor;

        dst[i] = _projectee3D.getByInterpolationFT(oldCor(0),
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
                        const mat22& rot,
                        const vec2& t) const
{
    //TODO
}

void Projector::project(Image& dst,
                        const mat33& rot,
                        const vec2& t) const
{
    project(dst, rot);
    translate(dst, dst, _maxRadius, t(0), t(1));
}

void Projector::project(Image& dst,
                        const mat22& rot,
                        const vec2& t,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    //TODO
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

    translate(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
}

void Projector::project(Complex* dst,
                        const mat22& rot,
                        const vec2& t,
                        const int nCol,
                        const int nRow,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl) const
{
    //TODO
}

void Projector::project(Complex* dst,
                        const mat33& rot,
                        const vec2& t,
                        const int nCol,
                        const int nRow,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl) const
{
    project(dst, rot, iCol, iRow, nPxl);

    translate(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl);
}

void Projector::projectMT(Image& dst,
                          const mat22& rot,
                          const vec2& t) const
{
    //TODO
}

void Projector::projectMT(Image& dst,
                          const mat33& rot,
                          const vec2& t) const
{
    projectMT(dst, rot);

    translateMT(dst, dst, _maxRadius, t(0), t(1));
}

void Projector::projectMT(Image& dst,
                          const mat22& rot,
                          const vec2& t,
                          const int* iCol,
                          const int* iRow,
                          const int* iPxl,
                          const int nPxl) const
{
    projectMT(dst, rot, iCol, iRow, iPxl, nPxl);

    translateMT(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
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

    translateMT(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
}

void Projector::projectMT(Complex* dst,
                          const mat22& rot,
                          const vec2& t,
                          const int nCol,
                          const int nRow,
                          const int* iCol,
                          const int* iRow,
                          const int nPxl) const
{
    projectMT(dst, rot, iCol, iRow, nPxl);

    translateMT(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl);
}

void Projector::projectMT(Complex* dst,
                          const mat33& rot,
                          const vec2& t,
                          const int nCol,
                          const int nRow,
                          const int* iCol,
                          const int* iRow,
                          const int nPxl) const
{
    projectMT(dst, rot, iCol, iRow, nPxl);

    translateMT(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl);
}

void Projector::gridCorrection()
{
    if ((_interp == LINEAR_INTERP) ||
        (_interp == NEAREST_INTERP))
    {
        FFT fft;

        fft.bwMT(_projectee3D);

        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(_projectee3D)
            _projectee3D.setRL(_projectee3D.getRL(i, j, k)
                             / TIK_RL(NORM_3(i, j, k)
                                    / (_projectee3D.nColRL() * _pf)),
                               i,
                               j,
                               k);

        fft.fwMT(_projectee3D);
    }
}

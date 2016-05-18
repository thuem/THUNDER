/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef PROJECTOR_H
#define PROJECTOR_H

#include "Complex.h"
#include "Error.h"
#include "Enum.h"

#include "Euler.h"

#include "Image.h"
#include "Volume.h"

#include "Coordinate5D.h"

#include "ImageFunctions.h"

class Projector
{
    MAKE_DEFAULT_MOVE(Projector)

    private:

        int _maxRadius = -1;
        // Only the signal in the _maxRadius circle in Fourier space will be
        // processed. Otherwise, it will set to 0.
        // _maxRadius should be smaller than half of the shortest dimension
        // of the _projectee.
        // _maxRadius will be automatically set to the properly value when
        // projectee is set.

        int _interp = LINEAR_INTERP;

        int _pf = 2; // padding factor

        Volume _projectee;

    public:

        Projector();

        ~Projector();

        bool isEmpty() const;

        int maxRadius() const;

        void setMaxRadius(const int maxRadius);

        int interp() const;

        void setInterp(const int interp);

        int pf() const;

        void setPf(const int pf);

        const Volume& projectee() const;

        void setProjectee(Volume src);

        void project(Image& dst,
                     const mat33& mat) const;

        void project(Image& dst,
                     const double phi,
                     const double theta,
                     const double psi) const;

        void project(Image& dst,
                     const double phi,
                     const double theta,
                     const double psi,
                     const double x,
                     const double y) const;

        void project(Image& dst,
                     const Coordinate5D& coordinate5D) const;

        void project(Image& dst,
                     const mat33& rot,
                     const vec2& t) const;

    private:

        void gridCorrection();
        /* perform gridding correction on _projectee */
};

#endif // PROJECTOR_H

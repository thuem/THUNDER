/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <memory>
#include <vector>

#include "Typedef.h"

#include "Image.h"
#include "Volume.h"
#include "Parallel.h"
#include "Filter.h"
#include "Spectrum.h"
#include "Projector.h"
#include "Symmetry.h"
#include "Reconstructor.h"
#include "Particle.h"

#define FOR_EACH_CLASS \
    for (int i = 0; i < _k; i++)

#define MAX_GAP 20
#define MIN_GAP 10

#define A_B_AVERAGE_THRES 40 // Angstrom

using namespace std;

class MLModel : public Parallel
{
    private:

        /**
         * references in Fourier space
         */
        vector<Volume> _ref;

        /**
         * Fourier Shell Coefficient
         * each column stands for a FSC of a certain reference
         */
        mat _FSC;

        /**
         * Signal Noise Ratio
         * each column stands for a SNR of a certain reference
         */
        mat _SNR;

        /**
         * projectors
         */
        vector<Projector> _proj;

        /**
         * reconstructors
         */
        vector<unique_ptr<Reconstructor>> _reco;

        /**
         * number of references
         */
        int _k;

        /**
         * size of references bfore padding
         */
        int _size;

        /**
         * radius of calculating FSC and SNR before padding
         */
        int _r;

        /**
         * padding factor
         */
        int _pf;

        /**
         * pixel size of 2D images (in Angstrom)
         */
        double _pixelSize;

        /**
         * width of modified Kaiser-Bessel function
         */
        double _a = 1.9;

        /**
         * smoothness parameter of modified Kaiser-Bessel function
         */
        double _alpha;

        /**
         * the concentration parameter of the rotation
         */
        double _rVari;

        /**
         * variance 2D Gaussian distribution of the translation in X
         */
        double _tVariS0;

        /**
         * variance 2D Gaussian distribution of the translation in Y
         */
        double _tVariS1;

        /**
         * a parameter indicating the change of rotation between iterations
         */
        double _rChange;

        const Symmetry* _sym = NULL;

    public:

        MLModel();

        ~MLModel();

        void init(const int k,
                  const int size,
                  const int r,
                  const int pf,
                  const double pixelSize,
                  const double a,
                  const double alpha,
                  const Symmetry* sym);

        void initProjReco();
        /* initialise Projectors and Reconstructors */

        Volume& ref(const int i);

        void appendRef(Volume ref);

        int k() const;

        int size() const;

        int r() const;

        void setR(const int r);

        Projector& proj(const int i = 0);

        Reconstructor& reco(const int i = 0);

        void BcastFSC();

        void lowPassRef(const double thres,
                        const double ew);
        /* perform a low pass filtering on each reference */

        void refreshSNR();

        int resolutionP(const int i) const;
        /* get the resolution of _ref[i] */

        int resolutionP() const;
        /* get the highest resolution among all references */

        double resolutionA(const int i) const;
        /* get the resolution of _ref[i] */

        double resolutionA() const;
        /* get the highest resolution among all references */

        void refreshProj();

        void refreshReco();

        void updateR();
        /* increase _r according to wether FSC is high than 0.2 at current _r */

        double rVari() const;

        double tVariS0() const;
        
        double tVariS1() const;

        /**
         * @param par a vector of Particle
         * @param n number of images in the hemisphere
         */
        void allReduceVari(const vector<Particle>& par,
                           const int n);

        void clear();

    private:

        /***
        void addRVari(const double rVari);

        void addTVariS0(const double tVariS0);

        void addTVariS1(const double tVariS1);
        ***/
};

#endif // ML_MODEL_H

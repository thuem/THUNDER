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

#define SEARCH_TYPE_GLOBAL 0

#define SEARCH_TYPE_LOCAL 1

#define A_B_AVERAGE_THRES 40 // Angstrom

#define MAX_ITER_R_CHANGE_NO_DECREASE 2

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
         * (_r * _pf) x _k
         */
        mat _FSC;

        /**
         * Signal Noise Ratio
         * each column stands for a SNR of a certain reference
         * (_r * _pf) x _k
         */
        mat _SNR;

        /**
         * tau^2
         * each column stands for the power spectrum of a certain reference
         * (_r * _pf) x _k
         */
        mat _tau;

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
         * size of references before padding
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
         * variance of 2D Gaussian distribution of the translation in X
         */
        double _tVariS0;

        /**
         * variance of 2D Gaussian distribution of the translation in Y
         */
        double _tVariS1;

        /**
         * a parameter indicating the change of rotation between iterations
         */
        double _rChange = 1;

        /**
         * a parameter indicating the change of rotation between iterations of
         * the previous
         */
        double _rChangePrev = 1;

        int _nRChangeNoDecrease = 0;

        /**
         * the symmetry information
         */
        const Symmetry* _sym = NULL;

        /**
         * the suggest search type
         */
        int _searchType = 0;

    public:

        /**
         * default constructor
         */
        MLModel();

        /**
         * default deconstructor
         */
        ~MLModel();

        /**
         * This function initialises the MLModel object.
         *
         * @param number of references
         * @param size size of references before padding
         * @param r radius of calculating FSC and SNR before padding
         * @param pf padding factor
         * @param pixelSize pixel size of 2D images (in Angstrom)
         * @param a width of modified Kaiser-Bessel function
         * @param alpha smoothness parameter of modified Kaiser-Bessel function
         * @param sym the symmetry information
         */
        void init(const int k,
                  const int size,
                  const int r,
                  const int pf,
                  const double pixelSize,
                  const double a,
                  const double alpha,
                  const Symmetry* sym);

        /**
         * This function initialises projectors and reconstructors.
         */
        void initProjReco();

        /**
         * This function returns a reference of the i-th reference.
         *
         * @param i index of the reference
         */
        Volume& ref(const int i);

        /**
         * This function appends a reference to the vector of references.
         * 
         * @param ref the reference to be appended
         */
        void appendRef(Volume ref);

        /**
         * This function returns the number of references.
         */
        int k() const;

        /**
         * This function returns the size of references before padding.
         */
        int size() const;

        /**
         * This function returns the maximum possible value of _r.
         */
        int maxR() const;

        /**
         * This function returns the radius of calculating FSC and SNR before
         * padding.
         */
        int r() const;

        /**
         * This function sets the radius of calculating FSC and SNR before
         * padding.
         *
         * @param r the radius of calculating FSC and SNR before padding
         */
        void setR(const int r);

        /**
         * This function returns a reference to the projector of the i-th
         * reference.
         *
         * @param i the index of the reference
         */
        Projector& proj(const int i = 0);

        /**
         * This function returns a reference to the reconstructor of the i-th
         * reference.
         *
         * @param i the index of the reference
         */
        Reconstructor& reco(const int i = 0);

        /**
         * This function performs the following procedure. The MASTER process
         * fetchs references both from A hemisphere and B hemisphere. It
         * compares the references from two hemisphere respectively for FSC. It
         * broadcast the FSC to all process.
         */
        void BcastFSC();

        /**
         * This function performs a low pass filter on each reference.
         * 
         * @param thres threshold of spatial frequency of low pass filter
         * @param ew edge width of spatial frequency of low pass filter
         */
        void lowPassRef(const double thres,
                        const double ew);

        /**
         * This function calculates SNR from FSC.
         */
        void refreshSNR();

        /**
         * This function calculates tau^2 (power spectrum) of each references.
         */
        void refreshTau();

        /**
         * This function returns the tau^2 (power spectrum) of the i-th
         * reference.
         *
         * @param i the index of the refefence
         */
        vec tau(const int i) const;

        /**
         * This function returns the resolution in pixel of the i-th
         * reference.
         *
         * @param i the index of the reference
         */
        int resolutionP(const int i) const;

        /**
         * This function returns the highest resolution in pixel of the
         * references.
         */
        int resolutionP() const;

        /**
         * This function returns the resolution in Angstrom(-1) of the i-th
         * reference.
         *
         * @param i the index of the reference
         */
        double resolutionA(const int i) const;

        /**
         * This function returns the highest resolution in Angstrom(-1) of the
         * references.
         */
        double resolutionA() const;

        /**
         * This function sets the max radius of all projector to a certain
         * value.
         *
         * @param maxRadius max radius
         */
        void setProjMaxRadius(const int maxRadius);

        /**
         * This function refreshs the projectors by resetting the projectee, the
         * frequency threshold and padding factor, respectively.
         */
        void refreshProj();

        /**
         * This function refreshs the reconstructors by resetting the size,
         * padding factor, symmetry information, MKB kernel parameters,
         * respectively.
         */
        void refreshReco();

        /** 
         * This function increases _r according to wether FSC is high than 0.2
         * at current _r.
         */
        void updateR();

        /**
         * This function returns the concentration parameter of the rotation.
         */
        double rVari() const;

        /** 
         * This function returns the variance of 2D Gaussian distribution of the
         * translation in X.
         */
        double tVariS0() const;
        
        /** 
         * This function returns the variance of 2D Gaussian distribution of the
         * translation in Y.
         */
        double tVariS1() const;

        /**
         * This function calculates the variance paramters and averages those in
         * the same hemisphere. The variance paramters include the concentration
         * parameter of the rotation, the variances of 2D Guassian distribution
         * of the translation in X and Y.
         *
         * @param par a vector of Particle
         * @param n number of images in the hemisphere
         */
        void allReduceVari(const vector<Particle>& par,
                           const int n);

        /**
         * This function returns the average rotation change between iterations.
         */
        double rChange() const;

        /**
         * This function calculates the difference of the rotation between
         * iterations and averages those in the same hemisphere.
         *
         * @param par a vector of Particle
         * @param n number of images in the hemisphere
         */
        void allReduceRChange(vector<Particle>& par,
                              const int n);

        /**
         * This function returns the suggested search type.
         */
        int searchType();

        void clear();
};

#endif // ML_MODEL_H

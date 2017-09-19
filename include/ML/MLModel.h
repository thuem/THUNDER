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

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"

#include "Image.h"
#include "Volume.h"
#include "ImageFile.h"
#include "Parallel.h"
#include "Filter.h"
#include "Spectrum.h"
#include "Projector.h"
#include "Symmetry.h"
#include "Reconstructor.h"
#include "Particle.h"

#include <boost/container/vector.hpp>
#include <boost/move/make_unique.hpp>

#define FOR_EACH_CLASS \
    for (int l = 0; l < _k; l++)

#define SEARCH_RES_GAP_GLOBAL 10

#define SEARCH_TYPE_STOP -1

#define SEARCH_TYPE_GLOBAL 0

#define SEARCH_TYPE_LOCAL 1

#define SEARCH_TYPE_CTF 2

/**
#define MAX_ITER_R_CHANGE_NO_DECREASE_GLOBAL 0

#define MAX_ITER_R_CHANGE_NO_DECREASE_LOCAL 0
***/

#define MAX_ITER_R_CHANGE_NO_DECREASE_GLOBAL 2

#define MAX_ITER_R_CHANGE_NO_DECREASE_LOCAL 1

#define MAX_ITER_RES_NO_IMPROVE 1

#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE

#define R_CHANGE_DECREASE_GLOBAL 0.5

#define R_CHANGE_DECREASE_STUN 0.5

#define R_CHANGE_DECREASE_LOCAL 0.5

#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI

#define T_VARI_DECREASE_GLOBAL 0.05

#define T_VARI_DECREASE_STUN 0.05

#define T_VARI_DECREASE_LOCAL 0.05

#endif

#define INTERP_TYPE_GLOBAL LINEAR_INTERP
//#define INTERP_TYPE_GLOBAL NEAREST_INTERP

#define INTERP_TYPE_LOCAL LINEAR_INTERP

/**
 * resolution resolution for averaging reference(s) from A hemisphere and B
 * hemisphere
 */
#define A_B_AVERAGE_THRES 10

#define CUTOFF_BEYOND_RES 0

class MLModel : public Parallel
{
    private:

        int _mode;

        /**
         * perform global search or not
         */
        bool _gSearch;

        /**
         * perform local search or not
         */
        bool _lSearch;

        /**
         * perform CTF search or not
         */
        bool _cSearch;

        bool _coreFSC;

        bool _maskFSC;

        bool _goldenStandard;

        const Volume* _mask;

        int _coreR;

        /**
         * references in Fourier space
         */
        boost::container::vector<Volume> _ref;

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
         * (_rU * _pf - 1) x _k
         */
        mat _tau;

        /**
         * sig^2
         * average power spectrum of noise
         */
        vec _sig;

        /**
         * projectors
         */
        boost::container::vector<Projector> _proj;

        /**
         * reconstructors
         */
        boost::container::vector< boost::movelib::unique_ptr<Reconstructor> > _reco;

        /**
         * number of references
         */
        int _k;

        /**
         * size of references before padding
         */
        int _size;

        /**
         * frequency before padding (in pixel)
         */
        int _r;

        /**
         * frequency for reconstruction and calculating FSC, SNR 
         */
        int _rU;

        /**
         * frequency of the previous iteration
         */
        int _rPrev;

        int _rUPrev;

        /**
         * the top frequency ever reached
         */
        int _rT;

        /*
         * resolution before padding (in pixel)
         */
        int _res;

        /**
         * the top resolution ever reached
         */
        int _resT;

        /**
         * padding factor
         */
        int _pf;

        /**
         * pixel size of 2D images (in Angstrom)
         */
        double _pixelSize;

        /**
         * upper boundary of frequency during global search before padding (in
         * pixel)
         */
        int _rGlobal;

        /**
         * width of modified Kaiser-Bessel function
         */
        double _a;

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

        double _tVariS0Prev;

        double _tVariS1Prev;

        double _stdRVari;

        double _stdTVariS0;

        double _stdTVariS1;

        /**
         * a parameter indicating the change of rotation between iterations
         */
        double _rChange;

        /**
         * a parameter indicating the change of rotation between iterations of
         * the previous
         */
        double _rChangePrev;

        /**
         * a parameter indicating the standard deviation of rotation between
         * iterations
         */
        double _stdRChange;

        /**
         * a parameter indicating the standard deviation of rotation between
         * iteration of the previous
         */
        double _stdRChangePrev;

        /**
         * number of iterations without decreasing in rotation change
         */
        int _nRChangeNoDecrease;

        /**
         * number of iterations without top resolution improvement
         */
        int _nTopResNoImprove;

        /**
         * the symmetry information
         */
        const Symmetry* _sym;

        /**
         * the suggest search type
         */
        int _searchType;

        int _searchTypePrev;

        /**
         * whether the frequency should be increased or not
         */
        bool _increaseR;

    public:

        /**
         * default constructor
         */
        MLModel()
        {
            _mode = MODE_3D;
            _gSearch = true;
            _lSearch = true;
            _cSearch = true;
            _coreFSC = false;
            _maskFSC = false;
            _goldenStandard = false;
            _coreR = 0;
            _r = 1;
            _rU = 1;
            _rPrev = 1;
            _rUPrev = 1;
            _rT = 1;
            _res = 1;
            _resT = 1;
            _pf = 2;
            _a = 1.9;
            _rVari = 0;
            _tVariS0Prev = DBL_MAX;
            _tVariS0Prev = DBL_MAX;
            _tVariS0 = DBL_MAX;
            _tVariS1 = DBL_MAX;
            _stdRVari = 0;
            _stdTVariS0 = 0;
            _stdTVariS1 = 0;
            _rChange = DBL_MAX;
            _rChangePrev = DBL_MAX;
            _stdRChange = 0;
            _stdRChangePrev = 0;
            _nRChangeNoDecrease = 0;
            _nTopResNoImprove = 0;
            _sym = NULL;
            _searchType = SEARCH_TYPE_GLOBAL;
            _increaseR = false;
        }

        /**
         * default deconstructor
         */
        ~MLModel();

        /**
         * This function initialises the MLModel object.
         *
         * @param mode      2D or 3D mode
         * @param k         number of references
         * @param size      size of references before padding
         * @param r         radius of calculating FSC and SNR before padding
         * @param pf        padding factor
         * @param pixelSize pixel size of 2D images (in Angstrom)
         * @param a         width of modified Kaiser-Bessel function
         * @param alpha     smoothness parameter of modified Kaiser-Bessel function
         * @param sym       the symmetry information
         */
        void init(const int mode,
                  const bool gSearch,
                  const bool lSearch,
                  const bool cSearch,
                  const bool coreFSC,
                  const int coreR,
                  const bool maskFSC,
                  const Volume* mask,
                  const bool goldenStandard,
                  const int k,
                  const int size,
                  const int r,
                  const int pf,
                  const double pixelSize,
                  const double a,
                  const double alpha,
                  const Symmetry* sym);

        int mode() const;

        void setMode(const int mode);

        bool gSearch() const;

        void setGSearch(const bool gSearch);

        bool lSearch() const;

        void setLSearch(const bool lSearch);

        /***
        bool refine() const;

        void setRefine(const bool refine);
        ***/

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

        void clearRef();

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
         * This function returns the frequency before padding (in pixel).
         */
        int r() const;

        /**
         * This function sets the frequency before padding (in pixel).
         *
         * @param r the frequency before padding (in pixel)
         */
        void setR(const int r);

        /**
         * This function returns the frequency for reconstruction and
         * calculating FSC, SNR.
         */
        int rU() const;

        void setRU(const int rU);

        void setMaxRU();

        /**
         * This function returns the frequency before padding (in pixel) of the
         * previous iteration.
         */
        int rPrev() const;

        int rUPrev() const;

        /**
         * This function returns the highest frequency ever reached.
         */
        int rT() const;

        /**
         * This function sets the highest frequency ever reached.
         *
         * @param rT the highest frequency ever reached
         */
        void setRT(const int rT);

        int res() const;

        void setRes(const int res);

        int resT() const;

        void setResT(const int resT);

        /**
         * This function returns the upper boundary frequency during global
         * search before padding (in pixel).
         */
        int rGlobal() const;

        /**
         * This function sets the upper boundary frequency during global
         * search before padding (in pixel).
         *
         * @param rGlobal the upper boundary frequency during global search stage
         */
        void setRGlobal(const int rGlobal);

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
        void BcastFSC(const double thres);

        /**
         * This function performs a low pass filter on each reference.
         * 
         * @param thres threshold of spatial frequency of low pass filter
         * @param ew    edge width of spatial frequency of low pass filter
         */
        void lowPassRef(const double thres,
                        const double ew);

        /**
         * This function returns the FSCs as each column stands for the FSC of a
         * reference.
         */
        mat fsc() const;

        /**
         * This function returns the SNRs as each column stands for the SNR of a
         * reference.
         */
        mat snr() const;

        /**
         * This function returns the FSC of the i-th reference.
         *
         * @param i the index of the reference
         */
        vec fsc(const int i) const;

        /**
         * This function returns the SNR of the i-th reference.
         *
         * @param i the index of the reference
         */
        vec snr(const int i) const;

        /**
         * This function calculates SNR from FSC.
         */
        void refreshSNR();

        /**
         * This function calculates tau^2 (power spectrum) of each references.
         */
        void refreshTau();

        void refreshSig(const vec& sig);

        /***
        void resetTau();

        void resetTau(const vec tau);
        ***/

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
         * @param i       the index of the reference
         * @param thres   the threshold for determining resolution
         * @param inverse whether to search from high frequency to low frequency
         *                or not
         */
        int resolutionP(const int i,
                        const double thres = 0.143,
                        const bool inverse = false) const;

        /**
         * This function returns the highest resolution in pixel of the
         * references.
         *
         * @param thres the threshold for determining resolution
         * @param inverse whether to search from high frequency to low frequency
         *                or not
         */
        int resolutionP(const double thres = 0.143,
                        const bool inverse = false) const;

        /**
         * This function returns the resolution in Angstrom(-1) of the i-th
         * reference.
         *
         * @param i the index of the reference
         * @param thres the threshold for determining resolution
         */
        double resolutionA(const int i,
                           const double thres = 0.143) const;

        /**
         * This function returns the highest resolution in Angstrom(-1) of the
         * references.
         */
        double resolutionA(const double thres = 0.143) const;

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

        void resetReco();

        /***
        void refreshRecoSigTau(const int rSig,
                               const int rTau);
                               ***/

        /** 
         * This function increases _r according to wether FSC is high than 0.2
         * at current _r.
         *
         * @param thres the threshold for determining resolution
         */
        void updateR(const double thres = 0.143);

        void elevateR(const double thres = 0.143);

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

        double stdRVari() const;

        double stdTVariS0() const;

        double stdTVariS1() const;

        double tVariS0Prev() const;

        double tVariS1Prev() const;

        void setRVari(const double rVari);

        void setTVariS0(const double tVariS0);

        void setTVariS1(const double tVariS1);

        void resetTVari();

        void setStdRVari(const double stdRVari);

        void setStdTVariS0(const double stdTVariS0);

        void setStdTVariS1(const double stdTVariS1);

        /**
         * This function returns the average rotation change between iterations.
         */
        double rChange() const;

        /**
         * This function returns the average rotation change between the
         * previous two iterations.
         */
        double rChangePrev() const;

        /**
         * This function returns the standard deviation of the rotation change
         * between iterations.
         */
        double stdRChange() const;

        /**
         * This function sets the mean value of rotation change. This function
         * will automatically save the previous rotation change to another
         * attribute.
         *
         * @param rChange mean value of rotation change
         */
        void setRChange(const double rChange);

        /**
         * This function resets the mean value of rotation change and the
         * previous rotation change to 1.
         */
        void resetRChange();

        /**
         * This function sets the standard deviation of rotation change. This
         * function will automatically save the previous standard devation of
         * rotation change to another attribute.
         *
         * @param stdRChange standard devation of rotation change
         */
        void setStdRChange(const double stdRChange);

        /**
         * This function returns the number of iterations that rotation change
         * between iterations does not decrease.
         */
        int nRChangeNoDecrease() const;

        /**
         * This function sets the number of iterations that rotation change
         * between iterations does not decrease.
         *
         * @param nRChangeNoDecrease the number of iterations that rotation
         *                           change between iterations does not decrease
         */
        void setNRChangeNoDecrease(const int nRChangeNoDecrease);

        /**
         * This function returns the number of iterations that the resolution
         * does not elevate.
         */
        int nTopResNoImprove() const;

        /**
         * This function set the number of iterations that the resolution does
         * not elevate.
         *
         * @param nTopResNoImprove the number of iterations that the resolution
         *                         does not elevate
         */
        void setNTopResNoImprove(const int nTopResNoImprove);

        /**
         * This function returns the suggested search type.
         */
        int searchType();

        void setSearchType(const int searchType);

        int searchTypePrev() const;

        void setSearchTypePrev(const int searchTypePrev);

        /**
         * This function returns whether to increase cutoff frequency or not.
         */
        bool increaseR() const;

        /**
         * This function sets whether to increase cutoff frequency or not.
         *
         * @param increaseR increase cutoff increase or not
         */
        void setIncreaseR(const bool increaseR);

        /**
         * This function update the frequency for reconstruction and calculating
         * FSC, SNR by the frequency before padding (in pixel).
         */
        void updateRU();

        /**
         * This function clears up references, projectors and reconstructors.
         */
        void clear();

    private:

        /**
         * This function checks whether rotation change has room for decreasing
         * or not at current frequency. If there is still room, return false,
         * otherwise, return true.
         */

#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE
        bool determineIncreaseR(const double rChangeDecreaseFactor);
#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI
        bool determineIncreaseR(const double tVariDecreaseFactor);
#endif

        void avgHemi();
};

#endif // ML_MODEL_H

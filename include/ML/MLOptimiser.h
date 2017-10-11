/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ML_OPTIMISER_H
#define ML_OPTIMISER_H

#include <cstdlib>
#include <sstream>
#include <string>
#include <climits>
#include <queue>
#include <functional>

#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>

#include <omp_compat.h>

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"

#include "Image.h"
#include "Volume.h"
#include "ImageFile.h"
#include "Spectrum.h"
#include "Symmetry.h"
#include "CTF.h"
#include "Mask.h"
#include "Particle.h"
#include "Database.h"
#include "MLModel.h"

#define FOR_EACH_2D_IMAGE for (ptrdiff_t l = 0; l < static_cast<ptrdiff_t>(_ID.size()); l++)

#define ALPHA_GLOBAL_SEARCH 1.0
#define ALPHA_LOCAL_SEARCH 0

#define MIN_N_PHASE_PER_ITER_GLOBAL 10
#define MIN_N_PHASE_PER_ITER_LOCAL 3
#define MAX_N_PHASE_PER_ITER 100

#define PARTICLE_FILTER_DECREASE_FACTOR 0.95

#define N_PHASE_WITH_NO_VARI_DECREASE 1

#define N_SAVE_IMG 20 

#define TRANS_Q 0.01

#define MIN_STD_FACTOR 3

struct MLOptimiserPara
{
    /**
     * maximum number of threads in a process
     */
    int nThreadsPerProcess;

    /**
     * 2D or 3D mode
     */
    int mode;

    /**
     * perform global search or not
     */
    bool gSearch;

    /**
     * perform local search or not
     */
    bool lSearch;

    /**
     * perform ctf search or not
     */
    bool cSearch;

    /**
     * number of classes
     */
    int k;

    /**
     * size of image (pixel)
     */
    int size;

    /**
     * pixel size (Angstrom)
     */
    double pixelSize;

    /**
     * radius of mask on images (Angstrom)
     */
    double maskRadius;

    /**
     * estimated translation (pixel)
     */
    double transS;

    /**
     * initial resolution (Angstrom)
     */
    double initRes;

    /**
     * scan resolution (Angstrom)
     */
    double scanRes;

    /**
     * resolution threshold for performing global search
     */
    double globalSearchRes;

    /**
     * symmetry
     */
    char sym[SYM_ID_LENGTH];

    /**
     * initial model
     */
    char initModel[FILE_NAME_LENGTH];

    /**
     * sqlite3 file storing paths and CTFs of images
     */
    char db[FILE_NAME_LENGTH];

    char parPrefix[FILE_NAME_LENGTH];

    char dstPrefix[FILE_NAME_LENGTH];

    bool coreFSC;

    bool maskFSC;

    /**
     * whether to perform masking on the reference
     */
    bool performMask;

    /**
     * whether to automatically generate a mask
     */
    bool autoMask;

    /**
     * mask
     */
    char mask[FILE_NAME_LENGTH];

    /**
     * max number of iteration
     */
    int iterMax;

    bool goldenStandard;
    
    /**
     * padding factor
     */
    int pf;
    
    /**
     * MKB kernel radius
     */
    double a;

    /**
     * MKB kernel smooth factor
     */
    double alpha;

    /**
     * number of sampling points for scanning in global search
     */
    int mS;

    /**
     * number of sampling points in global search
     */
    int mGMax;

    int mGMin;

    int mLR;

    /**
     * number of sampling points in local search
     */
    int mLT;

    int mLD;

    /**
     * number of sampling points used in reconstruction
     */
    int mReco;

    /**
     * the information below this resolution will be ignored
     */
    double ignoreRes;

    /**
     * the resolution boundary for performing intensity scale correction
     */
    double sclCorRes;

    /**
     * the FSC threshold for determining cutoff frequency
     */
    double thresCutoffFSC;

    /**
     * the FSC threshold for reporting resolution
     */
    double thresReportFSC;

    double thresSclCorFSC;

    /**
     * grouping or not when calculating sigma
     */
    bool groupSig;

    /**
     * grouping or not when calculating intensity scale
     */
    bool groupScl;

    /**
     * mask the 2D images with zero background or gaussian noise
     */
    bool zeroMask;

    bool parGra;

    double transSearchFactor;

    double perturbFactorL;

    double perturbFactorSGlobal;

    double perturbFactorSLocal;

    double perturbFactorSCTF;

    double ctfRefineS;

    /**
     * whether skip expectation or not
     */
    bool skipE;

    /**
     * whether skip maximization or not
     */
    bool skipM;

    /**
     * whether skip reconstruction or not
     */
    bool skipR;

    MLOptimiserPara()
    {
        nThreadsPerProcess = 1;
        mode = MODE_3D;
        gSearch = true;
        lSearch = true;
        cSearch = true;
        coreFSC = false;
        maskFSC = false;
        performMask = false;
        autoMask = false;
        goldenStandard = false;
        pf = 2;
        a = 1.9;
        alpha = 15;
        thresCutoffFSC = 0.143;
        thresReportFSC = 0.143;
        thresSclCorFSC = 0.75;
        transSearchFactor = 1;
        perturbFactorL = 0.8;
        perturbFactorSGlobal = 0.8;
        perturbFactorSLocal = 0.8;
        perturbFactorSCTF = 0.8;
        ctfRefineS = 0.01;
        skipE = false;
        skipM = false;
        skipR = false;
    }
};

void display(const MLOptimiserPara& para);

class MLOptimiser : public Parallel
{
    private:

        MLOptimiserPara _para;

        /**
         * total number of 2D images
         */
        int _nPar;

        /**
         * total number of 2D images in each hemisphere
         */
        int _N;

        /**
         * cutoff frequency (in pixels)
         */
        int _r;

        /**
         * the information below this frequency will be ignored during
         * comparison
         */
        double _rL;

        /**
         * the information below this frequency will be used for performing
         * intensity scale correction
         */
        int _rS;

        /**
         * current number of iterations
         */
        int _iter;

        /**
         * current cutoff resolution (Angstrom)
         */
        double _resCutoff;

        /**
         * current report resolution (Angstrom)
         */
        double _resReport;

        /**
         * current search type
         */
        int _searchType;

        /**
         * model containting references, projectors, reconstruuctors, information 
         * about FSC, SNR and determining the cutoff frequency and search type
         */
        MLModel _model;

        /**
         * a database containing information of 2D images, CTFs, group and
         * micrograph information
         */
        Database _db;

        /**
         * the symmetry information of this reconstruction
         */
        Symmetry _sym; 

        /**
         * a unique ID for each 2D image
         */
        vector<int> _ID;

        /**
         * 2D images
         */
        vector<Image> _img;

        /**
         * unmasked 2D images
         */
        vector<Image> _imgOri;

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
        /**
         * the offset between images and original images
         * an original image will become the corresponding image by this
         * translation
         */
        vector<vec2> _offset;
#endif

        /**
         * a particle filter for each 2D image
         */
        vector<Particle> _par;

        vector<CTFAttr> _ctfAttr;

        /**
         * a CTF for each 2D image
         */
        vector<Image> _ctf;

        vector<int> _nP;

        /**
         * Each row stands for sigma^2 of a certain group, thus the size of this
         * matrix is _nGroup x (maxR() + 1)
         */
        mat _sig;

        /**
         * Each row stands for -0.5 / sigma^2 of a certain group, thus the size
         * of this matrix is _nGroup x maxR()
         */
        mat _sigRcp;

        /**
         * intensity scale of a certain group
         */
        vec _scale;

        /**
         * number of groups
         */
        int _nGroup;

        /**
         * a unique ID for each group
         */
        vector<int> _groupID;

        double _mean;

        /*
         * standard deviation of noise
         */
        double _stdN;

        /*
         * standard deviation of data
         */
        double _stdD;

        /*
         * standard deviation of signal
         */
        double _stdS;

        /*
         * standard deviation of standard deviation of noise
         */
        double _stdStdN;

        /**
         * images distribution over classes
         */
        vec _cDistr;

        /**
         * whether to generate mask or not
         */
        bool _genMask;

        /**
         * mask
         */
        Volume _mask;

        /**
         * number of performed filtering in an iteration of a process
         */
        int _nF;

        /**
         * number of performed images in an iteration of a process
         */
        int _nI;

        /**
         * number of performed rotations in the scanning phase of the global
         * search stage
         */
        int _nR;

        int _nPxl;

        int* _iPxl;

        int* _iCol;
        
        int* _iRow;

        int* _iSig;

        Complex* _datP;

        double* _ctfP;

        double* _sigRcpP;

        /**
         * spatial frequency of each pixel
         */
        double* _frequency;

        /**
         * defocus of each pixel of each image
         */
        double* _defocusP;

        /**
         * K1 of CTF of each image
         */
        double* _K1;

        /**
         * K2 of CTF of each image
         */
        double* _K2;

        FFT _fftImg;

    public:
        
        MLOptimiser()
        {
            _stdN = 0;
            _stdD = 0;
            _stdS = 0;
            _stdStdN = 0;
            _genMask = false;
            _nF = 0;
            _nI = 0;
            _nR = 0;

            _searchType = SEARCH_TYPE_GLOBAL;

            _nPxl = 0;
            _iPxl = NULL;
            _iCol = NULL;
            _iRow = NULL;
            _iSig = NULL;

            _datP = NULL;
            _ctfP = NULL;
            _sigRcpP = NULL;
        }

        ~MLOptimiser();

        MLOptimiserPara& para();

        void setPara(const MLOptimiserPara& para);

        void init();

        void expectation();

        void maximization();

        void run();

        void clear();

    private:

        /**
         * broadcast the number of images in each hemisphere
         */
        void bCastNPar();

        /**
         * allreduce the total number of images
         */
        void allReduceN();

        /**
         * the size of the image
         */
        int size() const;
        
        /**
         * maximum frequency (pixel)
         */
        int maxR() const;

        /**
         * broadcast the group information
         */
        void bcastGroupInfo();

        /**
         * initialise the reference
         */
        void initRef();

        /**
         * read mask
         */
        void initMask();

        /**
         * initialise the ID of each image
         */
        void initID();

        /*
         * read 2D images from hard disk and perform a series of processing
         */
        void initImg();

        /**
         * do statistics on the signal and noise of the images
         */
        void statImg();

        /**
         * display the statistics result of the signal and noise of the images
         */
        void displayStatImg();

        /**
         * substract the mean of background from the images, make the noise of
         * the images has zero mean
         */
        void substractBgImg();

        /**
         * mask the images
         */
        void maskImg();

        /**
         * normlise the images, make the noise of the images has a standard
         * deviation equals to 1
         */
        void normaliseImg();

        /**
         * perform Fourier transform on images
         */
        void fwImg();

        /**
         * perform inverse Fourier transform on images
         */
        void bwImg();

        /**
         * initialise CTFs
         */
        void initCTF();

        /**
         * correct the intensity scale
         *
         * @param init  whether it is an initial correction or not
         * @param group grouping or not
         */
        void correctScale(const bool init = false,
                          const bool coord = false,
                          const bool group = true);

        /**
         * initialise sigma
         */
        void initSigma();

        /**
         * initialise particle filters
         */
        void initParticles();

        void avgStdR(double& stdR);

        void avgStdT(double& stdT);

        void loadParticles();

        /**
         * re-calculate the rotation change between this iteration and the
         * previous one
         */
        void refreshRotationChange();

        void refreshClassDistr();

        void balanceClass(const double thres = 0.2);

        /**
         * re-calculate the rotation and translation variance
         */
        void refreshVariance();

        /**
         * re-calculate the intensity scale
         *
         * @param init  whether using given coordiantes or not
         * @param group grouping or not
         */
        void refreshScale(const bool coord = false,
                          const bool group = true);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
        /**
         * re-centre images according to translation expectation of the last
         * ieration; mask if neccessary
         */
        void reCentreImg();
#endif

        void reMaskImg();

        void normCorrection();

        /**
         * re-calculate sigma
         *
         * @param group grouping or not
         */
        void allReduceSigma(const bool group = true);

        /**
         * reconstruct reference
         */
        void reconstructRef();

        /***
         * @param mask           whether mask on the reference is allowed or
         *                       not
         * @param solventFlatten whether solvent flatten on the reference is
         *                       allowed or not when mask is off
         */
        void solventFlatten(const bool mask = true);

        void allocPreCalIdx(const double rU,
                            const double rL);

        void allocPreCal(const bool pixelMajor,
                         const bool ctf);

        void freePreCalIdx();

        void freePreCal(const bool ctf);

        void saveDatabase() const;

        /**
         * for debug, save the best projections
         */
        void saveBestProjections();

        /**
         * for debug, save the images
         */
        void saveImages();

        /**
         * for debug, save the CTFs
         */
        void saveCTFs();

        /**
         * save the reference(s)
         *
         * @param finished whether it is the final round or not
         */
        void saveReference(const bool finished = false);

        /**
         * save FSC
         */
        void saveFSC(const bool finished = false) const;

        void saveSig() const;

        void saveTau() const;
};

/***
int searchPlace(double* topW,
                const double w,
                const int l,
                const int r);

void recordTopK(double* topW,
                unsigned int* iTopR,
                unsigned int* iTopT,
                const double w,
                const unsigned int iR,
                const unsigned int iT,
                const int k);
                ***/

/**
 * This function calculates the logarithm of the possibility that the image is
 * from the projection.
 *
 * @param dat image
 * @param pri projection
 * @param ctf CTF
 * @param sig sigma of noise
 * @param rU  the upper boundary of frequency of the signal for comparison
 * @param rL  the lower boundary of frequency of the signal for comparison
 */
double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sigRcp,
                      const double rU,
                      const double rL);

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sigRcp,
                      const int* iPxl,
                      const int* iSig,
                      const int m);

double logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const double* ctf,
                      const double* sigRcp,
                      const int m);

double logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const double* frequency,
                      const double* defocus,
                      const double df,
                      const double K1,
                      const double K2,
                      const double* sigRcp,
                      const int m);

/**
 * This function calculates the logarithm of the possibility that the image is
 * from the projection translation couple.
 *
 * @param dat image
 * @param pri projection
 * @param tra translation
 * @param ctf CTF
 * @param sig sigma of noise
 * @param rU  the upper boundary of frequency of the signal for comparison
 * @param rL  the lower boundary of frequency of the signal for comparison
 */
double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sigRcp,
                      const double rU,
                      const double rL);

/**
 * This function calculates th logarithm of possibility that images is from the
 * projection translation couple. The pixels needed for calculation are assigned
 * by an array.
 *
 * @param dat  image
 * @param pri  projection
 * @param tra  translation
 * @param ctf  CTF
 * @param sig  sigma of noise
 * @param iPxl the indices of the pixels
 * @param iSig the indices of the sigma of the corresponding pixels
 */
double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sigRcp,
                      const int* iPxl,
                      const int* iSig,
                      const int m);

/**
 * This function calculates the logarithm of the possibilities of a series of
 * images is from a certain projection.
 *
 * @param dat     a series of images
 * @param pri     a certain projection
 * @param ctf     a series of CTFs corresponding to the images
 * @param groupID the group the corresponding image belongs to
 * @param sig     sigma of noise
 * @param rU  the upper boundary of frequency of the signal for comparison
 * @param rL  the lower boundary of frequency of the signal for comparison
 */
vec logDataVSPrior(const vector<Image>& dat,
                   const Image& pri,
                   const vector<Image>& ctf,
                   const vector<int>& groupID,
                   const mat& sigRcp,
                   const double rU,
                   const double rL);

vec logDataVSPrior(const vector<Image>& dat,
                   const Image& pri,
                   const vector<Image>& ctf,
                   const vector<int>& groupID,
                   const mat& sigRcp,
                   const int* iPxl,
                   const int* iSig,
                   const int m);

/***
vec logDataVSPrior(const Complex* const* dat,
                   const Complex* pri,
                   const double* const* ctf,
                   const double* const* sigRcp,
                   const int n,
                   const int m);
***/

/**
 * This function calculates the logarithm of the possibilities of a series of
 * images is from a certain projection. The series of images have been packed in
 * a continous allocated memory. Besides, the corresponding CTF value and
 * reciprocal of sigma of noise of each pixel have also been packed in a 
 * continous allocated memory.
 *
 * @param dat    a series of images
 * @param pri    a certain projection
 * @param ctf    CTF values of each pixel correspondingly
 * @param sigRcp the reciprocal of sigma of noise of each pixel correspondingly
 * @param n      the number of images
 * @param m      the number of pixels in each image
 */
vec logDataVSPrior(const Complex* dat,
                   const Complex* pri,
                   const double* ctf,
                   const double* sigRcp,
                   const int n,
                   const int m);

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sigRcp,
                   const double rU,
                   const double rL);

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& tra,
                   const Image& ctf,
                   const vec& sigRcp,
                   const double rU,
                   const double rL);

void scaleDataVSPrior(vec& sXA,
                      vec& sAA,
                      const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const double rU,
                      const double rL);

#endif // ML_OPTIMSER_H

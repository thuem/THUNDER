//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef OPTIMISER_H
#define OPTIMISER_H

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
#include "Model.h"

#define FOR_EACH_2D_IMAGE for (ptrdiff_t l = 0; l < static_cast<ptrdiff_t>(_ID.size()); l++)

#define MIN_M_S 1500

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

#define CLASS_BALANCE_FACTOR 0.05

struct OptimiserPara
{

#define KEY_N_THREADS_PER_PROCESS "Number of Threads Per Process"

    /**
     * maximum number of threads in a process
     */
    int nThreadsPerProcess;

#define KEY_MODE "2D or 3D Mode"

    /**
     * 2D or 3D mode
     */
    int mode;

#define KEY_G_SEARCH "Global Search"

    /**
     * perform global search or not
     */
    bool gSearch;

#define KEY_L_SEARCH "Local Search"

    /**
     * perform local search or not
     */
    bool lSearch;

#define KEY_C_SEARCH "CTF Search"

    /**
     * perform ctf search or not
     */
    bool cSearch;

#define KEY_K "Number of Classes"

    /**
     * number of classes
     */
    int k;

#define KEY_SIZE "Size of Image"

    /**
     * size of image (pixel)
     */
    int size;

    /**
     * pixel size (Angstrom)
     */
    RFLOAT pixelSize;

    /**
     * radius of mask on images (Angstrom)
     */
    RFLOAT maskRadius;

    /**
     * estimated translation (pixel)
     */
    RFLOAT transS;

    /**
     * initial resolution (Angstrom)
     */
    RFLOAT initRes;

    /**
     * scan resolution (Angstrom)
     */
    RFLOAT scanRes;

    /**
     * resolution threshold for performing global search
     */
    RFLOAT globalSearchRes;

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
     * whether to perform masking during global search
     */
    bool globalMask;

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
    RFLOAT a;

    /**
     * MKB kernel smooth factor
     */
    RFLOAT alpha;

    /**
     * number of sampling points for scanning in global search
     */
    int mS;

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
    RFLOAT ignoreRes;

    /**
     * the resolution boundary for performing intensity scale correction
     */
    RFLOAT sclCorRes;

    /**
     * the FSC threshold for determining cutoff frequency
     */
    RFLOAT thresCutoffFSC;

    /**
     * the FSC threshold for reporting resolution
     */
    RFLOAT thresReportFSC;

    RFLOAT thresSclCorFSC;

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

    RFLOAT transSearchFactor;

    RFLOAT perturbFactorL;

    RFLOAT perturbFactorSGlobal;

    RFLOAT perturbFactorSLocal;

    RFLOAT perturbFactorSCTF;

    RFLOAT ctfRefineS;

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

    OptimiserPara()
    {
        nThreadsPerProcess = 1;
        mode = MODE_3D;
        gSearch = true;
        lSearch = true;
        cSearch = true;
        coreFSC = false;
        maskFSC = false;
        performMask = false;
        globalMask = false;
        autoMask = false;
        goldenStandard = true;
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

void display(const OptimiserPara& para);

class Optimiser : public Parallel
{
    private:

        OptimiserPara _para;

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
        RFLOAT _rL;

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
        RFLOAT _resCutoff;

        /**
         * current report resolution (Angstrom)
         */
        RFLOAT _resReport;

        /**
         * current search type
         */
        int _searchType;

        /**
         * model containting references, projectors, reconstruuctors, information 
         * about FSC, SNR and determining the cutoff frequency and search type
         */
        Model _model;

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

        RFLOAT _mean;

        /*
         * standard deviation of noise
         */
        RFLOAT _stdN;

        /*
         * standard deviation of data
         */
        RFLOAT _stdD;

        /*
         * standard deviation of signal
         */
        RFLOAT _stdS;

        /*
         * standard deviation of standard deviation of noise
         */
        RFLOAT _stdStdN;

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

        RFLOAT* _ctfP;

        RFLOAT* _sigRcpP;

        /**
         * spatial frequency of each pixel
         */
        RFLOAT* _frequency;

        /**
         * defocus of each pixel of each image
         */
        RFLOAT* _defocusP;

        /**
         * K1 of CTF of each image
         */
        RFLOAT* _K1;

        /**
         * K2 of CTF of each image
         */
        RFLOAT* _K2;

        FFT _fftImg;

    public:
        
        Optimiser()
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

        ~Optimiser();

        OptimiserPara& para();

        void setPara(const OptimiserPara& para);

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

        void avgStdR(RFLOAT& stdR);

        void avgStdT(RFLOAT& stdT);

        void loadParticles();

        /**
         * re-calculate the rotation change between this iteration and the
         * previous one
         */
        void refreshRotationChange();

        void refreshClassDistr();

        void balanceClass(const RFLOAT thres,
                          const bool refreshDistr);

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
        void allReduceSigma(const bool mask,
                            const bool group);
        /**
         * reconstruct reference
         */
        void reconstructRef(const bool fscFlag,
                            const bool avgFlag,
                            const bool fscSave,
                            const bool avgSave,
                            const bool finished = false);

        /***
         * @param mask           whether mask on the reference is allowed or
         *                       not
         * @param solventFlatten whether solvent flatten on the reference is
         *                       allowed or not when mask is off
         */
        void solventFlatten(const bool mask = true);

        void allocPreCalIdx(const RFLOAT rU,
                            const RFLOAT rL);

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
        void saveMapHalf(const bool finished = false);

        void saveMapJoin(const bool finished = false);

        /**
         * save FSC
         */
        void saveFSC(const bool finished = false) const;

        void saveClassInfo(const bool finished = false) const;

        void saveSig() const;

        void saveTau() const;
};

/***
int searchPlace(RFLOAT* topW,
                const RFLOAT w,
                const int l,
                const int r);

void recordTopK(RFLOAT* topW,
                size_t* iTopR,
                size_t* iTopT,
                const RFLOAT w,
                const size_t iR,
                const size_t iT,
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
RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sigRcp,
                      const RFLOAT rU,
                      const RFLOAT rL);

RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sigRcp,
                      const int* iPxl,
                      const int* iSig,
                      const int m);

RFLOAT logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const RFLOAT* ctf,
                      const RFLOAT* sigRcp,
                      const int m);

RFLOAT logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const RFLOAT* frequency,
                      const RFLOAT* defocus,
                      const RFLOAT df,
                      const RFLOAT K1,
                      const RFLOAT K2,
                      const RFLOAT* sigRcp,
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
RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sigRcp,
                      const RFLOAT rU,
                      const RFLOAT rL);

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
RFLOAT logDataVSPrior(const Image& dat,
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
                   const RFLOAT rU,
                   const RFLOAT rL);

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
                   const RFLOAT* const* ctf,
                   const RFLOAT* const* sigRcp,
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
                   const RFLOAT* ctf,
                   const RFLOAT* sigRcp,
                   const int n,
                   const int m);

RFLOAT dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sigRcp,
                   const RFLOAT rU,
                   const RFLOAT rL);

RFLOAT dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& tra,
                   const Image& ctf,
                   const vec& sigRcp,
                   const RFLOAT rU,
                   const RFLOAT rL);

void scaleDataVSPrior(vec& sXA,
                      vec& sAA,
                      const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const RFLOAT rU,
                      const RFLOAT rL);

#endif // OPTIMSER_H

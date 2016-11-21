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

#include <vector>
#include <cstdlib>
#include <sstream>
#include <string>
#include <climits>
#include <queue>
#include <functional>

#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>

#include "Typedef.h"

#include "Image.h"
#include "Volume.h"
#include "ImageFile.h"
#include "Spectrum.h"
#include "Symmetry.h"
#include "CTF.h"
#include "Mask.h"

#include "Experiment.h"

#include "Particle.h"

#include "MLModel.h"

#define FOR_EACH_2D_IMAGE for (ptrdiff_t l = 0; l < static_cast<ptrdiff_t>(_ID.size()); l++)

#define ALPHA_GLOBAL_SEARCH 1.0
#define ALPHA_LOCAL_SEARCH 0

#define MIN_N_PHASE_PER_ITER 3
#define MAX_N_PHASE_PER_ITER 100

#define PERTURB_FACTOR_L 100
#define PERTURB_FACTOR_S 0.01

#define GEN_MASK_RES 30

#define TRANS_SEARCH_FACTOR 1
//#define TRANS_SEARCH_FACTOR 0.5

#define SWITCH_FACTOR 3

#define N_SAVE_IMG 20

#define PROCESS_LOGW_SOFT(logW) \
    [](vec& _logW) \
    { \
        _logW.array() -= _logW.maxCoeff(); \
        _logW.array() *= -1; \
        _logW.array() += 1; \
        _logW.array() = 1.0 / _logW.array(); \
    }(logW)

#define PROCESS_LOGW_HARD(logW) \
    [](vec& _logW) \
    { \
        _logW.array() -= _logW.maxCoeff(); \
        _logW.array() = exp(_logW.array()); \
    }(logW)

using namespace std;

typedef struct ML_OPTIMISER_PARA
{
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
     * number of sampling points for scanning in global search
     */
    int mS;

    /**
     * estimated translation (pixel)
     */
    double transS;

    /**
     * initial resolution (Angstrom)
     */
    double initRes;

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

    bool autoSelection = false;

    bool localCTF = false;

    /**
     * whether to perform masking on the reference
     */
    bool performMask = true;

    /**
     * whether to automatically generate a mask
     */
    bool autoMask = true;

    /**
     * mask
     */
    char mask[FILE_NAME_LENGTH];

    bool performSharpen = true;

    bool estBFactor = false;

    double bFactor = 200;

    /**
     * max number of iteration
     */
    int iterMax;
    
    /**
     * padding factor
     */
    int pf = 2;
    
    /**
     * MKB kernel radius
     */
    double a = 1.9;

    /**
     * MKB kernel smooth factor
     */
    double alpha = 10;

    /**
     * number of sampling points in global search
     */
    int mG;

    /**
     * number of sampling points in local search
     */
    int mL;

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
    double thresCutoffFSC = 0.5;

    /**
     * the FSC threshold for reporting resolution
     */
    double thresReportFSC = 0.143;

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

} MLOptimiserPara;

void display(const MLOptimiserPara& para);

typedef struct CTF_ATTR
{

    double voltage;

    double defocusU;

    double defocusV;

    double defocusAngle;

    double CS;

} CTFAttr;

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
        int _searchType = SEARCH_TYPE_GLOBAL;

        /**
         * model containting references, projectors, reconstruuctors, information 
         * about FSC, SNR and determining the cutoff frequency and search type
         */
        MLModel _model;

        /**
         * a database containing information of 2D images, CTFs, group and
         * micrograph information
         */
        Experiment _exp;

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
         * a particle filter for each 2D image
         */
        vector<Particle> _par;

        /**
         * a CTF for each 2D image
         */
        vector<Image> _ctf;

        /**
         * whether to use the image in calculating sigma and reconstruction or
         * not
         */
        vector<bool> _switch;

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

        /*
         * standard deviation of noise
         */
        double _stdN = 0;

        /*
         * standard deviation of data
         */
        double _stdD = 0;

        /*
         * standard deviation of signal
         */
        double _stdS = 0;

        /*
         * standard deviation of standard deviation of noise
         */
        double _stdStdN = 0;

        /**
         * whether to generate mask or not
         */
        bool _genMask = false;

        /**
         * mask
         */
        Volume _mask;

        /**
         * number of performed filtering in an iteration of a process
         */
        int _nF = 0;

        /**
         * number of performed images in an iteration of a process
         */
        int _nI = 0;

        /**
         * number of performed rotations in the scanning phase of the global
         * search stage
         */
        int _nR = 0;

        int* _iPxl = NULL;

        int* _iCol = NULL;
        
        int* _iRow = NULL;

        int* _iSig = NULL;

        Complex** _datP = NULL;

        double** _ctfP = NULL;

        double** _sigRcpP = NULL;

    public:
        
        MLOptimiser();

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
         * initialise the switches on images which determine whether an image
         * will be used in calculating sigma and recosntruction or not
         */
        void initSwitch();

        /**
         * correct the intensity scale
         *
         * @param init  whether it is an initial correction or not
         * @param group grouping or not
         */
        void correctScale(const bool init = false,
                          const bool group = true);

        /**
         * initialise sigma
         */
        void initSigma();

        /**
         * initialise particle filters
         */
        void initParticles();

        /**
         * re-calculate the rotation change between this iteration and the
         * previous one
         */
        void refreshRotationChange();

        /**
         * re-calculate the rotation and translation variance
         */
        void refreshVariance();

        /**
         * re-determine whether to use an image in calculating sigma and
         * reconstruction or not
         */
        void refreshSwitch();

        /**
         * re-calculate the intensity scale
         *
         * @param init  whether it is an initial correction or not
         * @param group grouping or not
         */
        void refreshScale(const bool init = false,
                          const bool group = true);

        /**
         * re-calculate sigma
         *
         * @param group grouping or not
         */
        void allReduceSigma(const bool group = true);

        /**
         * reconstruct reference
         *
         * @param mask whether mask on the reference is allowed or not
         */
        void reconstructRef(const bool mask = true);

        void allocPreCal(int& nPxl,
                         const double rU,
                         const double rL);

        void freePreCal();

        /**
         * for debug, save the best projections
         */
        void saveBestProjections();

        /**
         * for debug, save the images
         */
        void saveImages();

        /**
         * for debug, save the binning images
         */
        void saveBinImages();

        /**
         * for debug, save the CTFs
         */
        void saveCTFs();

        /**
         * for debug, save the low pass filtered images
         */
        void saveLowPassImages();

        /**
         * save the reference(s)
         *
         * @param finished whether it is the final round or not
         */
        void saveReference(const bool finished = false);

        void saveSharpReference();

        /**
         * save the mask
         */
        void saveMask();

        /**
         * save FSC
         */
        void saveFSC(const bool finished = false) const;
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

vec logDataVSPrior(const Complex* const* dat,
                   const Complex* pri,
                   const double* const* ctf,
                   const double* const* sigRcp,
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

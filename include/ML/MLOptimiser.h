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

#define TRANS_SEARCH_FACTOR 0.1

#define SWITCH_FACTOR 3

#define PROCESS_LOGW(logW) \
    [](vec& _logW) \
    { \
        _logW.array() -= _logW.maxCoeff(); \
        _logW.array() *= -1; \
        _logW.array() += 1; \
        _logW.array() = 1.0 / _logW.array(); \
        _logW.array() -= _logW.minCoeff(); \
    }(logW);

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

    /**
     * max number of iteration
     */
    int iterMax;
    
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

        void substractBgImg();

        void maskImg();

        /* normlise 2D images */
        void normaliseImg();

        /* perform Fourier transform */
        void fwImg();

        /* perform inverse Fourier transform */
        void bwImg();

        void initCTF();

        void initSwitch();

        void initImgReduceCTF();

        void correctScale(const bool init = false,
                          const bool group = true);

        void initSigma();

        void initParticles();

        void refreshRotationChange();

        void refreshVariance();

        void refreshSwitch();

        void refreshScale(const bool init = false,
                          const bool group = true);

        void allReduceSigma(const bool group = true);

        void reconstructRef(const bool mask = true);

        // for debug
        // save the best projections to BMP file
        void saveBestProjections();

        // for debug
        // save images to BMP file
        void saveImages();

        void saveBinImages();

        // debug
        // save CTFs to BMP file
        void saveCTFs();

        /***
        // for debug
        // save images after removing CTFs
        void saveReduceCTFImages();
        ***/

        // for debug
        // save low pass filtered images
        void saveLowPassImages();

        /***
        // for debug
        // save low pass filtered images after removing CTFs
        void saveLowPassReduceCTFImages();
        ***/

        void saveReference(const bool finished = false);

        void saveMask();

        void saveFSC() const;
};

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sig,
                      const double rU,
                      const double rL);
// dat -> data, pri -> prior, ctf

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sig,
                      const double rU,
                      const double rL);

vec logDataVSPrior(const vector<Image>& dat,
                   const Image& pri,
                   const vector<Image>& ctf,
                   const vector<int>& groupID,
                   const mat& sig,
                   const double rU,
                   const double rL);

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sig,
                   const double rU,
                   const double rL);

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& tra,
                   const Image& ctf,
                   const vec& sig,
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

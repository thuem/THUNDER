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

#define FREQ_DOWN_CUTOFF 1.5
#define ALPHA_GLOBAL_SEARCH 1.0
#define ALPHA_LOCAL_SEARCH 0
#define TOTAL_GLOBAL_SEARCH_RES_LIMIT 15 // Angstrom
//#define TOTAL_GLOBAL_SEARCH_RES_LIMIT 20 // Angstrom
//#define TOTAL_GLOBAL_SEARCH_RES_LIMIT 50 // Angstrom

#define MIN_N_PHASE_PER_ITER 3
#define MAX_N_PHASE_PER_ITER 100

#define PERTURB_FACTOR_L 100
#define PERTURB_FACTOR_S 0.01

#define MASK_RATIO 1.0
#define GEN_MASK_RES 30

#define TRANS_SEARCH_FACTOR 0.1

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
    int iterMax;
    // max number of iterations
    
    int k;
    // number of references

    int size;
    // size of references and images

    int pf;
    // pading factor
    
    double a;
    // parameter of the kernel MKB_FT

    double alpha;
    // parameter of the kernel MKB_FT

    double pixelSize;
    // pixel size of 2D images

    int mG;
    // number of samplings in particle filter

    int mL;
    // number of samplings in particle filter

    double transS;

    // initial estimated resolution (Angstrom)
    double initRes;

    // the information below this resolution will be ignored
    double ignoreRes;

    char sym[SYM_ID_LENGTH];

    char initModel[FILE_NAME_LENGTH];
    // the initial model for this iteration

    char db[FILE_NAME_LENGTH];

} MLOptimiserPara;

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
         * current number of iterations
         */
        int _iter;

        /**
         * current resolution (in Angstrom)
         */
        double _res;

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
         * 2D images after reducing CTF using Wiener filter
         */
        vector<Image> _imgReduceCTF;

        /**
         * a particle filter for each 2D image
         */
        vector<Particle> _par;

        /**
         * a CTF for each 2D image
         */
        vector<Image> _ctf;

        /**
         * Each row stands for sigma^2 of a certain group, thus the size of this
         * matrix is _nGroup x (maxR() + 1)
         */
        mat _sig;

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

        bool _genMask = false;

        Volume _mask;

        /***
        double _noiseStddev = 0;

        double _dataStddev = 0;

        double _signalStddev = 0;
        ***/

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
        /* set parameters of _model
         * setMPIEnv of _model
         * read in images from hard-disk
         * generate corresponding CTF */

        //Yu Hongkun ,Wang Kunpeng
        void expectation();

        //Guo Heng, Li Bing
        void maximization();

        void run();

        void clear();

    private:

        void bCastNPar();

        void allReduceN();

        int size() const;
        /* size of 2D image */
        
        int maxR() const;
        /* max value of _r */

        void bcastGroupInfo();
        /* broadcast information of groups */

        void initRef();

        void initID();
        /* save IDs from database */

        /* read 2D images from hard disk */
        void initImg();

        void statImg();

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

        void initImgReduceCTF();

        void correctScale();

        void refreshRotationChange();

        void initSigma();

        void initParticles();

        void allReduceSigma();

        void reconstructRef();

        // for debug
        // save the best projections to BMP file
        void saveBestProjections();

        // for debug
        // save images to BMP file
        void saveImages();

        // debug
        // save CTFs to BMP file
        void saveCTFs();

        // for debug
        // save images after removing CTFs
        void saveReduceCTFImages();

        // for debug
        // save low pass filtered images
        void saveLowPassImages();

        // for debug
        // save low pass filtered images after removing CTFs
        void saveLowPassReduceCTFImages();

        void saveReference();

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

#endif // ML_OPTIMSER_H

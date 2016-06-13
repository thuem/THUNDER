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

#define TOP_K 1

#define FREQ_DOWN_CUTOFF 3
#define N_ITER_TOTAL_GLOBAL_SEARCH 15
#define N_ITER_PARTIAL_GLOBAL_SEARCH 10
#define ALPHA_TOTAL_GLOBAL_SEARCH 1.0
#define ALPHA_TOTAL_GLOBAL_SEARCH_1_PASS 0.5
#define ALPHA_TOTAL_GLOBAL_SEARCH_2_PASS 0.3
#define ALPHA_GLOBAL_SEARCH_MAX 0.75
#define ALPHA_GLOBAL_SEARCH_MIN 0.75
//#define ALPHA_GLOBAL_SEARCH_MAX 1.0
//#define ALPHA_GLOBAL_SEARCH_MIN 1.0
#define ALPHA_LOCAL_SEARCH 0.1
#define ALPHA_SEARCH_BG 0.05
#define TOTAL_GLOBAL_SEARCH_RES_LIMIT 20 // Angstrom

#define MAX_N_PHASE_PER_ITER 30
#define MAX_N_SEARCH_PER_PHASE 3

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

    int m;
    // number of samplings in particle filter
    
    int mf;
    // factor of number of samplings in particle filter

    int maxX;

    int maxY;

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

        int _N;
        /* total number of 2D images in each hemisphere */

        int _r;
        /* radius of calculating posterior possibility */

        int _iter;
        /* number of iterations performed */

        double _res;
        /* current resolution in pixel */

        MLModel _model;
        /* model, including references, projectors and reconstructors */

        Experiment _exp;
        /* information of 2D images, groups and micrographs */

        Symmetry _sym; 

        vector<int> _ID;
        /* IDs for each 2D images */

        vector<Image> _img;

        vector<Particle> _par;

        vector<Image> _ctf;

        // vector<vec> _sig;

        mat _sig;
        // each row is a sigma value
        // size : _nGroup * (maxR() + 1)

        int _nGroup;

        // vector<int> _groupSize;

        vector<int> _groupID;

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

        void initImg();
        /* read 2D images from hard disk */

        void initCTF();

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
};

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sig,
                      const int r);
// dat -> data, pri -> prior, ctf

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sig,
                   const int r);

#endif // ML_OPTIMSER_H

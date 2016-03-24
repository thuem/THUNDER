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
#include "Parallel.h"

#include "Experiment.h"

#include "Particle.h"

#include "MLModel.h"

using namespace std;

typedef struct ML_OPTIMISER_PARA
{
    int iterMax;
    // max number of iterations

    int pf;
    // pading factor
    
    double a;
    // parameter of the kernel MKB_FT

    double alpha;
    // parameter of the kernel MKB_FT

    double pixelSize;
    // pixel size of 2D images

    int M;
    // number of samplings in particle filter

    int maxX;

    int maxY;

    char sym[SYM_ID_LENGTH];

    char initModel[FILE_NAME_LENGTH];
    // the initial model for this iteration

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

        vector<vec> _sig;

    public:

        MLOptimiser();

        ~MLOptimiser();

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

        void initID();
        /* save IDs from database */

        void initImg();
        /* read 2D images from hard disk */

        void initCTF();

        void initSigma();

        void initParticles();

        void allReduceSigma();

        void reconstructRef();
};

double dataVSPrior(const Image& A,
                   const Image& B,
                   const Image& ctf,
                   const vec& sig,
                   const int r);

#endif // ML_OPTIMSER_H

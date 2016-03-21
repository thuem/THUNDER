/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
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

#include "Typedef.h"

#include "Image.h"
#include "Volume.h"
#include "ImageFile.h"
#include "Spectrum.h"
#include "Symmetry.h"
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

} MLOptimiserPara;

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

        vector<Image> _img;

        vector<Particle> _par;

        vector<vec> _ctf;

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

        void initSigma();

        void allReduceSigma();

        void reconstructRef();
};

#endif // ML_OPTIMSER_H

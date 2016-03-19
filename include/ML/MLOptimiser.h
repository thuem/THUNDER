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

#include "Parallel.h"

#include "Particle.h"

#include "MLModel.h"

using namespace std;

typedef struct ML_OPTIMISER_PARA
{
    int _pf; // pading factor
    double a; // parameter of the kernel MKB_FT
    double alpha; // parameter of the kernel MKB_FT
    double pixelSize; // pixel size of 2D images
} MLOptimiserPara;

class MLOptimiser : public Parallel
{
    private:

        int _r;
        /* radius of calculating posterior possibility */

        MLOptimiserPara _para;

        MLModel _model;

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
        void maxmization();

        void run();

        void clear();

    private:

        void resetProjectors();
};

#endif // ML_OPTIMSER_H

/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ML_OPTIMISER_H
#define ML_OPTIMISER_H

#include "Typedef.h"

#include "Image.h"
#include "Volume.h"

#include "Parallel.h"
#include "Particle.h"

#include "MLModel.h"

class MLOptimiser : public Parallel
{
    private:

        MLModel _model;

        vector<Image> _img;
        vector<Particle> _par;
        vector<vec> _ctf;

        vector<vec> _sig;

    public:

        MLOptimiser();
        ~MLOptimiser();

        void init();

        //Yu Hongkun ,Wang Kunpeng
        void expectation();

        //Guo Heng, Li Bing
        void maxmization();

        void clear();

    private:

        void resetProjectors();
};

#endif // ML_OPTIMSER_H

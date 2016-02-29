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

#include "Image.h"
#include "Volume.h"
#include "Projector.h"

class MLOptimiser
{
    private:

        vector<Volume> _references;
        /* references in Fourier space */

        vector<Projector> _projectors;

    public:

        MLOptimiser();

        initLowPassFilterReferences();
};

#endif // ML_OPTIMSER_H

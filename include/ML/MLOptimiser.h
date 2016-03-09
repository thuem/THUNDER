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

#include "MLModel.h"

class MLOptimiser
{
    private:

        MLModel _model;

    public:

        MLOptimiser();

    private:

        resetProjectors();
};

#endif // ML_OPTIMSER_H

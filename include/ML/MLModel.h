/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <armadillo>

#include "Image.h"
#include "Volume.h"

using namespace std;
using namespace arma;

class MLModel
{
    private:

        vector<Volume> _ref;
        /* references in Fourier space */

        vector<vec> _FSC;

        vector<vec> _SNR;

    public:

        MLModel();

        int K(); const;

        int size() const;

        void BCastFSC();

        void BCastSNR();
};

#endif // ML_MODEL_H

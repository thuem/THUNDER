#ifndef RECONSTRUCTMANAGER_CUH
#define RECONSTRUCTMANAGER_CUH

#include "Config.h"
#include "Precision.h"

#include "Device.cuh"
#include "TabFunction.cuh"

namespace cuthunder {

///////////////////////////////////////////////////////////////
/*
class ReconstructManager
{
    public:

        static ReconstructManager* GetInstance();

        static void DestroyInstance();

        ~ReconstructManager();

    private:

        ReconstructManager();

        static ReconstructManager *_instance;

    public:

        void RegisterTabFunction(double begin,
                                 double end,
                                 double step,
                                 double *tab,
                                 int size);

        TabFunction GetTabFunction() const;

        void DestroyTabFunction();

        void RegisterPGLKVolume(double *vol, int dim);

    private:

        TabFunction _tabfunc;
};

ReconstructManager* ReconstructManager::_instance = NULL;
*/
///////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////
#endif

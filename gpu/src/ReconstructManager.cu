#include "ReconstructManager.cuh"

namespace cuthunder {

///////////////////////////////////////////////////////////////
/*
ReconstructManager::ReconstructManager() {}

ReconstructManager::~ReconstructManager() {

    DestroyTabFunction();
}

ReconstructManager* ReconstructManager::GetInstance()
{
    if (_instance == NULL) {
        _instance = new ReconstructManager();
    }

    return _instance;
}

void ReconstructManager::DestroyInstance()
{
    if (_instance) {
        delete _instance;
        _instance = NULL;
    }
}

void ReconstructManager::RegisterTabFunction(double begin,
                                             double end,
                                             double step,
                                             double *tab,
                                             int size)
{
    DestroyTabFunction();

    double *dev_tab;

    cudaMalloc((void**)&dev_tab, size * sizeof(double));
    cudaCheckErrors("Allocate device tabfunction.");

    cudaMemcpy(dev_tab,
               tab,
               size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("Data copy to device tabfunction.");

    _tabfunc.init(begin, end, step, dev_tab, size);
}

void ReconstructManager::DestroyTabFunction()
{
    if (_tabfunc.devPtr()) {
        cudaFree(_tabfunc.devPtr());
        cudaCheckErrors("Free tabfunc data.");

        _tabfunc = TabFunction();
    }
}

TabFunction ReconstructManager::GetTabFunction() const
{
    return _tabfunc;
}
*/
///////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////

#include <MLOptimiser.h>


MLOptimiser::MLOptimiser() {}

MLOptimiser::~MLOptimiser()
{
    clear();
}


void MLOptimiser::init()
{
    // set MPI environment of _model
    _model.setMPIEnv(_commSize, _commRank, _hemi);
}


void MLOptimiser::expectation()
{

}

void MLOptimiser::maxmization()
{

}

void MLOptimiser::run()
{

}

void MLOptimiser::clear()
{
    _img.clear();
    _par.clear();
    _ctf.clear();
}

void MLOptimiser::resetProjectors()
{

}

/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

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

    // append initial references into _model

    // apply low pass filter on initial references

    // read in images from hard disk

    // genereate corresponding CTF
    
    // estimate initial sigma values
    initSigma();
}

void MLOptimiser::expectation()
{

    int N = 6000;
    double maxX = 30;
    double maxY = 30;

    vector<Image>::iterator imgIter;

    for (imgIter = _img.begin(); imgIter < _img.end(); imgIter++) 
    {
        Particle p(N, maxX, maxY, &_sym);

       //handle one ref 
        for (int i = 0; i < N; i++)
        {

        }


    }
    


}

void MLOptimiser::maximization()
{
    /* generate sigma for the next iteration */
    allReduceSigma();

    /* reconstruct references */
    reconstructRef();
}

void MLOptimiser::run()
{
    init();

    for (int i = 0; i < _para.iterMax; i++)
    {
        expectation();

        maximization();

        /* calculate FSC */
        _model.BcastFSC();

        /* record current resolution */
        _res = _model.resolutionP();

        /* update the radius of frequency for computing */
        _model.updateR();
        _r = _model.r() / 2;
    }
}

void MLOptimiser::clear()
{
    _img.clear();
    _par.clear();
    _ctf.clear();
}

void MLOptimiser::initSigma()
{
}

void MLOptimiser::allReduceSigma()
{
    // TODO
}

void MLOptimiser::reconstructRef()
{
    // TODO
}

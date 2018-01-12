//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Bing Li, Heng Guo
 * Dependency:
 * Test:
 * Execution: * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Optimiser.h"

/***
void display(const OptimiserPara& para)
{
    printf("Number of Classes:                                       %12d\n", para.k);
    printf("Size of Image:                                           %12d\n", para.size);
    printf("Pixel Size (Angstrom):                                   %12.6lf\n", para.pixelSize); 
    printf("Radius of Mask on Images (Angstrom):                     %12.6lf\n", para.maskRadius);
    printf("Number of Sampling Points for Scanning in Global Search: %12d\n", para.mS);
    printf("Estimated Translation (Pixel):                           %12.6lf\n", para.transS);
    printf("Initial Resolution (Angstrom):                           %12.6lf\n", para.initRes);
    printf("Perform Global Search Under (Angstrom):                  %12.6lf\n", para.globalSearchRes);
    printf("Symmetry:                                                %12s\n", para.sym);
    printf("Initial Model:                                           %12s\n", para.initModel);
    printf(".thu File Storing Paths and CTFs of Images:           %12s\n", para.db);
    
    printf("Perform Reference Mask:                                  %12d\n", para.performMask);
    printf("Automask:                                                %12d\n", para.autoMask);
    printf("Mask:                                                    %12s\n", para.mask);

    printf("Max Number of Iteration:                                 %12d\n", para.iterMax);
    printf("Padding Factor:                                          %12d\n", para.pf);
    printf("MKB Kernel Radius:                                       %12.6lf\n", para.a);
    printf("MKB Kernel Smooth Factor:                                %12.6lf\n", para.alpha);
    printf("Number of Sampling Points in Global Search (Max):        %12d\n", para.mGMax);
    printf("Number of Sampling Points in Global Search (Min):        %12d\n", para.mGMin);
    //printf("Number of Sampling Points in Local Search:               %12d\n", para.mL);
    printf("Ignore Signal Under (Angstrom):                          %12.6lf\n", para.ignoreRes);
    printf("Correct Intensity Scale Using Signal Under (Angstrom):   %12.6lf\n", para.sclCorRes);
    printf("FSC Threshold for Cutoff Frequency:                      %12.6lf\n", para.thresCutoffFSC);
    printf("FSC Threshold for Reporting Resolution:                  %12.6lf\n", para.thresReportFSC);
    printf("Grouping when Calculating Sigma:                         %12d\n", para.groupSig);
    printf("Grouping when Correcting Intensity Scale:                %12d\n", para.groupScl);
    printf("Mask Images with Zero Noise:                             %12d\n", para.zeroMask);
    printf("CTF Refine Factor:                                       %12.6lf\n", para.ctfRefineFactor);
    printf("CTF Refine Standard Deviation                            %12.6lf\n", para.ctfRefineS);
}
***/
RFLOAT logDataVSPrior_m_huabin(const Complex* dat, const Complex* pri, const RFLOAT* ctf, const RFLOAT* sigRcp, const int m);

vec logDataVSPrior_m_n_huabin(const Complex* dat, const Complex* pri, const RFLOAT* ctf, const RFLOAT* sigRcp, const int n, const int m);



void compareDVPVariable(vec& dvpHuabin, vec& dvpOrig, int processRank, int threadID, int n ,int m)
{
    fprintf(stderr, "n = %d, m = %d\n", n, m);
    fprintf(stderr, "[%d:%d]: Elements number in dvpHuabin: [rows, cols] = [%ld, %ld]\n", processRank ,threadID, dvpHuabin.rows(), dvpHuabin.cols());
    fprintf(stderr, "[%d:%d]: Elements number in dvpHuabin: [rows, cols] = [%ld, %ld]\n", processRank ,threadID, dvpOrig.rows(), dvpOrig.cols());
    size_t cnt = 0;
    for(int i = 0; i < dvpHuabin.rows(); i ++)
    {
        RFLOAT v1 = dvpHuabin(i);
        RFLOAT v2 = dvpOrig(i);

        RFLOAT error = fabsf(v1 -v2);
        if(error >= 10E-6)
        {
            fprintf(stderr, "[%d:%d]: [v1hb, v2orig, error] = [%f, %f, %f]\n", processRank ,threadID, v1, v2, error);
            cnt ++;

            if(cnt == 100)
            {
                abort();
            }
        }
    }

    if(cnt == 0)
    {
    
        fprintf(stderr, "[%d:%d]: dvpHuabin and dvpOrig is exactly the same\n", processRank ,threadID);
    }
}

void compareWInmHuabin(RFLOAT wOrig, RFLOAT wHuabin, int processRank, int threadID, int m)
{

    fprintf(stderr, "[%d:%d]:  m = %d\n", processRank, threadID, m);
    RFLOAT error = fabs(wOrig - wHuabin);
    size_t cnt = 0;
    if(error >= 10E-6)
    {
        fprintf(stderr, "[%d:%d]: [wOrig, wHuabin, error] = [%f, %f, %f]\n", processRank ,threadID, wOrig, wHuabin, error);

        cnt ++;

        if(cnt == 100)
        {
            abort();
        }
    }

    if(cnt == 0)
    {
    
        fprintf(stderr, "[%d:%d]: wHuabin and wOrig is exactly the same\n", processRank ,threadID);
    }
}


Optimiser::~Optimiser()
{
    clear();

    _fftImg.fwDestroyPlanMT();
    _fftImg.bwDestroyPlanMT();
}

OptimiserPara& Optimiser::para()
{
    return _para;
}

void Optimiser::setPara(const OptimiserPara& para)
{
    _para = para;
}

void Optimiser::init()
{
    if (_para.mode == MODE_2D)
    {
        MLOG(INFO, "LOGGER_INIT") << "The Program is Running under 2D Mode";
    }
    else if (_para.mode == MODE_3D)
    {
        MLOG(INFO, "LOGGER_INIT") << "The Program is Running under 3D Mode";
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");
        
        abort();
    }

    MLOG(INFO, "LOGGER_INIT") << "Setting MPI Environment of _model";
    _model.setMPIEnv(_commSize, _commRank, _hemi);

    MLOG(INFO, "LOGGER_INIT") << "Setting up Symmetry";
    _sym.init(_para.sym);

    MLOG(INFO, "LOGGER_INIT") << "Number of Class(es): " << _para.k;

    MLOG(INFO, "LOGGER_INIT") << "Initialising FFTW Plan";

    _fftImg.fwCreatePlanMT(_para.size, _para.size);
    _fftImg.bwCreatePlanMT(_para.size, _para.size);

    MLOG(INFO, "LOGGER_INIT") << "Initialising Class Distribution";
    _cDistr.resize(_para.k);

    MLOG(INFO, "LOGGER_INIT") << "Passing Parameters to _model";
    _model.init(_para.mode,
                _para.gSearch,
                _para.lSearch,
                _para.cSearch,
                _para.coreFSC,
                AROUND(_para.maskRadius / _para.pixelSize),
                _para.maskFSC,
                &_mask,
                _para.goldenStandard,
                _para.k,
                _para.size,
                0,
                _para.pf,
                _para.pixelSize,
                _para.a,
                _para.alpha,
                &_sym);

    MLOG(INFO, "LOGGER_INIT") << "Determining Search Type";

    if (_para.gSearch)
    {
        _searchType = SEARCH_TYPE_GLOBAL;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : Global";
    }
    else if (_para.lSearch)
    {
        _searchType = SEARCH_TYPE_LOCAL;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : Local";
    }
    else if (_para.cSearch)
    {
        _searchType = SEARCH_TYPE_CTF;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : CTF";
    }
    else
    {
        REPORT_ERROR("WRONG INITIAL SEARCH TYPE");

        abort();
    }

    _model.setSearchType(_searchType);

    /***
    MLOG(INFO, "LOGGER_INIT") << "Initialising Upper Boundary of Reconstruction";

    _model.updateRU();
    ***/

    /***
    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _para.ignoreRes
                              << " Angstrom will be Ingored during Comparison";

                              ***/
    _rL = FLOOR(resA2P(1.0 / _para.ignoreRes, _para.size, _para.pixelSize));
    //_rL = 0;
    //_rL = 1.5;
    //_rL = 3.5;
    //_rL = 6;
    //_rL = resA2P(1.0 / (2 * _para.maskRadius), _para.size, _para.pixelSize);
    //_rL = resA2P(1.0 / _para.maskRadius, _para.size, _para.pixelSize);

    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _rL
                              << " Pixels in Fourier Space will be Ignored during Comparison";

    MLOG(INFO, "LOGGER_INIT") << "Checking Radius of Mask";

    /***
    CLOG(INFO, "LOGGER_SYS") << "_para.size / 2 = " << _para.size / 2;
    CLOG(INFO, "LOGGER_SYS") << "CEIL(_para.maskRadius / _para.pxielSize) = "
                             << CEIL(_para.maskRadius / _para.pixelSize);
    ***/

    if (_para.size / 2 - CEIL(_para.maskRadius / _para.pixelSize) < 1)
    {
        REPORT_ERROR("INPROPER RADIUS OF MASK");
        abort();
    }

    //_rS = AROUND(resA2P(1.0 / _para.sclCorRes, _para.size, _para.pixelSize)) + 1;

    if (_para.gSearch)
    {
        MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                  << _para.sclCorRes
                                  << " Angstrom will be Used for Performing Intensity Scale Correction";

        _rS = AROUND(resA2P(1.0 / _para.sclCorRes, _para.size, _para.pixelSize)) + 1;
        
        MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                  << _rS
                                  << " (Pixel) will be Used for Performing Intensity Scale Correction";

    }
    else
    {
         MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                   << _para.initRes
                                   << " Angstrom will be Used for Performing Intensity Scale Correction";

         _rS = AROUND(resA2P(1.0 / _para.initRes, _para.size, _para.pixelSize)) + 1;
         
         MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                   << _rS
                                   << " (Pixel) will be Used for Performing Intensity Scale Correction";
    }

    if (_para.gSearch)
    {
        MLOG(INFO, "LOGGER_INIT") << "Seting Frequency Upper Boudary during Global Search";

        _model.setRGlobal(AROUND(resA2P(1.0 / _para.globalSearchRes,
                                 _para.size,
                                 _para.pixelSize)) + 1);

        MLOG(INFO, "LOGGER_INIT") << "Global Search Resolution Limit : "
                                  << _para.globalSearchRes
                                  << " (Angstrom), "
                                  << _model.rGlobal()
                                  << " (Pixel)";
    }

    MLOG(INFO, "LOGGER_INIT") << "Setting Parameters: _r, _iter";

    _iter = 0;

    _r = AROUND(resA2P(1.0 / _para.initRes, _para.size, _para.pixelSize)) + 1;
    _model.setR(_r);

    MLOG(INFO, "LOGGER_INIT") << "Setting MPI Environment of _exp";
    _db.setMPIEnv(_commSize, _commRank, _hemi);

    MLOG(INFO, "LOGGER_INIT") << "Openning Database File";
    _db.openDatabase(_para.db);

    MLOG(INFO, "LOGGER_INIT") << "Shuffling Particles";
    _db.shuffle();

    MLOG(INFO, "LOGGER_INIT") << "Assigning Particles to Each Process";
    _db.assign();

    MLOG(INFO, "LOGGER_INIT") << "Indexing the Offset in Database";
    _db.index();

    MLOG(INFO, "LOGGER_INIT") << "Appending Initial References into _model";
    initRef();

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting Total Number of 2D Images";
    bCastNPar();

    MLOG(INFO, "LOGGER_INIT") << "Total Number of Images: " << _nPar;

    if ((_para.maskFSC) ||
        (_para.performMask && !_para.autoMask))
    {
        MLOG(INFO, "LOGGER_INIT") << "Reading Mask";

        initMask();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_INIT") << "Mask Read";
#endif
    }

    NT_MASTER
    {
        ALOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";
        BLOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";

        initID();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "IDs of 2D Images Initialised";
        BLOG(INFO, "LOGGER_INIT") << "IDs of 2D Images Initialised";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Setting Parameter _N";
        BLOG(INFO, "LOGGER_INIT") << "Setting Parameter _N";

        allReduceN();

        ALOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere A: " << _N;
        BLOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere B: " << _N;

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "Parameter _N Set";
        BLOG(INFO, "LOGGER_INIT") << "Parameter _N Set";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";
        BLOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";

        initImg();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "2D Images Initialised";
        BLOG(INFO, "LOGGER_INIT") << "2D Images Initialised";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Generating CTFs";
        BLOG(INFO, "LOGGER_INIT") << "Generating CTFs";

        initCTF();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "CTFs Generated";
        BLOG(INFO, "LOGGER_INIT") << "CTFs Generated";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";
        BLOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";

        initParticles();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "Particle Filters Initialised";
        BLOG(INFO, "LOGGER_INIT") << "Particle Filters Initialised";
#endif

        if (!_para.gSearch)
        {
            ALOG(INFO, "LOGGER_INIT") << "Loading Particle Filters";
            BLOG(INFO, "LOGGER_INIT") << "Loading Particle Filters";

            loadParticles();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(_hemi);

            ALOG(INFO, "LOGGER_INIT") << "Particle Filters Loaded";
            BLOG(INFO, "LOGGER_INIT") << "Particle Filters Loaded";
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

            ALOG(INFO, "LOGGER_INIT") << "Re-Centring Images";
            BLOG(INFO, "LOGGER_INIT") << "Re-Centring Images";

            reCentreImg();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(_hemi);

            ALOG(INFO, "LOGGER_INIT") << "Images Re-Centred";
            BLOG(INFO, "LOGGER_INIT") << "Images Re-Centred";
#endif
#endif

#ifdef OPTIMISER_MASK_IMG

            MLOG(INFO, "LOGGER_ROUND") << "Re-Masking Images";
            reMaskImg();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(_hemi);

            ALOG(INFO, "LOGGER_INIT") << "Images Re-Masked";
            BLOG(INFO, "LOGGER_INIT") << "Images Re-Masked";
#endif
#endif
        }
    }

    MLOG(INFO, "LOGGER_INIT") << "Broadacasting Information of Groups";

    bcastGroupInfo();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Information of Groups Broadcasted";
#endif

    NT_MASTER
    {
        /***
#ifdef OPTIMISER_SOLVENT_FLATTEN
        ALOG(INFO, "LOGGER_ROUND") << "Applying Solvent Flatten on Reference(s)";
        BLOG(INFO, "LOGGER_ROUND") << "Applying Solvent Flatten on Reference(s)";

        solventFlatten(_para.performMask);
#endif
        ***/

        ALOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";
        BLOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";

        _model.initProjReco();
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Projectors and Reconstructors Set Up";
#endif

    if (strcmp(_para.initModel, "") != 0)
    {
        MLOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale";

        if (_para.gSearch)
        {
            MLOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale Using Random Projections";

            correctScale(true, false, false);
        }
        else
        {
            MLOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale Using Given Projections";

            correctScale(true, true, false);
        }

        NT_MASTER
        {
            ALOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors After Intensity Scale Correction";
            BLOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors After Intensity Scale Correction";

            _model.refreshProj();
        }

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);
    
        MLOG(INFO, "LOGGER_INIT") << "Intensity Scale Re-balanced";
#endif
    }

    NT_MASTER
    {
        ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";
        BLOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";

        initSigma();

        if (_para.gSearch)
        {
            ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma Using Random Projections";
            BLOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma Using Random Projections";
            
            initSigma();
        }
        else
        {
            ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma Using Given Projections";
            BLOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma Using Given Projections";

            if (_para.k == 1)
            {
#ifdef OPTIMISER_SIGMA_MASK
                allReduceSigma(true, false);
#else
                // refinement, focus on avoiding over-refinement
                allReduceSigma(false, false);
#endif
            }
            else
            {
                // classification, focus on searching for difference
                allReduceSigma(true, false);
            }
        }
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Sigma Initialised";
#endif
}

struct Sp
{
    RFLOAT _w;
    size_t _k;
    size_t _iR;
    size_t _iT;

    Sp() : _w(-TS_MAX_RFLOAT_VALUE), _k(0), _iR(0), _iT(0) {};

    Sp(const RFLOAT w,
       const size_t k,
       const size_t iR,
       const size_t iT)
    {
        _w = w;
        _k = k;
        _iR = iR;
        _iT = iT;
    };
};

struct SpWeightComparator
{
    bool operator()(const Sp& a, const Sp& b) const
    {
        return a._w > b._w;
    }
};

void Optimiser::expectation()
{
    IF_MASTER return;

    int nPer = 0;

    ALOG(INFO, "LOGGER_ROUND") << "Allocating Space for Pre-calcuation in Expectation";
    BLOG(INFO, "LOGGER_ROUND") << "Allocating Space for Pre-calcuation in Expectation";

    allocPreCalIdx(_r, _rL);

    if (_searchType == SEARCH_TYPE_GLOBAL)
    {
        if (_searchType != SEARCH_TYPE_CTF)
            allocPreCal(true, false);
        else
            allocPreCal(true, true);

        ALOG(INFO, "LOGGER_ROUND") << "Space for Pre-calcuation in Expectation Allocated";
        BLOG(INFO, "LOGGER_ROUND") << "Space for Pre-calcuation in Expectation Allocated";

        // initialse a particle filter

        int nR;
        if (_para.mode == MODE_2D)
        { 
            nR = _para.mS;
        }
        else if (_para.mode == MODE_3D)
        {
            nR = _para.mS / (1 + _sym.nSymmetryElement());
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        int nT = GSL_MAX_INT(30,
                             AROUND(M_PI
                                  * TSGSL_pow_2(_para.transS
                                            * TSGSL_cdf_chisq_Qinv(0.5, 2))
                                  * _para.transSearchFactor));

        RFLOAT scanMinStdR;
        if (_para.mode == MODE_2D)
        { 
            scanMinStdR = 1.0 / _para.mS;
        }
        else if (_para.mode == MODE_3D)
        {
            scanMinStdR =  pow(_para.mS, -1.0 / 3);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        RFLOAT scanMinStdT = 1.0
                           / TSGSL_cdf_chisq_Qinv(0.5, 2)
                           / sqrt(_para.transSearchFactor * M_PI);

        ALOG(INFO, "LOGGER_ROUND") << "Minimum Standard Deviation of Rotation in Scanning Phase: "
                                   << scanMinStdR;
        ALOG(INFO, "LOGGER_ROUND") << "Minimum Standard Deviation of Translation in Scanning Phase: "
                                   << scanMinStdT;

        Particle par = _par[0].copy();

        par.reset(_para.k, nR, nT, 1);

        FOR_EACH_2D_IMAGE
        {
            // the previous top class, translation, rotation remain
            par.copy(_par[l]);
        }

        mat22 rot2D;
        mat33 rot3D;
        vec2 t;

        // generate "translations"

        Complex* traP = (Complex*)TSFFTW_malloc(nT * _nPxl * sizeof(Complex));

        #pragma omp parallel for schedule(dynamic) private(t)
        for (size_t m = 0; m < (size_t)nT; m++)
        {
            par.t(t, m);

            translate(traP + m * _nPxl,
                      t(0),
                      t(1),
                      _para.size,
                      _para.size,
                      _iCol,
                      _iRow,
                      _nPxl);
        }

        mat wC = mat::Zero(_ID.size(), _para.k);
        mat wR = mat::Zero(_ID.size(), nR);
        mat wT = mat::Zero(_ID.size(), nT);

        _nR = 0;

        omp_lock_t* mtx = new omp_lock_t[_ID.size()];

        RFLOAT* baseLine = new RFLOAT[_ID.size()];

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            omp_init_lock(&mtx[l]);
            baseLine[l] = GSL_NAN;
        }

        // t -> class
        // m -> rotation
        // n -> translation
        
        for (size_t t = 0; t < (size_t)_para.k; t++)
        {
            Complex* poolPriRotP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));
            Complex* poolPriAllP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));

            #pragma omp parallel for schedule(dynamic) private(rot2D, rot3D)
            for (size_t m = 0; m < (size_t)nR; m++)
            {
                Complex* priRotP = poolPriRotP + _nPxl * omp_get_thread_num();

                Complex* priAllP = poolPriAllP + _nPxl * omp_get_thread_num();

                // perform projection

                if (_para.mode == MODE_2D)
                {
                    par.rot(rot2D, m);

                    _model.proj(t).project(priRotP, rot2D, _iCol, _iRow, _nPxl);
                }
                else if (_para.mode == MODE_3D)
                {
                    par.rot(rot3D, m);

                    _model.proj(t).project(priRotP, rot3D, _iCol, _iRow, _nPxl);
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }

                for (size_t n = 0; n < (size_t)nT; n++)
                {
                    for (int i = 0; i < _nPxl; i++)
                        priAllP[i] = traP[_nPxl * n + i] * priRotP[i];

                    // higher logDataVSPrior, higher prabibility

                    vec dvp = logDataVSPrior_m_n_huabin(_datP,
                                             priAllP,
                                             _ctfP,
                                             _sigRcpP,
                                             (int)_ID.size(),
                                             _nPxl);

/*
 *                    vec dvpOrig = logDataVSPrior(_datP,
 *                                             priAllP,
 *                                             _ctfP,
 *                                             _sigRcpP,
 *                                             (int)_ID.size(),
 *                                             _nPxl);
 *
 *                    compareDVPVariable(dvp, dvpOrig, _commRank, omp_get_thread_num(), (int)_ID.size(), _nPxl
 *);
 */

#ifndef NAN_NO_CHECK

                    FOR_EACH_2D_IMAGE
                        if (TSGSL_isnan(dvp(l)))
                        {
                            REPORT_ERROR("DVP CONTAINS NAN");

                            abort();
                        }

#endif

                    FOR_EACH_2D_IMAGE
                    {
                        omp_set_lock(&mtx[l]);

                        if (TSGSL_isnan(baseLine[l]))
                            baseLine[l] = dvp(l);
                        else
                        {
                            if (dvp(l) > baseLine[l])
                            {
                                RFLOAT offset = dvp(l) - baseLine[l];

                                RFLOAT nf = exp(-offset);

                                wC.row(l) *= nf;
                                wR.row(l) *= nf;
                                wT.row(l) *= nf;

                                /***
                                RFLOAT nf = exp(offset);

                                if (TSGSL_isinf(nf))
                                {
                                    wC.row(l) = vec::Zero(_para.k).transpose();
                                    wR.row(l) = vec::Zero(nR).transpose();
                                    wT.row(l) = vec::Zero(nT).transpose();
                                }
                                else
                                {
                                    wC.row(l) /= nf;
                                    wR.row(l) /= nf;
                                    wT.row(l) /= nf;
                                }
                                ***/

                                baseLine[l] += offset;
                            }
                        }

                        RFLOAT w = exp(dvp(l) - baseLine[l]);

                        wC(l, t) += w;

                        wR(l, m) += w;

                        wT(l, n) += w;

                        omp_unset_lock(&mtx[l]);
                    }
                }

                #pragma omp atomic
                _nR += 1;

                #pragma omp critical
                if (_nR > (int)(nR * _para.k / 10))
                {
                    _nR = 0;

                    nPer += 1;

                    ALOG(INFO, "LOGGER_ROUND") << nPer * 10
                                               << "\% Initial Phase of Global Search Performed";
                    BLOG(INFO, "LOGGER_ROUND") << nPer * 10
                                               << "\% Initial Phase of Global Search Performed";
                }
            }

            TSFFTW_free(poolPriRotP);
            TSFFTW_free(poolPriAllP);
        }

        delete[] mtx;
        delete[] baseLine;
        
        // reset weights of particle filter

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {

#ifndef NAN_NO_CHECK

            if ((wC.row(l).sum() == 0) || (TSGSL_isnan(wC.row(l).sum())))
            {
                REPORT_ERROR("WC, NAN DETECTED");

                abort();
            }

            if ((wR.row(l).sum() == 0) || (TSGSL_isnan(wR.row(l).sum())))
            {
                REPORT_ERROR("WR, NAN DETECTED");

                abort();
            }

            if ((wT.row(l).sum() == 0) || (TSGSL_isnan(wT.row(l).sum())))
            {
                REPORT_ERROR("WT, NAN DETECTED");

                abort();
            }

#endif

            for (int iC = 0; iC < _para.k; iC++)
                _par[l].setUC(wC(l, iC), iC);
            for (int iR = 0; iR < nR; iR++)
                _par[l].setUR(wR(l, iR), iR);
            for (int iT = 0; iT < nT; iT++)
                _par[l].setUT(wT(l, iT), iT);

#ifdef OPTIMISER_SAVE_PARTICLES
            if (_ID[l] < N_SAVE_IMG)
            {
                _par[l].sort();

                char filename[FILE_NAME_LENGTH];

                snprintf(filename,
                         sizeof(filename),
                         "C_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_C);
                snprintf(filename,
                         sizeof(filename),
                         "R_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_R);
                snprintf(filename,
                         sizeof(filename),
                         "T_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_T);
                snprintf(filename,
                         sizeof(filename),
                         "D_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_D);
            }
#endif

            _par[l].resample(_para.k, PAR_C);

            _par[l].resample(_para.mLR, PAR_R);
            _par[l].resample(_para.mLT, PAR_T);

            _par[l].calVari(PAR_R);
            _par[l].calVari(PAR_T);

            if (_para.mode == MODE_2D)
            {
                _par[l].setK1(TSGSL_MAX_RFLOAT((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                ? _para.perturbFactorSGlobal
                                                : _para.perturbFactorSLocal))
                                        * MIN_STD_FACTOR * scanMinStdR,
                                          _par[l].k1()));
            }
            else if (_para.mode == MODE_3D)
            {
                _par[l].setK1(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                          ? _para.perturbFactorSGlobal
                                                          : _para.perturbFactorSLocal))
                                                  * MIN_STD_FACTOR * scanMinStdR),
                                          _par[l].k1()));
                _par[l].setK2(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                          ? _para.perturbFactorSGlobal
                                                          : _para.perturbFactorSLocal))
                                                  * MIN_STD_FACTOR * scanMinStdR),
                                          _par[l].k2()));

                _par[l].setK3(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                          ? _para.perturbFactorSGlobal
                                                          : _para.perturbFactorSLocal))
                                                   * MIN_STD_FACTOR * scanMinStdR),
                                          _par[l].k3()));
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

            _par[l].setS0(TSGSL_MAX_RFLOAT(1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                           ? _para.perturbFactorSGlobal
                                           : _para.perturbFactorSLocal)
                                    * MIN_STD_FACTOR * scanMinStdT,
                                      _par[l].s0()));

            _par[l].setS1(TSGSL_MAX_RFLOAT(1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                           ? _para.perturbFactorSGlobal
                                           : _para.perturbFactorSLocal)
                                    * MIN_STD_FACTOR * scanMinStdT,
                                      _par[l].s1()));

#ifdef OPTIMISER_SAVE_PARTICLES
            if (_ID[l] < N_SAVE_IMG)
            {
                _par[l].sort();

                char filename[FILE_NAME_LENGTH];
                snprintf(filename,
                         sizeof(filename),
                         "C_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_C);
                snprintf(filename,
                         sizeof(filename),
                         "R_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_R);
                snprintf(filename,
                         sizeof(filename),
                         "T_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_T);
                snprintf(filename,
                         sizeof(filename),
                         "D_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_D);
            }
#endif

#ifdef OPTIMISER_KEEP_ONLY_ONE_CLASS

            size_t cls;
            _par[l].rand(cls);

            _par[l].setNC(1);
            _par[l].setC(uvec::Constant(1, cls));
            _par[l].setWC(vec::Constant(1, 1));
            _par[l].setUC(vec::Constant(1, 1));
#endif
        }

        ALOG(INFO, "LOGGER_ROUND") << "Initial Phase of Global Search Performed.";
        BLOG(INFO, "LOGGER_ROUND") << "Initial Phase of Global Search Performed.";

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_ROUND") << "Initial Phase of Global Search in Hemisphere A Performed";
        BLOG(INFO, "LOGGER_ROUND") << "Initial Phase of Global Search in Hemisphere B Performed";
#endif

        TSFFTW_free(traP);

        if (_searchType != SEARCH_TYPE_CTF)
            freePreCal(false);
        else
            freePreCal(true);
    }

#ifdef OPTIMISER_PARTICLE_FILTER

    if (_searchType != SEARCH_TYPE_CTF)
        allocPreCal(false, false);
    else
        allocPreCal(false, true);

    _nP.resize(_ID.size(), 0);

    _nF = 0;
    _nI = 0;

    nPer = 0;

    Complex* poolPriRotP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));
    Complex* poolPriAllP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));

    Complex* poolTraP = (Complex*)TSFFTW_malloc(_para.mLT * _nPxl * omp_get_max_threads() * sizeof(Complex));

    RFLOAT* poolCtfP;

    if (_searchType == SEARCH_TYPE_CTF)
        poolCtfP = (RFLOAT*)TSFFTW_malloc(_para.mLD * _nPxl * omp_get_max_threads() * sizeof(RFLOAT));

    #pragma omp parallel for schedule(dynamic)
    FOR_EACH_2D_IMAGE
    {
        RFLOAT baseLine = GSL_NAN;

        Complex* priRotP = poolPriRotP + _nPxl * omp_get_thread_num();
        Complex* priAllP = poolPriAllP + _nPxl * omp_get_thread_num();

        int nPhaseWithNoVariDecrease = 0;

#ifdef OPTIMISER_COMPRESS_CRITERIA
        RFLOAT topCmp = 0;
#else
        RFLOAT k1 = 1;
        RFLOAT k2 = 1;
        RFLOAT k3 = 1;
        RFLOAT tVariS0 = 5 * _para.transS;
        RFLOAT tVariS1 = 5 * _para.transS;
        RFLOAT dVari = 5 * _para.ctfRefineS;
#endif

        for (int phase = (_searchType == SEARCH_TYPE_GLOBAL) ? 1 : 0; phase < MAX_N_PHASE_PER_ITER; phase++)
        {
            if (phase == 0)
            {
                /***
                _par[l].resample(_para.mLR, PAR_R);
                _par[l].resample(_para.mLT, PAR_T);
                ***/

                _par[l].perturb(_para.perturbFactorL, PAR_R);
                _par[l].perturb(_para.perturbFactorL, PAR_T);

                /***
                if (_model.r() > _model.rPrev())
                {
                    _par[l].perturb(_para.perturbFactorL, PAR_R);
                    _par[l].perturb(_para.perturbFactorL, PAR_T);
                }
                else
                {
                    _par[l].perturb((_searchType == SEARCH_TYPE_GLOBAL)
                                  ? _para.perturbFactorSGlobal
                                  : _para.perturbFactorSLocal,
                                    PAR_R);
                    _par[l].perturb((_searchType == SEARCH_TYPE_GLOBAL)
                                  ? _para.perturbFactorSGlobal
                                  : _para.perturbFactorSLocal,
                                    PAR_T);
                }
                ***/

                if (_searchType == SEARCH_TYPE_CTF)
                    _par[l].initD(_para.mLD, _para.ctfRefineS);
            }
            else
            {
                _par[l].perturb((_searchType == SEARCH_TYPE_GLOBAL)
                              ? _para.perturbFactorSGlobal
                              : _para.perturbFactorSLocal,
                                PAR_R);
                _par[l].perturb((_searchType == SEARCH_TYPE_GLOBAL)
                              ? _para.perturbFactorSGlobal
                              : _para.perturbFactorSLocal,
                                PAR_T);

                if (_searchType == SEARCH_TYPE_CTF)
                    _par[l].perturb(_para.perturbFactorSCTF, PAR_D);
            }

#ifdef OPTIMISER_KEEP_ONLY_ONE_CLASS
            vec wC = vec::Zero(1);
#else
            vec wC = vec::Zero(_para.k);
#endif

            vec wR = vec::Zero(_para.mLR);
            vec wT = vec::Zero(_para.mLT);
            vec wD = vec::Zero(_para.mLD);

            size_t c;
            mat22 rot2D;
            mat33 rot3D;
            RFLOAT d;
            vec2 t;

            FOR_EACH_C(_par[l])
            {
                _par[l].c(c, iC);

                Complex* traP = poolTraP + _par[l].nT() * _nPxl * omp_get_thread_num();

                FOR_EACH_T(_par[l])
                {
                    _par[l].t(t, iT);

                    translate(traP + iT * _nPxl,
                              t(0),
                              t(1),
                              _para.size,
                              _para.size,
                              _iCol,
                              _iRow,
                              _nPxl);
                }

                RFLOAT* ctfP;

                if (_searchType == SEARCH_TYPE_CTF)
                {
                    ctfP = poolCtfP + _par[l].nD() * _nPxl * omp_get_thread_num();

                    FOR_EACH_D(_par[l])
                    {
                        _par[l].d(d, iD);

                        for (int i = 0; i < _nPxl; i++)
                        {
                            RFLOAT ki = _K1[l]
                                      * _defocusP[l * _nPxl + i]
                                      * d
                                      * TSGSL_pow_2(_frequency[i])
                                      + _K2[l]
                                      * TSGSL_pow_4(_frequency[i]);

                            /***
                            RFLOAT ki = K1 * defocus[i] * df * TSGSL_pow_2(frequency[i])
                                      + K2 * TSGSL_pow_4(frequency[i]);
                            ***/

                            ctfP[_nPxl * iD + i] = -w1 * sin(ki) + w2 * cos(ki);
                        }
                    }
                }

                FOR_EACH_R(_par[l])
                {
                    if (_para.mode == MODE_2D)
                    {
                        _par[l].rot(rot2D, iR);
                    }
                    else if (_para.mode == MODE_3D)
                    {
                        _par[l].rot(rot3D, iR);
                    }
                    else
                    {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }

                    if (_para.mode == MODE_2D)
                    {
                        _model.proj(c).project(priRotP,
                                               rot2D,
                                               _iCol,
                                               _iRow,
                                               _nPxl);
                    }
                    else if (_para.mode == MODE_3D)
                    {
                        _model.proj(c).project(priRotP,
                                               rot3D,
                                               _iCol,
                                               _iRow,
                                               _nPxl);
                    }
                    else
                    {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }

                    FOR_EACH_T(_par[l])
                    {
                        for (int i = 0; i < _nPxl; i++)
                            priAllP[i] = traP[_nPxl * iT + i] * priRotP[i];

                        FOR_EACH_D(_par[l])
                        {
                            _par[l].d(d, iD);

                            RFLOAT w;

                            /*
                             *RFLOAT wOrig;
                             */
                            if (_searchType != SEARCH_TYPE_CTF)
                            {
                                w = logDataVSPrior_m_huabin(_datP + l * _nPxl,
                                                   priAllP,
                                                   _ctfP + l * _nPxl,
                                                   _sigRcpP + l * _nPxl,
                                                   _nPxl);
                                /*
                                 *wOrig = logDataVSPrior(_datP + l * _nPxl,
                                 *                   priAllP,
                                 *                   _ctfP + l * _nPxl,
                                 *                   _sigRcpP + l * _nPxl,
                                 *                   _nPxl);
                                 */
                            }
                            else
                            {
                                w = logDataVSPrior_m_huabin(_datP + l * _nPxl,
                                                   priAllP,
                                                   ctfP + iD * _nPxl,
                                                   _sigRcpP + l * _nPxl,
                                                   _nPxl);
                                /*
                                 *wOrig = logDataVSPrior(_datP + l * _nPxl,
                                 *                   priAllP,
                                 *                   ctfP + iD * _nPxl,
                                 *                   _sigRcpP + l * _nPxl,
                                 *                   _nPxl);
                                 */
                            }

                            /*
                             *compareWInmHuabin(wOrig, w, _commRank, omp_get_thread_num(), _nPxl);
                             */

                            //if (TSGSL_isnan(baseLine)) baseLine = w;
                            baseLine = TSGSL_isnan(baseLine) ? w : baseLine;

                            if (w > baseLine)
                            {
                                RFLOAT nf = exp(baseLine - w);

                                wC *= nf;
                                wR *= nf;
                                wT *= nf;
                                wD *= nf;

                                baseLine = w;
                            }

                            w = exp(w - baseLine);

                            wC(iC) += w;
                            wR(iR) += w;
                            wT(iT) += w;
                            wD(iD) += w;

                            if (_ID[l] == 2)
                            {
                                CLOG(INFO, "LOGGER_SYS") << "Add, wC(0) = " << wC(0);
                                CLOG(INFO, "LOGGER_SYS") << "Add, wC.sum() = " << wC.sum();
                            }
                        }
                    }
                }
            }

            /***
#ifndef NAN_NO_CHECK

            if ((wC.sum() < 1) || (TSGSL_isnan(wC.sum())))
            {
                CLOG(INFO, "LOGGER_SYS") << "wC.sum() = " << wC.sum();

                REPORT_ERROR("WC ERROR");

                abort();
            }

            if ((wR.sum() < 1) || (TSGSL_isnan(wR.sum())))
            {
                CLOG(INFO, "LOGGER_SYS") << "wR.sum() = " << wR.sum();

                REPORT_ERROR("WC ERROR");

                abort();
            }

            for (int i = 0; i < _para.mLR; i++)
            {
                if ((wR(i) < 0) || (TSGSL_isnan(wR(i))))
                {
                    CLOG(WARNING, "LOGGER_SYS") << "wR " << i << ": " << wR(i);

                    REPORT_ERROR("WRONG!");

                    abort();
                }
            }

            if ((wT.sum() < 1) || (TSGSL_isnan(wT.sum())))
            {
                CLOG(INFO, "LOGGER_SYS") << "wT.sum() = " << wR.sum();

                REPORT_ERROR("WT ERROR");

                abort();
            }

            if ((wD.sum() < 1) || (TSGSL_isnan(wD.sum())))
            {
                CLOG(INFO, "LOGGER_SYS") << "wD.sum() = " << wR.sum();

                REPORT_ERROR("WD ERROR");

                abort();
            }
#endif
            ***/


#ifdef OPTIMISER_KEEP_ONLY_ONE_CLASS
            _par[l].setUC(wC(0), 0);
#else
            for (int iC = 0; iC < _para.k; iC++)
                _par[l].setUC(wC(iC), iC);
#endif

            for (int iR = 0; iR < _para.mLR; iR++)
                _par[l].setUR(wR(iR), iR);
            for (int iT = 0; iT < _para.mLT; iT++)
                _par[l].setUT(wT(iT), iT);

            if (_searchType == SEARCH_TYPE_CTF)
                for (int iD = 0; iD < _para.mLD; iD++)
                    _par[l].setUD(wD(iD), iD);

#ifdef OPTIMISER_SAVE_PARTICLES
            if (_ID[l] < N_SAVE_IMG)
            {
                _par[l].sort();

                char filename[FILE_NAME_LENGTH];

                snprintf(filename,
                         sizeof(filename),
                         "C_Particle_%04d_Round_%03d_%03d.par",
                         _ID[l],
                         _iter,
                         phase);
                save(filename, _par[l], PAR_C);
                snprintf(filename,
                         sizeof(filename),
                         "R_Particle_%04d_Round_%03d_%03d.par",
                         _ID[l],
                         _iter,
                         phase);
                save(filename, _par[l], PAR_R);
                snprintf(filename,
                         sizeof(filename),
                         "T_Particle_%04d_Round_%03d_%03d.par",
                         _ID[l],
                         _iter,
                         phase);
                save(filename, _par[l], PAR_T);
                snprintf(filename,
                         sizeof(filename),
                         "D_Particle_%04d_Round_%03d_%03d.par",
                         _ID[l],
                         _iter,
                         phase);
                save(filename, _par[l], PAR_D);
            }
#endif

#ifndef OPTIMISER_KEEP_ONLY_ONE_CLASS
            _par[l].resample(_para.k, PAR_C);
#endif

            _par[l].calVari(PAR_R);
            _par[l].calVari(PAR_T);

            _par[l].resample(_para.mLR, PAR_R);
            _par[l].resample(_para.mLT, PAR_T);

            if (_searchType == SEARCH_TYPE_CTF)
            {
                _par[l].calVari(PAR_D);
                _par[l].resample(_para.mLD, PAR_D);
            }

            /***
            RFLOAT k1 = _par[l].k1();
            RFLOAT s0 = _par[l].s0();
            RFLOAT s1 = _par[l].s1();

            _par[l].resample(_para.mLR, PAR_R);
            _par[l].resample(_para.mLT, PAR_T);

            _par[l].calVari(PAR_R);
            _par[l].calVari(PAR_T);

            _par[l].setK1(TSGSL_MAX_RFLOAT(k1 * gsl_pow_2(MIN_STD_FACTOR
                                                   * pow(_par[l].nR(), -1.0 / 3)),
                                      _par[l].k1()));

            _par[l].setS0(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * s0 / sqrt(_par[l].nT()), _par[l].s0()));

            _par[l].setS1(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * s1 / sqrt(_par[l].nT()), _par[l].s1()));
            ***/

            if (phase >= ((_searchType == SEARCH_TYPE_GLOBAL)
                        ? MIN_N_PHASE_PER_ITER_GLOBAL
                        : MIN_N_PHASE_PER_ITER_LOCAL))
            {
                RFLOAT k1Cur;
                RFLOAT k2Cur;
                RFLOAT k3Cur;
                RFLOAT tVariS0Cur;
                RFLOAT tVariS1Cur;
                RFLOAT dVariCur;

                _par[l].vari(k1Cur, k2Cur, k3Cur, tVariS0Cur, tVariS1Cur, dVariCur);

                if (_para.mode == MODE_2D)
                {
                    if ((k1Cur < k1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (tVariS0Cur < tVariS0 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (tVariS1Cur < tVariS1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (dVariCur < dVari * PARTICLE_FILTER_DECREASE_FACTOR))
                    {
                        // there is still room for searching
                        nPhaseWithNoVariDecrease = 0;
                    }
                    else
                        nPhaseWithNoVariDecrease += 1;
                }
                else if (_para.mode == MODE_3D)
                {
                    if ((k1Cur < k1 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                        (k2Cur < k2 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                        (k3Cur < k3 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                        (tVariS0Cur < tVariS0 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (tVariS1Cur < tVariS1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (dVariCur < dVari * PARTICLE_FILTER_DECREASE_FACTOR))
                    {
                        // there is still room for searching
                        nPhaseWithNoVariDecrease = 0;
                    }
                    else
                        nPhaseWithNoVariDecrease += 1;
                }
                else
                {
                    REPORT_ERROR("EXISTENT MODE");

                    abort();
                }

#ifdef OPTIMISER_COMPRESS_CRITERIA
                topCmp = _par[l].compress();
#else
                // make tVariS0, tVariS1, rVari the smallest variance ever got
                if (k1Cur < k1) k1 = k1Cur;
                if (k2Cur < k2) k2 = k2Cur;
                if (k3Cur < k3) k3 = k3Cur;
                if (tVariS0Cur < tVariS0) tVariS0 = tVariS0Cur;
                if (tVariS1Cur < tVariS1) tVariS1 = tVariS1Cur;
                if (dVariCur < dVari) dVari = dVariCur;
#endif

                // break if in a few continuous searching, there is no improvement
                if (nPhaseWithNoVariDecrease == N_PHASE_WITH_NO_VARI_DECREASE)
                {
                    _nP[l] = phase;

                    #pragma omp atomic
                    _nF += phase;

                    #pragma omp atomic
                    _nI += 1;

                    break;
                }
            }
        }

        #pragma omp critical
        if (_nI > (int)(_ID.size() / 10))
        {
            _nI = 0;

            nPer += 1;

            ALOG(INFO, "LOGGER_ROUND") << nPer * 10 << "\% Expectation Performed";
            BLOG(INFO, "LOGGER_ROUND") << nPer * 10 << "\% Expectation Performed";
        }

#ifdef OPTIMISER_SAVE_PARTICLES
        if (_ID[l] < N_SAVE_IMG)
        {
            char filename[FILE_NAME_LENGTH];

            snprintf(filename,
                     sizeof(filename),
                     "C_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, _par[l], PAR_C);
            snprintf(filename,
                     sizeof(filename),
                     "R_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, _par[l], PAR_R);
            snprintf(filename,
                     sizeof(filename),
                     "T_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, _par[l], PAR_T);
            snprintf(filename,
                     sizeof(filename),
                     "D_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, _par[l], PAR_D);
        }
#endif
    }

    TSFFTW_free(poolPriRotP);
    TSFFTW_free(poolPriAllP);

    TSFFTW_free(poolTraP);

    if (_searchType == SEARCH_TYPE_CTF)
        TSFFTW_free(poolCtfP);

    ALOG(INFO, "LOGGER_ROUND") << "Freeing Space for Pre-calcuation in Expectation";
    BLOG(INFO, "LOGGER_ROUND") << "Freeing Space for Pre-calcuation in Expectation";

    if (_searchType != SEARCH_TYPE_CTF)
        freePreCal(false);
    else
        freePreCal(true);

#endif // OPTIMISER_PARTICLE_FILTER

    freePreCalIdx();
}

void Optimiser::maximization()
{
#ifdef OPTIMISER_NORM_CORRECTION
    MLOG(INFO, "LOGGER_ROUND") << "Normalisation Noise";

    normCorrection();
#endif

#ifdef OPTIMISER_REFRESH_SIGMA
    ALOG(INFO, "LOGGER_ROUND") << "Generate Sigma for the Next Iteration";
    BLOG(INFO, "LOGGER_ROUND") << "Generate Sigma for the Next Iteration";

    if (_para.k == 1)
    {
#ifdef OPTIMISER_SIGMA_MASK
        allReduceSigma(true, _para.groupSig);
#else
        // refinement, focus on avoiding over-refinement
        allReduceSigma(false, _para.groupSig);
#endif
    }
    else
    {
        // classification, focus on searching for difference
        allReduceSigma(true, _para.groupSig);
    }
#endif

#ifdef OPTIMISER_CORRECT_SCALE
    if ((_searchType == SEARCH_TYPE_GLOBAL) &&
        (_para.groupScl) &&
        (_iter != 0))
    {
        ALOG(INFO, "LOGGER_ROUND") << "Re-balancing Intensity Scale for Each Group";
        BLOG(INFO, "LOGGER_ROUND") << "Re-balancing Intensity Scale for Each Group";

        correctScale(false, true);
    }
#endif

    if (!_para.skipR)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Reconstruct Reference";
        BLOG(INFO, "LOGGER_ROUND") << "Reconstruct Reference";

        reconstructRef(true, true, true, false, false);
    }
}

void Optimiser::run()
{
    MLOG(INFO, "LOGGER_ROUND") << "Initialising Optimiser";

    init();

    MLOG(INFO, "LOGGER_ROUND") << "Saving Some Data";
    
#ifdef OPTIMISER_SAVE_IMAGES

    saveImages();

#endif
    
    /***
    saveCTFs();
    saveBinImages();
    saveLowPassImages();
    ***/

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_SAVE_SIGMA
    saveSig();
#endif

    MLOG(INFO, "LOGGER_ROUND") << "Entering Iteration";
    for (_iter = 0; _iter < _para.iterMax; _iter++)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter;

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type ( Round "
                                       << _iter
                                       << " ) : Global Search";
        }
        else if (_searchType == SEARCH_TYPE_LOCAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type ( Round "
                                       << _iter
                                       << " ) : Local Search";
        }
        else if (_searchType == SEARCH_TYPE_CTF)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type ( Round "
                                       << _iter
                                       << " ) : CTF Search";
        }
        else
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type ( Round "
                                       << _iter
                                       << " ) : Stop Search";

            MLOG(INFO, "LOGGER_ROUND") << "Exitting Searching";

            break;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if ((_iter == 0) || (!_para.skipE))
        {
            MLOG(INFO, "LOGGER_ROUND") << "Performing Expectation";

            expectation();

            MLOG(INFO, "LOGGER_ROUND") << "Waiting for All Processes Finishing Expectation";
            ILOG(INFO, "LOGGER_ROUND") << "Expectation Accomplished, with Filtering "
                                       << _nF
                                       << " Times over "
                                       << _ID.size()
                                       << " Images";

            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "All Processes Finishing Expectation";
        }

        MLOG(INFO, "LOGGER_ROUND") << "Determining Percentage of Images Belonging to Each Class";

        refreshClassDistr();

        for (int t = 0; t < _para.k; t++)
            MLOG(INFO, "LOGGER_ROUND") << _cDistr(t) * 100
                                       << "\% Percentage of Images Belonging to Class "
                                       << t;

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Percentage of Images Belonging to Each Class Determined";
#endif

#ifdef OPTIMISER_SAVE_BEST_PROJECTIONS

        MLOG(INFO, "LOGGER_ROUND") << "Saving Best Projections";
        saveBestProjections();

#endif

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Best Projections Saved";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Saving Database";
 
        saveDatabase();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Database Saved";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Calculating Variance of Rotation and Translation";

        refreshVariance();

        MLOG(INFO, "LOGGER_ROUND") << "Rotation Variance ( Round "
                                   << _iter
                                   << " ) : "
                                   << _model.rVari();

        MLOG(INFO, "LOGGER_ROUND") << "Translation Variance ( Round "
                                   << _iter
                                   << " ) : "
                                   << _model.tVariS0()
                                   << ", "
                                   << _model.tVariS1();

        MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Rotation Variance : "
                                   << _model.stdRVari();

        MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Translation Variance : "
                                   << _model.stdTVariS0()
                                   << ", "
                                   << _model.stdTVariS1();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Variance of Rotation and Translation Calculated";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Calculating Changes of Rotation Between Iterations";
        refreshRotationChange();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Changes of Rotation Between Iterations Calculated";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Average Rotation Change : " << _model.rChange();
        MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Rotation Change : "
                                   << _model.stdRChange();

        if (!_para.skipM)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Performing Maximization";

            maximization();
        }

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

        if (_searchType != SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Re-Centring Images";

            reCentreImg();
        }
        else
        {
            MLOG(INFO, "LOGGER_ROUND") << "Re-Loading Images from Original Images";

            _img.clear();
            FOR_EACH_2D_IMAGE
                _img.push_back(_imgOri[l].copyImage());
        }

#else

        MLOG(INFO, "LOGGER_ROUND") << "Re-Loading Images from Original Images";

        _img.clear();
        FOR_EACH_2D_IMAGE
            _img.push_back(_imgOri[l].copyImage());

#endif

#ifdef OPTIMISER_MASK_IMG
        MLOG(INFO, "LOGGER_ROUND") << "Re-Masking Images";
        reMaskImg();
#endif

#ifdef OPTIMISER_SAVE_SIGMA
        MLOG(INFO, "LOGGER_ROUND") << "Saving Sigma";
        saveSig();
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        MLOG(INFO, "LOGGER_ROUND") << "Maximization Performed";

        MLOG(INFO, "LOGGER_ROUND") << "Calculating SNR(s)";
        _model.refreshSNR();

#ifdef OPTIMISER_SAVE_FSC
        MLOG(INFO, "LOGGER_ROUND") << "Saving FSC(s)";
        saveFSC();
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Saving Class Information";
        saveClassInfo();

        MLOG(INFO, "LOGGER_ROUND") << "Current Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1,
                                                   _para.size,
                                                   _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Recording Current Resolution";

        _resReport = _model.resolutionP(_para.thresReportFSC, false);

        MLOG(INFO, "LOGGER_ROUND") << "Current Resolution for Report, ( Round "
                                   << _iter
                                   << " ) : "
                                   << _resReport
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_resReport, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        _model.setRes(_resReport);

        _resCutoff = _model.resolutionP(_para.thresCutoffFSC, false);

        MLOG(INFO, "LOGGER_ROUND") << "Current Resolution for Cutoff, ( Round "
                                   << _iter
                                   << " ) : "
                                   << _resCutoff
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_resCutoff, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Updating Cutoff Frequency in Model";

        _model.updateR(_para.thresCutoffFSC);

#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE
        MLOG(INFO, "LOGGER_ROUND") << "Increasing Cutoff Frequency or Not: "
                                   << _model.increaseR()
                                   << ", as the Rotation Change is "
                                   << _model.rChange()
                                   << " and the Previous Rotation Change is "
                                   << _model.rChangePrev();
#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI
        MLOG(INFO, "LOGGER_ROUND") << "Increasing Cutoff Frequency or Not: "
                                   << _model.increaseR()
                                   << ", as the Translation Variance is "
                                   << _model.tVariS0()
                                   << ", "
                                   << _model.tVariS1()
                                   << ", and the Previous Translation Variance is "
                                   << _model.tVariS0Prev()
                                   << ", "
                                   << _model.tVariS1Prev();
#endif

        if (_model.r() > _model.rT())
        {
            MLOG(INFO, "LOGGER_ROUND") << "Resetting Parameters Determining Increase Frequency";

            _model.resetTVari();
            _model.resetRChange();
            _model.setNRChangeNoDecrease(0);
            _model.setIncreaseR(false);

            MLOG(INFO, "LOGGER_ROUND") << "Recording Current Highest Frequency";

            _model.setRT(_model.r());
        }

#ifdef OPTIMISER_SOLVENT_FLATTEN

        if ((_para.globalMask) || (_searchType != SEARCH_TYPE_GLOBAL))
        {
            MLOG(INFO, "LOGGER_ROUND") << "Solvent Flattening";

            solventFlatten(_para.performMask);
        }

#endif

#ifdef OPTIMISER_SAVE_SOLVENT_FLATTENED_REFERENCE

        MLOG(INFO, "LOGGER_ROUND") << "Saving Solvent Flattened Reference(s)";

        saveReference();

#endif

        MLOG(INFO, "LOGGER_ROUND") << "Determining the Search Type of the Next Iteration";
        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            _searchType = _model.searchType();

            if (_para.performMask &&
                _para.autoMask &&
                (_searchType == SEARCH_TYPE_LOCAL))
            {
                MLOG(INFO, "LOGGER_ROUND") << "A Mask Should be Generated";

                _genMask = true;
            }
        }
        else
            _searchType = _model.searchType();

        MLOG(INFO, "LOGGER_ROUND") << "Recording Top Resolution";
        if (_resReport > _model.resT())
            _model.setResT(_resReport);

        MLOG(INFO, "LOGGER_ROUND") << "Updating Cutoff Frequency";
        _r = _model.r();

        MLOG(INFO, "LOGGER_ROUND") << "New Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Updating Frequency Boundary of Reconstructor";
        _model.updateRU();

        NT_MASTER
        {
            ALOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors";
            BLOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors";

            _model.refreshProj();

            /***
            if (_searchType == SEARCH_TYPE_CTF)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Resetting to Nyquist Limit in CTF Refine";
                BLOG(INFO, "LOGGER_ROUND") << "Resetting to Nyquist Limit in CTF Refine";

                _model.setRU(maxR());
            }
            ***/

            ALOG(INFO, "LOGGER_ROUND") << "Resetting Reconstructors";
            BLOG(INFO, "LOGGER_ROUND") << "Resetting Reconstructors";

            _model.resetReco(_para.thresReportFSC);
        }
    }

    MLOG(INFO, "LOGGER_ROUND") << "Preparing to Reconstruct Reference(s) at Nyquist";

    MLOG(INFO, "LOGGER_ROUND") << "Resetting to Nyquist Limit";
    _model.setMaxRU();

    MLOG(INFO, "LOGGER_ROUND") << "Refreshing Reconstructors";
    NT_MASTER
    {
        _model.resetReco(_para.thresReportFSC);
    }

    MLOG(INFO, "LOGGER_ROUND") << "Reconstructing References(s) at Nyquist";
    reconstructRef(true, false, true, false, true);

    MLOG(INFO, "LOGGER_ROUND") << "Saving Final FSC(s)";
    saveFSC(true);
}

void Optimiser::clear()
{
    _img.clear();
    _par.clear();
    _ctf.clear();
}

void Optimiser::bCastNPar()
{
    _nPar = _db.nParticle();
}

void Optimiser::allReduceN()
{
    IF_MASTER return;

    _N = _db.nParticleRank();

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_N, 1, MPI_INT, MPI_SUM, _hemi);

    MPI_Barrier(_hemi);
}

int Optimiser::size() const
{
    return _para.size;
}

int Optimiser::maxR() const
{
    return size() / 2 - 1;
}

void Optimiser::bcastGroupInfo()
{
    ALOG(INFO, "LOGGER_INIT") << "Storing GroupID";

    _groupID.clear();

    NT_MASTER
        FOR_EACH_2D_IMAGE
            _groupID.push_back(_db.groupID(_ID[l]));

    MLOG(INFO, "LOGGER_INIT") << "Getting Number of Groups from Database";

    _nGroup = _db.nGroup();

    MLOG(INFO, "LOGGER_INIT") << "Number of Group: " << _nGroup;

    MLOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Sigma";
    NT_MASTER
    {
        _sig.resize(_nGroup, maxR() + 1);
        _sigRcp.resize(_nGroup, maxR());
    }

    MLOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Intensity Scale";
    _scale.resize(_nGroup);
}

void Optimiser::initRef()
{
    FFT fft;

    if (strcmp(_para.initModel, "") != 0)
    {
        MLOG(INFO, "LOGGER_INIT") << "Read Initial Model from Hard-disk";

        Volume ref;

        ImageFile imf(_para.initModel, "rb");
        imf.readMetaData();
        imf.readVolume(ref);

        if (_para.mode == MODE_2D)
        {
            if ((ref.nColRL() != _para.size) ||
                (ref.nRowRL() != _para.size) ||
                (ref.nSlcRL() != 1))
            {
                CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Appending Reference"
                                          << ": size = " << _para.size
                                          << ", nCol = " << ref.nColRL()
                                          << ", nRow = " << ref.nRowRL()
                                          << ", nSlc = " << ref.nSlcRL();

                abort();
            }
        }
        else if (_para.mode == MODE_3D)
        {
            if ((ref.nColRL() != _para.size) ||
                (ref.nRowRL() != _para.size) ||
                (ref.nSlcRL() != _para.size))
            {
                CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Appending Reference"
                                          << ": size = " << _para.size
                                          << ", nCol = " << ref.nColRL()
                                          << ", nRow = " << ref.nRowRL()
                                          << ", nSlc = " << ref.nSlcRL();
 
                abort();
            }
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    
#ifdef OPTIMISER_INIT_REF_REMOVE_NEG
        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(ref)
            if (ref(i) < 0) ref(i) = 0;
#endif

        _model.clearRef();

        for (int t = 0; t < _para.k; t++)
        {
            if (_para.mode == MODE_2D)
            {
                _model.appendRef(ref.copyVolume());
            }
            else if (_para.mode == MODE_3D)
            {
                _model.appendRef(ref.copyVolume());
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

            fft.fwMT(_model.ref(t));
            _model.ref(t).clearRL();
        }
    }
    else
    {
        MLOG(INFO, "LOGGER_INIT") << "Initial Model is not Provided";

        if (_para.mode == MODE_2D)
        {
            Image ref(_para.size,
                      _para.size,
                      RL_SPACE);

            /***
            IMAGE_FOR_EACH_PIXEL_RL(ref)
            {
                if (NORM(i, j) < _para.maskRadius / _para.pixelSize)
                    ref.setRL(1, i, j);
                else
                    ref.setRL(0, i, j);
            }
            ***/

            // softMask(ref, _para.maskRadius / _para.pixelSize, EDGE_WIDTH_RL);

            SET_0_RL(ref);

            Volume volRef(_para.size,
                          _para.size,
                          1,
                          RL_SPACE);

            COPY_RL(volRef, ref);

            _model.clearRef();

            for (int t = 0; t < _para.k; t++)
            {
                _model.appendRef(volRef.copyVolume());

                fft.fwMT(_model.ref(t));
                _model.ref(t).clearRL();
            }
        }
        else if (_para.mode == MODE_3D)
        {
            Volume ref(_para.size,
                       _para.size,
                       _para.size,
                       RL_SPACE);

            /***
            VOLUME_FOR_EACH_PIXEL_RL(ref)
            {
                if (NORM_3(i, j, k) < _para.maskRadius / _para.pixelSize)
                    ref.setRL(1, i, j, k);
                else
                    ref.setRL(0, i, j, k);
            }
            ***/

            // softMask(ref, _para.maskRadius / _para.pixelSize, EDGE_WIDTH_RL);

            SET_0_RL(ref);

            _model.clearRef();

            for (int t = 0; t < _para.k; t++)
            {
                _model.appendRef(ref.copyVolume());

                fft.fwMT(_model.ref(t));
                _model.ref(t).clearRL();
            }

        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }
}

void Optimiser::initMask()
{
    ImageFile imf(_para.mask, "rb");
    imf.readMetaData();

    imf.readVolume(_mask);
}

void Optimiser::initID()
{
    _ID.clear();

    for (int i = _db.start(); i <= _db.end(); i++)
        _ID.push_back(i);
}

void Optimiser::initImg()
{
    ALOG(INFO, "LOGGER_INIT") << "Reading Images from Disk";
    BLOG(INFO, "LOGGER_INIT") << "Reading Images from Disk";

    _img.clear();
    _img.resize(_ID.size());

    string imgName;

    int nPer = 0;
    int nImg = 0;

    FOR_EACH_2D_IMAGE
    {
        nImg += 1;

        if (nImg >= (int)_ID.size() / 10)
        {
            nPer += 1;

            ALOG(INFO, "LOGGER_SYS") << nPer * 10 << "\% Percentage of Images Read";
            BLOG(INFO, "LOGGER_SYS") << nPer * 10 << "\% Percentage of Images Read";

            nImg = 0;
        }

        imgName = _db.path(_ID[l]);

        /***
        ILOG(INFO, "LOGGER_SYS") << "Path of Image: "
                                 << imgName;
                                 ***/

        if (imgName.find('@') == string::npos)
        {
            //ImageFile imf(imgName.c_str(), "rb");
            ImageFile imf((string(_para.parPrefix) + imgName).c_str(), "rb");
            imf.readMetaData();
            imf.readImage(_img[l]);
        }
        else
        {
            int nSlc = atoi(imgName.substr(0, imgName.find('@')).c_str()) - 1;
            string filename = string(_para.parPrefix) + imgName.substr(imgName.find('@') + 1);

            ImageFile imf(filename.c_str(), "rb");
            imf.readMetaData();
            imf.readImage(_img[l], nSlc);
        }

        if ((_img[l].nColRL() != _para.size) ||
            (_img[l].nRowRL() != _para.size))
        {
            CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of 2D Images, "
                                      << "Should be "
                                      << _para.size
                                      << " x "
                                      << _para.size
                                      << ", but "
                                      << _img[l].nColRL()
                                      << " x "
                                      << _img[l].nRowRL()
                                      << " Input.";

            abort();
        }
    }

#ifdef VERBOSE_LEVEL_1
    ILOG(INFO, "LOGGER_INIT") << "Images Read from Disk";
#endif

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Images Read from Disk";
    BLOG(INFO, "LOGGER_INIT") << "Images Read from Disk";
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    ALOG(INFO, "LOGGER_INIT") << "Setting 0 to Offset between Images and Original Images";
    BLOG(INFO, "LOGGER_INIT") << "Setting 0 to Offset between Images and Original Images";

    _offset = vector<vec2>(_img.size(), vec2(0, 0));

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Offset between Images and Original Images are Set to 0";
    BLOG(INFO, "LOGGER_INIT") << "Offset between Images and Original Images are Set to 0";
#endif
#endif

    ALOG(INFO, "LOGGER_INIT") << "Substructing Mean of Noise, Making the Noise Have Zero Mean";
    BLOG(INFO, "LOGGER_INIT") << "Substructing Mean of Noise, Making the Noise Have Zero Mean";

    substractBgImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Mean of Noise Substructed";
    BLOG(INFO, "LOGGER_INIT") << "Mean of Noise Substructed";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Performing Statistics of 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Performing Statistics of 2D Images";

    statImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Statistics Performed of 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Statistics Performed of 2D Images";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images Before Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images Before Normalising";

    displayStatImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images Bofore Normalising Displayed";
    BLOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images Bofore Normalising Displayed";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Masking on 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Masking on 2D Images";

    maskImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "2D Images Masked";
    BLOG(INFO, "LOGGER_INIT") << "2D Images Masked";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Normalising 2D Images, Making the Noise Have Standard Deviation of 1";
    BLOG(INFO, "LOGGER_INIT") << "Normalising 2D Images, Making the Noise Have Standard Deviation of 1";

    normaliseImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "2D Images Normalised";
    BLOG(INFO, "LOGGER_INIT") << "2D Images Normalised";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";

    displayStatImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images After Normalising Displayed";
    BLOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images After Normalising Displayed";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform on 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform on 2D Images";

    fwImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Fourier Transform on 2D Images Performed";
    BLOG(INFO, "LOGGER_INIT") << "Fourier Transform on 2D Images Performed";
#endif
}

void Optimiser::statImg()
{
    _mean = 0;

    _stdN = 0;
    _stdD = 0;
    _stdS = 0;
    
    _stdStdN = 0;

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        #pragma omp atomic
        _mean += regionMean(_img[l],
                            _para.maskRadius / _para.pixelSize,
                            0);
#else
        #pragma omp atomic
        _mean += regionMean(_img[l],
                            _para.size / 2,
                            0);
#endif

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        #pragma omp atomic
        _stdN += bgStddev(0,
                          _img[l],
                          _para.maskRadius / _para.pixelSize);
#else
        #pragma omp atomic
        _stdN += bgStddev(0,
                          _img[l],
                          _para.size / 2);
#endif

        #pragma omp atomic
        _stdD += stddev(0, _img[l]);

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        #pragma omp atomic
        _stdStdN += gsl_pow_2(bgStddev(0,
                                       _img[l],
                                       _para.maskRadius / _para.pixelSize));
#else
        #pragma omp atomic
        _stdStdN += gsl_pow_2(bgStddev(0,
                                       _img[l],
                                       _para.size / 2));
#endif
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_mean, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdN, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdD, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdStdN, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Barrier(_hemi);

    _mean /= _N;

    _stdN /= _N;
    _stdD /= _N;

    _stdStdN /= _N;

    _stdS = _stdD - _stdN;

    _stdStdN = sqrt(_stdStdN - TSGSL_pow_2(_stdN));
}

void Optimiser::displayStatImg()
{
    ALOG(INFO, "LOGGER_INIT") << "Mean of Centre : " << _mean;

    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Noise  : " << _stdN;
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Data   : " << _stdD;
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Signal : " << _stdS;

    ALOG(INFO, "LOGGER_INIT") << "Standard Devation of Standard Deviation of Noise : "
                              << _stdStdN;

    BLOG(INFO, "LOGGER_INIT") << "Mean of Centre : " << _mean;

    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Noise  : " << _stdN;
    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Data   : " << _stdD;
    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Signal : " << _stdS;

    BLOG(INFO, "LOGGER_INIT") << "Standard Devation of Standard Deviation of Noise : "
                              << _stdStdN;
}

void Optimiser::substractBgImg()
{
    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        RFLOAT bgMean, bgStddev;

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        bgMeanStddev(bgMean,
                     bgStddev,
                     _img[l],
                     _para.maskRadius / _para.pixelSize);
#else
        bgMeanStddev(bgMean,
                     bgStddev,
                     _img[l],
                     _para.size / 2);
#endif

        FOR_EACH_PIXEL_RL(_img[l])
        {
            _img[l](i) -= bgMean;
            _img[l](i) /= bgStddev;
        }

        /***
        RFLOAT bg = background(_img[l],
                               _para.maskRadius / _para.pixelSize,
                               EDGE_WIDTH_RL);

        FOR_EACH_PIXEL_RL(_img[l])
            _img[l](i) -= bg;
        ***/
    }
}

void Optimiser::maskImg()
{
    _imgOri.clear();

    FOR_EACH_2D_IMAGE
        _imgOri.push_back(_img[l].copyImage());

#ifdef OPTIMISER_MASK_IMG
    if (_para.zeroMask)
    {
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
            softMask(_img[l],
                     _img[l],
                     _para.maskRadius / _para.pixelSize,
                     EDGE_WIDTH_RL,
                     0);
    }
    else
    {
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
            softMask(_img[l],
                     _img[l],
                     _para.maskRadius / _para.pixelSize,
                     EDGE_WIDTH_RL,
                     0,
                     _stdN);
    }
#endif
}

void Optimiser::normaliseImg()
{
    RFLOAT scale = 1.0 / _stdN;

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        SCALE_RL(_img[l], scale);
        SCALE_RL(_imgOri[l], scale);
    }

    _stdN *= scale;
    _stdD *= scale;
    _stdS *= scale;
}

void Optimiser::fwImg()
{
    FOR_EACH_2D_IMAGE
    {
        _fftImg.fwExecutePlanMT(_img[l]);
        _img[l].clearRL();

        _fftImg.fwExecutePlanMT(_imgOri[l]);
        _imgOri[l].clearRL();
    }
}

void Optimiser::bwImg()
{
    FOR_EACH_2D_IMAGE
    {
        _fftImg.bwExecutePlanMT(_img[l]);
        _img[l].clearFT();

        _fftImg.bwExecutePlanMT(_imgOri[l]);
        _imgOri[l].clearFT();
    }
}

void Optimiser::initCTF()
{
    IF_MASTER return;

    _ctfAttr.clear();
    _ctf.clear();

    CTFAttr ctfAttr;

    FOR_EACH_2D_IMAGE
    {
        _db.ctf(ctfAttr, _ID[l]);

        _ctfAttr.push_back(ctfAttr);

        _ctf.push_back(Image(size(), size(), FT_SPACE));
    }

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
#ifdef VERBOSE_LEVEL_3
        ALOG(INFO, "LOGGER_SYS") << "Initialising CTF for Image " << _ID[l];
        BLOG(INFO, "LOGGER_SYS") << "Initialising CTF for Image " << _ID[l];
#endif

        CTF(_ctf[l],
            _para.pixelSize,
            _ctfAttr[l].voltage,
            _ctfAttr[l].defocusU,
            _ctfAttr[l].defocusV,
            _ctfAttr[l].defocusTheta,
            _ctfAttr[l].Cs,
            0);
    }
}

void Optimiser::correctScale(const bool init,
                             const bool coord,
                             const bool group)
{
    ALOG(INFO, "LOGGER_SYS") << "Refreshing Scale";
    BLOG(INFO, "LOGGER_SYS") << "Refreshing Scale";

    refreshScale(coord, group);

    IF_MASTER return;

    ALOG(INFO, "LOGGER_SYS") << "Correcting Scale";
    BLOG(INFO, "LOGGER_SYS") << "Correcting Scale";

    if (init)
    {
        for (int l = 0; l < _para.k; l++)
        {
            #pragma omp parallel for
            SCALE_FT(_model.ref(l), _scale(_groupID[0] - 1));
        }
    }
    else
    {
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            FOR_EACH_PIXEL_FT(_img[l])
            {
                _img[l][i] /= _scale(_groupID[l] - 1);
                _imgOri[l][i] /= _scale(_groupID[l] - 1);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < _nGroup; i++)
        {
            _sig.row(i) /= TSGSL_pow_2(_scale(i));
        }
    }
}

void Optimiser::initSigma()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Image";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Average Image";

#ifdef OPTIMISER_SIGMA_MASK
    Image avg = _img[0].copyImage();
#else
    Image avg = _imgOri[0].copyImage();
#endif

    for (size_t l = 1; l < _ID.size(); l++)
    {
        #pragma omp parallel for
#ifdef OPTIMISER_SIGMA_MASK
        ADD_FT(avg, _img[l]);
#else
        ADD_FT(avg, _imgOri[l]);
#endif
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &avg[0],
                  avg.sizeFT(),
                  TS_MPI_DOUBLE_COMPLEX,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    #pragma omp parallel for
    SCALE_FT(avg, 1.0 / _N);

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";

    vec avgPs = vec::Zero(maxR());

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        vec ps(maxR());

#ifdef OPTIMISER_SIGMA_MASK
        powerSpectrum(ps, _img[l], maxR());
#else
        powerSpectrum(ps, _imgOri[l], maxR());
#endif

        #pragma omp critical
        avgPs += ps;
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs.data(),
                  maxR(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);
    
    MPI_Barrier(_hemi);


    avgPs /= _N;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Expectation for Initializing Sigma";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Expectation for Initializing Sigma";

    vec psAvg(maxR());
    for (int i = 0; i < maxR(); i++)
    {
        psAvg(i) = ringAverage(i,
                               avg,
                               function<RFLOAT(const Complex)>(&gsl_real_imag_sum));
        psAvg(i) = TSGSL_pow_2(psAvg(i));
    }

    // avgPs -> average power spectrum
    // psAvg -> expectation of pixels
    ALOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";
    BLOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";

    _sig.leftCols(_sig.cols() - 1).rowwise() = (avgPs - psAvg).transpose() / 2;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Reciprocal of Sigma";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Reciprocal of Sigma";

    for (int i = 0; i < _nGroup; i++)
        for (int j = 0; j < maxR(); j++)
            _sigRcp(i, j) = -0.5 / _sig(i, j);
}

void Optimiser::initParticles()
{
    IF_MASTER return;

    _par.clear();
    _par.resize(_ID.size());

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
#ifdef VERBOSE_LEVEL_3
        ALOG(INFO, "LOGGER_SYS") << "Initialising Particle Filter for Image " << _ID[l];
        BLOG(INFO, "LOGGER_SYS") << "Initialising Particle Filter for Image " << _ID[l];
#endif
        _par[l].init(_para.mode,
                     _para.transS,
                     TRANS_Q,
                     &_sym);
    }
}

void Optimiser::avgStdR(RFLOAT& stdR)
{
    IF_MASTER return;

    /***
    stdR = 0;

    FOR_EACH_2D_IMAGE
        stdR += _db.stdR(_ID[l]);

    MPI_Allreduce(MPI_IN_PLACE,
                 &stdR,
                 1,
                 TS_MPI_DOUBLE,
                 MPI_SUM,
                 _hemi);

    stdR /= _N;
    ***/
}

void Optimiser::avgStdT(RFLOAT& stdT)
{
    IF_MASTER return;

    /***
    stdT = 0;

    FOR_EACH_2D_IMAGE
    {
        stdT += _db.stdTX(_ID[l]);
        stdT += _db.stdTY(_ID[l]);
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  &stdT,
                  1,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    stdT /= _N;
    stdT /= 2;
    ***/
}

void Optimiser::loadParticles()
{
    IF_MASTER return;

    /***
    RFLOAT stdR, stdT;

    avgStdR(stdR);
    avgStdT(stdT);

    ALOG(INFO, "LOGGER_SYS") << "Average Standard Deviation of Rotation: " << stdR;
    BLOG(INFO, "LOGGER_SYS") << "Average Standard Deviation of Rotation: " << stdR;

    ALOG(INFO, "LOGGER_SYS") << "Average Standard Deviation of Translation: " << stdT;
    BLOG(INFO, "LOGGER_SYS") << "Average Standard Deviation of Translation: " << stdT;
    ***/

    // size_t cls;
    vec4 quat;
    vec2 tran;
    RFLOAT d;

    RFLOAT k1, k2, k3, stdTX, stdTY, stdD;

    //#pragma omp parallel for private(cls, quat, stdR, tran, d)

    #pragma omp parallel for private(quat, tran, d, k1, k2, k3, stdTX, stdTY, stdD)
    FOR_EACH_2D_IMAGE
    {
        #pragma omp critical
        {
            // cls = _db.cls(_ID[l]);
            quat = _db.quat(_ID[l]);
            //stdR = _db.stdR(_ID[l]);
            tran = _db.tran(_ID[l]);
            d = _db.d(_ID[l]);

            k1 = _db.k1(_ID[l]);
            k2 = _db.k2(_ID[l]);
            k3 = _db.k3(_ID[l]);

            stdTX = _db.stdTX(_ID[l]);
            stdTY = _db.stdTY(_ID[l]);
            stdD = _db.stdD(_ID[l]);
        }

        _par[l].load(_para.mLR,
                     _para.mLT,
                     1,
                     quat,
                     k1,
                     k2,
                     k3,
                     tran,
                     stdTX,
                     stdTY,
                     d,
                     stdD);
    }

    for (int l = 0; l < 10; l++)
    {
        ALOG(INFO, "LOGGER_SYS") << "Compress of "
                                 << l
                                 << " : "
                                 << _par[l].compress();
    }
}

void Optimiser::refreshRotationChange()
{
    /***
    RFLOAT mean = 0;
    RFLOAT std = 0;

    int num = 0;

    NT_MASTER
    {
        FOR_EACH_2D_IMAGE
        {
            RFLOAT diffR = _par[l].diffTopR();

            if (_par[l].diffTopC())
            {
                mean += diffR;
                std += TSGSL_pow_2(diffR);
                num += 1;
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  &mean,
                  1,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  &std,
                  1,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  &num,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    mean /= num;

    std = sqrt(std / num - TSGSL_pow_2(mean));
    ***/

    vec rc = vec::Zero(_nPar);

    NT_MASTER
    {
        FOR_EACH_2D_IMAGE
        {
            RFLOAT diff = _par[l].diffTopR();

            rc(_ID[l]) = diff;

            /***
            if (_par[l].diffTopC())
                rc(_ID[l]) = _par[l].diffTopR();
            else
                rc(_ID[l]) = 1;
            ***/
        }
    }


    MPI_Allreduce(MPI_IN_PLACE,
                  rc.data(),
                  rc.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 
/***
    int nNoZero = 0;
    for (int i = 0; i < _nPar; i++)
        if (rc(i) != 0)
            nNoZero += 1;
    ***/

    //vec rcNoZero = vec::Zero(nNoZero);

    //TSGSL_sort_largest(rcNoZero.data(), nNoZero, rc.data(), 1, _nPar);
    //TSGSL_sort_largest(rc.data(), nNoZero, rc.data(), 1, _nPar);
    TSGSL_sort(rc.data(), 1, _nPar);

    RFLOAT mean, std;
    stat_MAS(mean, std, rc, _nPar);
    //stat_MAS(mean, std, rc, nNoZero);
    //stat_MAS(mean, std, rcNoZero, nNoZero);

    _model.setRChange(mean);
    _model.setStdRChange(std);
}

void Optimiser::refreshClassDistr()
{
    _cDistr = vec::Zero(_para.k);

    NT_MASTER
    {
        size_t cls;

        #pragma omp parallel for private(cls)
        FOR_EACH_2D_IMAGE
        {
            for (int k = 0; k < _para.k; k++)
            {
                _par[l].rand(cls); 

                #pragma omp atomic
                _cDistr(cls) += 1;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  _cDistr.data(),
                  _cDistr.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    _cDistr.array() /= (_nPar * _para.k);
}

void Optimiser::balanceClass(const RFLOAT thres,
                             const bool refreshDistr)
{
    int cls;
    RFLOAT num = _cDistr.maxCoeff(&cls);

    for (int t = 0; t < _para.k; t++)
        if (_cDistr(t) < thres / _para.k)
        {
            NT_MASTER _model.ref(t) = _model.ref(cls).copyVolume();

            if (refreshDistr) _cDistr(t) = num;
        }

    if (refreshDistr) _cDistr.array() /= _cDistr.sum();
}

void Optimiser::refreshVariance()
{
    vec rv = vec::Zero(_nPar);
    vec t0v = vec::Zero(_nPar);
    vec t1v = vec::Zero(_nPar);

    NT_MASTER
    {
        RFLOAT rVari, tVariS0, tVariS1, dVari;

        #pragma omp parallel for private(rVari, tVariS0, tVariS1)
        FOR_EACH_2D_IMAGE
        {
            _par[l].vari(rVari,
                         tVariS0,
                         tVariS1,
                         dVari);

            rv(_ID[l]) = rVari;
            t0v(_ID[l]) = tVariS0;
            t1v(_ID[l]) = tVariS1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  rv.data(),
                  rv.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Allreduce(MPI_IN_PLACE,
                  t0v.data(),
                  t0v.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Allreduce(MPI_IN_PLACE,
                  t1v.data(),
                  t1v.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);

    ALOG(INFO, "LOGGER_SYS") << "Maximum Rotation Variance: " << rv.maxCoeff();
    BLOG(INFO, "LOGGER_SYS") << "Maximum Rotation Variance: " << rv.maxCoeff();

    RFLOAT mean, std;

    stat_MAS(mean, std, rv, _nPar);

    _model.setRVari(mean);
    _model.setStdRVari(std);

    stat_MAS(mean, std, t0v, _nPar);

    _model.setTVariS0(mean);
    _model.setStdTVariS0(std);

    stat_MAS(mean, std, t1v, _nPar);

    _model.setTVariS1(mean);
    _model.setStdTVariS1(std);
}

void Optimiser::refreshScale(const bool coord,
                             const bool group)
{
    if (_iter != 0)
        _rS = _model.resolutionP(_para.thresSclCorFSC, false);

    if (_rS > _r)
    {
        MLOG(WARNING, "LOGGER_SYS") << "_rS is Larger than _r, Set _rS to _r";
        _rS = _r;
    }

    MLOG(INFO, "LOGGER_SYS") << "Upper Boundary Frequency for Scale Correction: "
                             << _rS;

    mat mXA = mat::Zero(_nGroup, _rS);
    mat mAA = mat::Zero(_nGroup, _rS);

    vec sXA = vec::Zero(_rS);
    vec sAA = vec::Zero(_rS);

    NT_MASTER
    {
        Image img(size(), size(), FT_SPACE);

        size_t cls;
        mat22 rot2D;
        mat33 rot3D;
        vec2 tran;
        RFLOAT d;

        FOR_EACH_2D_IMAGE
        {
#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Projecting from the Initial Reference from a Random Rotation for Image " << _ID[l];
            BLOG(INFO, "LOGGER_SYS") << "Projecting from the Initial Reference from a Random Rotation for Image " << _ID[l];
#endif

            if (!coord)
            {
                if (_para.mode == MODE_2D)
                {
                    randRotate2D(rot2D);
#ifdef VERBOSE_LEVEL_3
                ALOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot2D;
                BLOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot2D;
#endif
                }
                else if (_para.mode == MODE_3D)
                {
                    randRotate3D(rot3D);
#ifdef VERBOSE_LEVEL_3
                ALOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot3D;
                BLOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot3D;
#endif
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");

                if (_para.mode == MODE_2D)
                {
                    _model.proj(0).projectMT(img, rot2D);
                }
                else if (_para.mode == MODE_3D)
                {
                    _model.proj(0).projectMT(img, rot3D);
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");
            }
            else
            {
                if (_para.mode == MODE_2D)
                {
                    _par[l].rank1st(cls, rot2D, tran, d);
                }
                else if (_para.mode == MODE_3D)
                {
                    _par[l].rank1st(cls, rot3D, tran, d);
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_SCALE_MASK
                    _model.proj(cls).projectMT(img, rot2D, tran);
#else
                    _model.proj(cls).projectMT(img, rot2D, tran - _offset[l]);
#endif
#else
                    _model.proj(cls).projectMT(img, rot2D, tran);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_SCALE_MASK
                    _model.proj(cls).projectMT(img, rot3D, tran);
#else
                    _model.proj(cls).projectMT(img, rot3D, tran - _offset[l]);
#endif
#else
                    _model.proj(cls).projectMT(img, rot3D, tran);
#endif
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");
            }

#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Calculating Intensity Scale for Image " << l;
            BLOG(INFO, "LOGGER_SYS") << "Calculating Intensity Scale for Image " << l;
#endif

#ifdef OPTIMISER_REFRESH_SCALE_RL_ZERO
            RFLOAT rL = 0;
#else
            RFLOAT rL = _rL;
#endif

#ifdef OPTIMISER_SCALE_MASK
            scaleDataVSPrior(sXA,
                             sAA,
                             _img[l],
                             img,
                             _ctf[l],
                             _rS,
                             rL);
#else
            scaleDataVSPrior(sXA,
                             sAA,
                             _imgOri[l],
                             img,
                             _ctf[l],
                             _rS,
                             rL);
#endif

#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Accumulating Intensity Scale Information from Image " << l;
            BLOG(INFO, "LOGGER_SYS") << "Accumulating Intensity Scale Information from Image " << l;
#endif

            if (group)
            {
                mXA.row(_groupID[l] - 1) += sXA.transpose();
                mAA.row(_groupID[l] - 1) += sAA.transpose();
            }
            else
            {
                mXA.row(0) += sXA.transpose();
                mAA.row(0) += sAA.transpose();
            }
        }
    }

#ifdef VERBOSE_LEVEL_2
    ILOG(INFO, "LOGGER_SYS") << "Intensity Scale Information Calculated";
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Accumulating Intensity Scale Information from All Processes";

    MPI_Allreduce(MPI_IN_PLACE,
                  mXA.data(),
                  mXA.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  mAA.data(),
                  mAA.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    if (group)
    {
        for (int i = 0; i < _nGroup; i++)
        {
#ifdef OPTIMISER_REFRESH_SCALE_SPECTRUM
            RFLOAT sum = 0;
            int count = 0;

            for (int r = (int)rL; r < _rS; r++)
            {
                sum += mXA(i, r) / mAA(i, r);
                count += 1;
            }

            _scale(i) = sum / count;
#else
            _scale(i) = mXA.row(i).sum() / mAA.row(i).sum();
#endif
        }
    }
    else
    {
#ifdef OPTIMISER_REFRESH_SCALE_SPECTRUM
        RFLOAT sum = 0;
        int count = 0;

        for (int r = (int)rL; r < _rS; r++)
        {
            sum += mXA(0, r) / mAA(0, r);
            count += 1;
        }
        
        for (int i = 0; i < _nGroup; i++)
            _scale(i) = sum / count;
#else
        /***
        MLOG(INFO, "LOGGER_SYS") << mAA(0, 0)
                                 << ", "
                                 << mAA(0, 1);
        ***/

        for (int i = 0; i < _nGroup; i++)
            _scale(i) = mXA.row(0).sum() / mAA.row(0).sum();
#endif
    }

    RFLOAT medianScale = median(_scale, _scale.size());

    MLOG(INFO, "LOGGER_ROUND") << "Median Intensity Scale: " << medianScale;

    MLOG(INFO, "LOGGER_ROUND") << "Removing Extreme Values from Intensity Scale";

    for (int i = 0; i < _nGroup; i++)
    {
        if (fabs(_scale(i)) > fabs(medianScale * 5))
            _scale(i) = medianScale * 5;
        else if (fabs(_scale(i)) < fabs(medianScale / 5))
            _scale(i) = medianScale / 5;
    }

    RFLOAT meanScale = _scale.mean();
    
    MLOG(INFO, "LOGGER_ROUND") << "Average Intensity Scale: " << meanScale;

    if (meanScale < 0)
    {
        REPORT_ERROR("AVERAGE INTENSITY SCALE SHOULD NOT BE SMALLER THAN ZERO");
        abort();
    }

    /***
    if (medianScale * meanScale < 0)
        CLOG(FATAL, "LOGGER_SYS") << "Median and Mean of Intensity Scale Should Have the Same Sign";
    ***/

    MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Intensity Scale: "
                               << TSGSL_stats_sd(_scale.data(), 1, _scale.size());

    /***
    if (!init)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Making Average Intensity Scale be 1";

        for (int i = 0; i < _nGroup; i++)
            _scale(i) /= fabs(meanScale);
    }
    ***/

    IF_MASTER
    {
#ifdef VERBOSE_LEVEL_2
        for (int i = 0; i < _nGroup; i++)
            MLOG(INFO, "LOGGER_ROUND") << "Scale of Group " << i << " is " << _scale(i);
#endif
    }
}

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
void Optimiser::reCentreImg()
{
    IF_MASTER return;

    vec2 tran;

    #pragma omp parallel for private(tran)
    FOR_EACH_2D_IMAGE
    {
        _par[l].rank1st(tran);

        _offset[l](0) -= tran(0);
        _offset[l](1) -= tran(1);

        translate(_img[l],
                  _imgOri[l],
                  _offset[l](0),
                  _offset[l](1));

        _par[l].setT(_par[l].t().rowwise() - tran.transpose());
    }
}
#endif

void Optimiser::reMaskImg()
{
    IF_MASTER return;

#ifdef OPTIMISER_MASK_IMG
    if (_para.zeroMask)
    {
        Image mask(_para.size, _para.size, RL_SPACE);

        softMask(mask,
                 _para.maskRadius / _para.pixelSize,
                 EDGE_WIDTH_RL);

        FOR_EACH_2D_IMAGE
        {
            _fftImg.bwExecutePlanMT(_img[l]);

            #pragma omp parallel for
            MUL_RL(_img[l], mask);

            _fftImg.fwExecutePlanMT(_img[l]);

            _img[l].clearRL();
        }
    }
    else
    {
        //TODO Make the background a noise with PowerSpectrum of sigma2
    }
#endif
}

void Optimiser::normCorrection()
{
    // skip norm correction in the first iteration
    if (_iter == 0) return;

    RFLOAT rNorm = TSGSL_MIN_RFLOAT(_r, _model.resolutionP(0.75, false));

    vec norm = vec::Zero(_nPar);

    size_t cls;

    mat22 rot2D;
    mat33 rot3D;

    vec2 tran;

    RFLOAT d;

    NT_MASTER
    {
        #pragma omp parallel for private(cls, rot2D, rot3D, tran, d)
        FOR_EACH_2D_IMAGE
        {
            Image img(size(), size(), FT_SPACE);

            SET_0_FT(img);

            for (int m = 0; m < _para.mReco; m++)
            {
                if (_para.mode == MODE_2D)
                {
                    _par[l].rand(cls, rot2D, tran, d);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_NORM_MASK
                    _model.proj(cls).project(img, rot2D, tran);
#else
                    _model.proj(cls).project(img, rot2D, tran - _offset[l]);
#endif
#else
                    _model.proj(cls).project(img, rot2D, tran);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
                    _par[l].rand(cls, rot3D, tran, d);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_NORM_MASK
                    _model.proj(cls).project(img, rot3D, tran);
#else
                    _model.proj(cls).project(img, rot3D, tran - _offset[l]);
#endif
#else
                    _model.proj(cls).project(img, rot3D, tran);
#endif
                }

                if (_searchType != SEARCH_TYPE_CTF)
                {
                    FOR_EACH_PIXEL_FT(img)
                        img[i] *= REAL(_ctf[l][i]);
                }
                else
                {
                    Image ctf(_para.size, _para.size, FT_SPACE);
                    CTF(ctf,
                        _para.pixelSize, 
                        _ctfAttr[l].voltage,
                        _ctfAttr[l].defocusU * d,
                        _ctfAttr[l].defocusV * d,
                        _ctfAttr[l].defocusTheta,
                        _ctfAttr[l].Cs);

                    FOR_EACH_PIXEL_FT(img)
                        img[i] *= REAL(ctf[i]);
                }

#ifdef OPTIMISER_ADJUST_2D_IMAGE_NOISE_ZERO_MEAN
                _img[l][0] = img[0];
                _imgOri[l][0] = img[0];
#endif

                NEG_FT(img);

#ifdef OPTIMISER_NORM_MASK
                ADD_FT(img, _img[l]);
#else
                ADD_FT(img, _imgOri[l]);
#endif

                IMAGE_FOR_EACH_PIXEL_FT(img)
                {
                    if ((QUAD(i, j) >= TSGSL_pow_2(_rL)) ||
                        (QUAD(i, j) < TSGSL_pow_2(rNorm)))
                        norm(_ID[l]) += ABS2(img.getFTHalf(i, j));
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

     MPI_Allreduce(MPI_IN_PLACE,
                  norm.data(),
                  norm.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);

    /***
    IF_MASTER
    {
        for (int i = 0; i < 100; i++)
            MLOG(INFO, "LOGGER_SYS") << "norm "
                                     << i
                                     << " = "
                                     << norm[i];
    }
    ***/

    MLOG(INFO, "LOGGER_SYS") << "Max of Norm of Noise : "
                             << TSGSL_stats_max(norm.data(), 1, norm.size());

    MLOG(INFO, "LOGGER_SYS") << "Min of Norm of Noise : "
                             << TSGSL_stats_min(norm.data(), 1, norm.size());

    //RFLOAT m = TSGSL_stats_mean(norm.data(), 1, norm.size());

    RFLOAT m = median(norm, norm.size());

    MLOG(INFO, "LOGGER_SYS") << "Mean of Norm of Noise : "
                             << m;

    /***
    for (int i = 0; i < norm.size(); i++)
    {
        if (norm(i) < m / 5)
            norm(i) = m / 5;
        else if (norm(i) > m * 5)
            norm(i) = m * 5;
    }
    ***/

    /***
    RFLOAT sd = TSGSL_stats_sd_m(norm.data(), 1, norm.size(), m);

    MLOG(INFO, "LOGGER_SYS") << "Standard Deviation of Norm of Noise : "
                             << sd;
                             ***/

    NT_MASTER
    {
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            /***
            ALOG(INFO, "LOGGER_SYS") << "isEmptyRL of img " << _img[l].isEmptyRL();
            ALOG(INFO, "LOGGER_SYS") << "isEmptyFT of img " << _img[l].isEmptyFT();

            ALOG(INFO, "LOGGER_SYS") << "SizeRL of img " << _img[l].sizeRL();
            ALOG(INFO, "LOGGER_SYS") << "SizeFT of img " << _img[l].sizeFT();

            ALOG(INFO, "LOGGER_SYS") << "isEmptyRL of imgOri " << _imgOri[l].isEmptyRL();
            ALOG(INFO, "LOGGER_SYS") << "isEmptyFT of imgOri " << _imgOri[l].isEmptyFT();

            ALOG(INFO, "LOGGER_SYS") << "SizeRL of imgOri " << _imgOri[l].sizeRL();
            ALOG(INFO, "LOGGER_SYS") << "SizeFT of imgOri " << _imgOri[l].sizeFT();

            FFT fft;

            fft.bw(_imgOri[l]);

            FOR_EACH_PIXEL_RL(_imgOri[l])
                _imgOri[l](i) /= 2;

            fft.fw(_imgOri[l]);
            ***/
            
            /***
            fft.bw(_img[l]);

            FOR_EACH_PIXEL_RL(_img[l])
                _img[l](i) /= 2;

            fft.fw(_img[l]);
            ***/

            FOR_EACH_PIXEL_FT(_img[l])
            {
                _img[l][i] *= sqrt(m / norm(_ID[l]));
                _imgOri[l][i] *= sqrt(m / norm(_ID[l]));
            }
        }
    }
}

void Optimiser::allReduceSigma(const bool mask,
                               const bool group)
{
    IF_MASTER return;

#ifdef OPTIMISER_SIGMA_WHOLE_FREQUENCY
    int rSig = maxR();
#else
    int rSig = _r;
#endif

    ALOG(INFO, "LOGGER_ROUND") << "Clearing Up Sigma";
    BLOG(INFO, "LOGGER_ROUND") << "Clearing Up Sigma";

    // set re-calculating part to zero
    _sig.leftCols(rSig).setZero();
    _sig.rightCols(1).setZero();

    ALOG(INFO, "LOGGER_ROUND") << "Recalculating Sigma";
    BLOG(INFO, "LOGGER_ROUND") << "Recalculating Sigma";

    size_t cls;

    mat22 rot2D;
    mat33 rot3D;

    vec2 tran;

    RFLOAT d;

    omp_lock_t* mtx = new omp_lock_t[_nGroup];

    #pragma omp parallel for
    for (int l = 0; l < _nGroup; l++)
        omp_init_lock(&mtx[l]);

    #pragma omp parallel for private(cls, rot2D, rot3D, tran, d) schedule(dynamic)
    FOR_EACH_2D_IMAGE
    {
        for (int m = 0; m < _para.mReco; m++)
        {
            /***
            RFLOAT w;

            if (_para.parGra) 
                w = TSGSL_pow_2(_par[l].compress());
            else
                w = 1;
            ***/

            RFLOAT w = 1;

            Image img(size(), size(), FT_SPACE);

            SET_0_FT(img);

            vec sig(rSig);

            if (_para.mode == MODE_2D)
            {
                _par[l].rand(cls, rot2D, tran, d);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                if (mask)
                    _model.proj(cls).project(img, rot2D, tran);
                else
                    _model.proj(cls).project(img, rot2D, tran - _offset[l]);
#else
                _model.proj(cls).project(img, rot2D, tran);
#endif
            }
            else if (_para.mode == MODE_3D)
            {
                _par[l].rand(cls, rot3D, tran, d);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                if (mask)
                    _model.proj(cls).project(img, rot3D, tran);
                else
                    _model.proj(cls).project(img, rot3D, tran - _offset[l]);
#else
                _model.proj(cls).project(img, rot3D, tran);
#endif
            }

            if (_searchType != SEARCH_TYPE_CTF)
            {
                FOR_EACH_PIXEL_FT(img)
                    img[i] *= REAL(_ctf[l][i]);
            }
            else
            {
                Image ctf(_para.size, _para.size, FT_SPACE);
                CTF(ctf,
                    _para.pixelSize, 
                    _ctfAttr[l].voltage,
                    _ctfAttr[l].defocusU * d,
                    _ctfAttr[l].defocusV * d,
                    _ctfAttr[l].defocusTheta,
                    _ctfAttr[l].Cs);

                FOR_EACH_PIXEL_FT(img)
                    img[i] *= REAL(ctf[i]);
            }

            NEG_FT(img);

            if (mask)
                ADD_FT(img, _img[l]);
            else
                ADD_FT(img, _imgOri[l]);

            powerSpectrum(sig, img, rSig);

            if (group)
            {
                omp_set_lock(&mtx[_groupID[l] - 1]);

                _sig.row(_groupID[l] - 1).head(rSig) += w * sig.transpose() / 2;

                _sig(_groupID[l] - 1, _sig.cols() - 1) += w;

                omp_unset_lock(&mtx[_groupID[l] - 1]);
            }
            else
            {
                omp_set_lock(&mtx[0]);

                _sig.row(0).head(rSig) += w * sig.transpose() / 2;

                _sig(0, _sig.cols() - 1) += w;

                omp_unset_lock(&mtx[0]);
            }
        }
    }

    delete[] mtx;

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_ROUND") << "Averaging Sigma of Images Belonging to the Same Group";
    BLOG(INFO, "LOGGER_ROUND") << "Averaging Sigma of Images Belonging to the Same Group";

    
    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.data(),
                  rSig * _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.col(_sig.cols() - 1).data(),
                  _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    if (group)
    {
        #pragma omp parallel for
        for (int i = 0; i < _sig.rows(); i++)
            _sig.row(i).head(rSig) /= _sig(i, _sig.cols() - 1);
    }
    else
    {
        _sig.row(0).head(rSig) /= _sig(0, _sig.cols() - 1);

        #pragma omp parallel for
        for (int i = 1; i < _sig.rows(); i++)
            _sig.row(i).head(rSig) = _sig.row(0).head(rSig);
    }

    #pragma omp parallel for
    for (int i = rSig; i < _sig.cols() - 1; i++)
        _sig.col(i) = _sig.col(rSig - 1);

    #pragma omp parallel for
    for (int i = 0; i < _nGroup; i++)
        for (int j = 0; j < rSig; j++)
            _sigRcp(i, j) = -0.5 / _sig(i, j);
}

void Optimiser::reconstructRef(const bool fscFlag,
                               const bool avgFlag,
                               const bool fscSave,
                               const bool avgSave,
                               const bool finished)
{
    FFT fft;

    ALOG(INFO, "LOGGER_ROUND") << "Allocating Space for Pre-calcuation in Reconstruction";
    BLOG(INFO, "LOGGER_ROUND") << "Allocating Space for Pre-calcuation in Reconstruction";
    
    allocPreCalIdx(_model.rU(), 0);

    NT_MASTER
    {
        if ((_para.parGra) && (_para.k != 1))
        {
            ALOG(WARNING, "LOGGER_ROUND") << "PATTICLE GRADING IS ONLY RECOMMENDED IN REFINEMENT, NOT CLASSIFICATION";
            BLOG(WARNING, "LOGGER_ROUND") << "PATTICLE GRADING IS ONLY RECOMMENDED IN REFINEMENT, NOT CLASSIFICATION";
        }

        ALOG(INFO, "LOGGER_ROUND") << "Inserting High Probability 2D Images into Reconstructor";
        BLOG(INFO, "LOGGER_ROUND") << "Inserting High Probability 2D Images into Reconstructor";

        for (int t = 0; t < _para.k; t++)
            _model.reco(t).setPreCal(_nPxl, _iCol, _iRow, _iPxl, _iSig);

        bool cSearch = ((_searchType == SEARCH_TYPE_CTF) ||
                        ((_para.cSearch) &&
                        (_searchType == SEARCH_TYPE_STOP)));

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            Image ctf(_para.size, _para.size, FT_SPACE);

            RFLOAT w;

            if ((_para.parGra) && (_para.k == 1))
                w = _par[l].compress();
            else
                w = 1;

            w /= _para.mReco;

            Image transImg(_para.size, _para.size, FT_SPACE);

            for (int m = 0; m < _para.mReco; m++)
            {
                size_t cls;
                vec4 quat;
                vec2 tran;
                RFLOAT d;

                if (_para.mode == MODE_2D)
                {
                    _par[l].rand(cls, quat, tran, d);

                    mat22 rot2D;

                    rotate2D(rot2D, vec2(quat(0), quat(1)));

#ifdef OPTIMISER_RECONSTRUCT_WITH_UNMASK_IMAGE ///huabin
                    translate(transImg,
                              _imgOri[l],
                              -(tran - _offset[l])(0),
                              -(tran - _offset[l])(1),
                              _iCol,
                              _iRow,
                              _iPxl,
                              _nPxl);
#else
                    translate(transImg, //huabin
                              _img[l],
                              -tran(0),
                              -tran(1),
                              _iCol,
                              _iRow,
                              _iPxl,
                              _nPxl);
#endif

                    if (cSearch)
                        CTF(ctf,
                            _para.pixelSize,
                            _ctfAttr[l].voltage,
                            _ctfAttr[l].defocusU * d,
                            _ctfAttr[l].defocusV * d,
                            _ctfAttr[l].defocusTheta,
                            _ctfAttr[l].Cs);

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                    vec sig = _sig.row(_groupID[l] - 1).transpose();

                    _model.reco(cls).insertP(transImg,
                                             cSearch ? ctf : _ctf[l],
                                             rot2D,
                                             w,
                                             &sig);
#else
                    _model.reco(cls).insertP(transImg,
                                             cSearch ? ctf : _ctf[l],
                                             rot2D,
                                             w);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
                    _par[l].rand(cls, quat, tran, d);

                    mat33 rot3D;

                    rotate3D(rot3D, quat);
                
#ifdef OPTIMISER_RECONSTRUCT_WITH_UNMASK_IMAGE
                    translate(transImg,
                              _imgOri[l],
                              -(tran - _offset[l])(0),
                              -(tran - _offset[l])(1),
                              _iCol,
                              _iRow,
                              _iPxl,
                              _nPxl);
#else
                    translate(transImg,
                              _img[l],
                              -tran(0),
                              -tran(1),
                              _iCol,
                              _iRow,
                              _iPxl,
                              _nPxl);
#endif

                    /***
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE

                    IMAGE_FOR_EACH_PIXEL_FT(transImg)
                        if (QUAD(i, j) < TSGSL_pow_2(_model.reco(cls).maxRadius()))
                            transImg.setFTHalf(transImg.getFTHalf(i, j)
                                             / _sig(_groupID[l] - 1, AROUND(NORM(i, j))),
                                               i,
                                               j);
                                         
#endif
                    **/

                    if (cSearch)
                        CTF(ctf,
                            _para.pixelSize,
                            _ctfAttr[l].voltage,
                            _ctfAttr[l].defocusU * d,
                            _ctfAttr[l].defocusV * d,
                            _ctfAttr[l].defocusTheta,
                            _ctfAttr[l].Cs);

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                    vec sig = _sig.row(_groupID[l] - 1).transpose();

                    _model.reco(cls).insertP(transImg,
                                             cSearch ? ctf : _ctf[l],
                                             rot3D,
                                             w,
                                             &sig);
#else
                    _model.reco(cls).insertP(transImg,
                                             cSearch ? ctf : _ctf[l],
                                             rot3D,
                                             w);
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }
            }
        }

#ifdef VERBOSE_LEVEL_2
        ILOG(INFO, "LOGGER_ROUND") << "Inserting Images Into Reconstructor(s) Accomplished";
#endif

        MPI_Barrier(_hemi);

        for (int t = 0; t < _para.k; t++)
        {
            ALOG(INFO, "LOGGER_ROUND") << "Preparing Content in Reconstructor of Reference "
                                       << t;
            BLOG(INFO, "LOGGER_ROUND") << "Preparing Content in Reconstructor of Reference "
                                       << t;

            _model.reco(t).prepareTF();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (fscFlag)
    {
        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).setMAP(false);

#ifdef OPTIMISER_RECONSTRUCT_JOIN_HALF
                _model.reco(t).setJoinHalf(true);
#else
                _model.reco(t).setJoinHalf(false);
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_2D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_3D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }

                ALOG(INFO, "LOGGER_ROUND") << "Reconstructing Reference "
                                           << t
                                           << " for Determining FSC";
                BLOG(INFO, "LOGGER_ROUND") << "Reconstructing Reference "
                                           << t
                                           << " for Determining FSC";

                Volume ref;

                _model.reco(t).reconstruct(ref);

#ifdef VERBOSE_LEVEL_2
                ALOG(INFO, "LOGGER_ROUND") << "Fourier Transforming Reference " << t;
                BLOG(INFO, "LOGGER_ROUND") << "Fourier Transforming Reference " << t;
#endif

                fft.fwMT(ref);

                #pragma omp parallel for
                SET_0_FT(_model.ref(t));

                COPY_FT(_model.ref(t), ref);

#ifdef VERBOSE_LEVEL_2
                ALOG(INFO, "LOGGER_ROUND") << "Reference " << t << "Fourier Transformed";
                BLOG(INFO, "LOGGER_ROUND") << "Reference " << t << "Fourier Transformed";
#endif
            }
        }

        if (fscSave)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Saving Reference(s)";

            if (_para.mode == MODE_2D)
            {
#ifdef OPTIMISER_2D_SAVE_JOIN_MAP
                saveMapJoin(finished);
#else
                saveMapHalf(finished);
#endif
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_3D_SAVE_JOIN_MAP
                saveMapJoin(finished);
#else
                saveMapHalf(finished);
#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_BALANCE_CLASS

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Balancing Class(es)";

            balanceClass(CLASS_BALANCE_FACTOR, false);

#ifdef VERBOSE_LEVEL_1

            MLOG(INFO, "LOGGER_ROUND") << "Percentage of Images Belonging to Each Class After Balancing";

            for (int t = 0; t < _para.k; t++)
                MLOG(INFO, "LOGGER_ROUND") << _cDistr(t) * 100
                                           << "\% Percentage of Images Belonging to Class "
                                           << t;
#endif
        }

#endif

        _model.compareTwoHemispheres(true, false, _para.thresReportFSC);
    }

    if (avgFlag)
    {
        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).setMAP(true);

#ifdef OPTIMISER_RECONSTRUCT_JOIN_HALF
                _model.reco(t).setJoinHalf(true);
#else
                _model.reco(t).setJoinHalf(false);
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_2D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_3D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }

                ALOG(INFO, "LOGGER_ROUND") << "Reconstructing Reference "
                                           << t
                                           << " for Next Iteration";
                BLOG(INFO, "LOGGER_ROUND") << "Reconstructing Reference "
                                           << t
                                           << " for Next Iteration";

                Volume ref;

                _model.reco(t).reconstruct(ref);

#ifdef VERBOSE_LEVEL_2
                ALOG(INFO, "LOGGER_ROUND") << "Fourier Transforming Reference " << t;
                BLOG(INFO, "LOGGER_ROUND") << "Fourier Transforming Reference " << t;
#endif

                fft.fwMT(ref);

                #pragma omp parallel for
                SET_0_FT(_model.ref(t));

                COPY_FT(_model.ref(t), ref);

#ifdef VERBOSE_LEVEL_2
                ALOG(INFO, "LOGGER_ROUND") << "Reference " << t << "Fourier Transformed";
                BLOG(INFO, "LOGGER_ROUND") << "Reference " << t << "Fourier Transformed";
#endif
            }
        }

        if (avgSave)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Saving Reference(s)";

            if (_para.mode == MODE_2D)
            {
#ifdef OPTIMISER_2D_SAVE_JOIN_MAP
                saveMapJoin(finished);
#else
                saveMapHalf(finished);
#endif
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_3D_SAVE_JOIN_MAP
                saveMapJoin(finished);
#else
                saveMapHalf(finished);
#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_BALANCE_CLASS

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Balancing Class(es)";

            balanceClass(CLASS_BALANCE_FACTOR, true);

#ifdef VERBOSE_LEVEL_1

            MLOG(INFO, "LOGGER_ROUND") << "Percentage of Images Belonging to Each Class After Balancing";

            for (int t = 0; t < _para.k; t++)
                MLOG(INFO, "LOGGER_ROUND") << _cDistr(t) * 100
                                           << "\% Percentage of Images Belonging to Class "
                                           << t;
#endif
        }

#endif

        _model.compareTwoHemispheres(false, true, _para.thresReportFSC);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Freeing Space for Pre-calcuation in Reconstruction";
    BLOG(INFO, "LOGGER_ROUND") << "Freeing Space for Pre-calcuation in Reconstruction";

    freePreCalIdx();

    MPI_Barrier(MPI_COMM_WORLD);

    ALOG(INFO, "LOGGER_ROUND") << "Reference(s) Reconstructed";
    BLOG(INFO, "LOGGER_ROUND") << "Reference(s) Reconstructed";
}

void Optimiser::solventFlatten(const bool mask)
{
    IF_MASTER return;

    for (int t = 0; t < _para.k; t++)
    {
#ifdef OPTIMISER_SOLVENT_FLATTEN_LOW_PASS
        ALOG(INFO, "LOGGER_ROUND") << "Low Pass Filter on Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Low Pass Filter on Reference " << t;

        lowPassFilter(_model.ref(t),
                      _model.ref(t),
                      (RFLOAT)_r  / _para.size,
                      (RFLOAT)EDGE_WIDTH_FT / _para.size);
#endif

        ALOG(INFO, "LOGGER_ROUND") << "Inverse Fourier Transforming Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Inverse Fourier Transforming Reference " << t;

        FFT fft;
        fft.bwMT(_model.ref(t));

#ifdef OPTIMISER_SOLVENT_FLATTEN_STAT_REMOVE_BG

        RFLOAT bgMean, bgStddev;

        bgMeanStddev(bgMean,
                     bgStddev,
                     _model.ref(t),
                     _para.size / 2,
                     _para.maskRadius / _para.pixelSize);

        ALOG(INFO, "LOGGER_ROUND") << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgMean;
        BLOG(INFO, "LOGGER_ROUND") << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgMean;
        ALOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgStddev;
        BLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgStddev;

        //RFLOAT bgThres = bgMean + bgStddev * TSGSL_cdf_gaussian_Qinv(0.01, 1);
        RFLOAT bgThres = bgMean + bgStddev * TSGSL_cdf_gaussian_Qinv(1e-3, 1);

        ALOG(INFO, "LOGGER_ROUND") << "Threshold for Removing Background of Reference "
                                   << t
                                   << ": "
                                   << bgThres;
        BLOG(INFO, "LOGGER_ROUND") << "Threshold for Removing Background of Reference "
                                   << t
                                   << ": "
                                   << bgThres;

        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(_model.ref(t))
            if (_model.ref(t)(i) < bgThres)
                _model.ref(t)(i) = bgMean;

        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(_model.ref(t))
                _model.ref(t)(i) -= bgMean;
#endif

#ifdef OPTIMISER_SOLVENT_FLATTEN_SUBTRACT_BG
        ALOG(INFO, "LOGGER_ROUND") << "Subtracting Background from Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Subtracting Background from Reference " << t;

        RFLOAT bg = regionMean(_model.ref(t),
                               _para.maskRadius / _para.pixelSize + EDGE_WIDTH_RL);

        ALOG(INFO, "LOGGER_ROUND") << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bg;
        BLOG(INFO, "LOGGER_ROUND") << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bg;

        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(_model.ref(t))
            (_model.ref(t))(i) -= bg;
#endif

#ifdef OPTIMISER_SOLVENT_FLATTEN_REMOVE_NEG
        ALOG(INFO, "LOGGER_ROUND") << "Removing Negative Values from Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Removing Negative Values from Reference " << t;

        #pragma omp parallel for
        REMOVE_NEG(_model.ref(t));
#endif

        if (mask && !_mask.isEmptyRL())
        {
            ALOG(INFO, "LOGGER_ROUND") << "Performing Reference Masking";
            BLOG(INFO, "LOGGER_ROUND") << "Performing Reference Masking";

            if (_para.mode == MODE_2D)
            {
                REPORT_ERROR("2D MODE DO NOT SUPPORTS PROVIDED MASK");

                abort();
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_SOLVENT_FLATTEN_LOW_PASS_MASK

                fft.fwMT(_mask);
                _mask.clearRL();

                Volume lowPassMask(_para.size, _para.size, _para.size, FT_SPACE);

                lowPassFilter(lowPassMask,
                              _mask,
                              (RFLOAT)_r / _para.size,
                              (RFLOAT)EDGE_WIDTH_FT / _para.size);

                fft.bwMT(lowPassMask);

#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(_model.ref(t), _model.ref(t), lowPassMask, 0);
#else
                softMask(_model.ref(t), _model.ref(t), lowPassMask);
#endif

#else
                
#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(_model.ref(t), _model.ref(t), _mask, 0);
#else
                softMask(_model.ref(t), _model.ref(t), _mask);
#endif

#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }
        else
        {
            ALOG(INFO, "LOGGER_ROUND") << "Performing Solvent Flatten of Reference " << t;
            BLOG(INFO, "LOGGER_ROUND") << "Performing Solvent Flatten of Reference " << t;

            if (_para.mode == MODE_2D)
            {
                Image ref(_para.size,
                          _para.size,
                          RL_SPACE);

                SLC_EXTRACT_RL(ref, _model.ref(t), 0);

#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(ref,
                         ref, 
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         0);
#else
                softMask(ref,
                         ref, 
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL);
#endif

                COPY_RL(_model.ref(t), ref);
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(_model.ref(t),
                         _model.ref(t),
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         0);
#else
                softMask(_model.ref(t),
                         _model.ref(t),
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL);
#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }

        ALOG(INFO, "LOGGER_ROUND") << "Fourier Transforming Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Fourier Transforming Reference " << t;

        fft.fwMT(_model.ref(t));
        _model.ref(t).clearRL();
    }
}

void Optimiser::allocPreCalIdx(const RFLOAT rU,
                               const RFLOAT rL)
{
    IF_MASTER return;

    _iPxl = new int[_img[0].sizeFT()];

    _iCol = new int[_img[0].sizeFT()];

    _iRow = new int[_img[0].sizeFT()];

    _iSig = new int[_img[0].sizeFT()];

    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    _nPxl = 0;

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));

            if ((v < rU) && (v >= rL))
            {
                _iPxl[_nPxl] = _img[0].iFTHalf(i, j);

                _iCol[_nPxl] = i;

                _iRow[_nPxl] = j;

                _iSig[_nPxl] = v;

                _nPxl++;
            }
        }
    }
}

void Optimiser::allocPreCal(const bool pixelMajor,
                            const bool ctf)
{
    IF_MASTER return;

    _datP = (Complex*)TSFFTW_malloc(_ID.size() * _nPxl * sizeof(Complex));

    _ctfP = (RFLOAT*)TSFFTW_malloc(_ID.size() * _nPxl * sizeof(RFLOAT));

    _sigRcpP = (RFLOAT*)TSFFTW_malloc(_ID.size() * _nPxl * sizeof(RFLOAT));

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        for (int i = 0; i < _nPxl; i++)
        {
            _datP[pixelMajor
                ? (i * _ID.size() + l)
                : (_nPxl * l + i)] = _img[l].iGetFT(_iPxl[i]);

            _ctfP[pixelMajor
                ? (i * _ID.size() + l)
                : (_nPxl * l + i)] = REAL(_ctf[l].iGetFT(_iPxl[i]));

            _sigRcpP[pixelMajor
                   ? (i * _ID.size() + l)
                   : (_nPxl * l + i)] = _sigRcp(_groupID[l] - 1, _iSig[i]);
        }
    }

    if (ctf)
    {
        _frequency = new RFLOAT[_nPxl];

        _defocusP = new RFLOAT[_ID.size() * _nPxl];

        _K1 = new RFLOAT[_ID.size()];

        _K2 = new RFLOAT[_ID.size()];

        for (int i = 0; i < _nPxl; i++)
            _frequency[i] = NORM(_iCol[i],
                                 _iRow[i])
                          / _para.size
                          / _para.pixelSize;

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            for (int i = 0; i < _nPxl; i++)
            {
                RFLOAT angle = atan2(_iRow[i],
                                     _iCol[i])
                             - _ctfAttr[l].defocusTheta;

                RFLOAT defocus = -(_ctfAttr[l].defocusU
                                 + _ctfAttr[l].defocusV
                                 + (_ctfAttr[l].defocusU - _ctfAttr[l].defocusV)
                                 * cos(2 * angle))
                                 / 2;

                _defocusP[pixelMajor
                        ? (i * _ID.size() + l)
                        : (_nPxl * l + i)] = defocus;
            }

            RFLOAT lambda = 12.2643274 / sqrt(_ctfAttr[l].voltage
                                            * (1 + _ctfAttr[l].voltage * 0.978466e-6));

            _K1[l] = M_PI * lambda;
            _K2[l] = M_PI / 2 * _ctfAttr[l].Cs * TSGSL_pow_3(lambda);
        }
    }
}

void Optimiser::freePreCalIdx()
{
    IF_MASTER return;

    delete[] _iPxl;
    delete[] _iCol;
    delete[] _iRow;
    delete[] _iSig;
}

void Optimiser::freePreCal(const bool ctf)
{
    IF_MASTER return;

    TSFFTW_free(_datP);
    TSFFTW_free(_ctfP);
    TSFFTW_free(_sigRcpP);

    /***
    delete[] _datP;
    delete[] _ctfP;
    delete[] _sigRcpP;
    ***/

    if (ctf)
    {
        delete[] _frequency;
        delete[] _defocusP;
        delete[] _K1;
        delete[] _K2;
    }
}

void Optimiser::saveDatabase() const
{
    IF_MASTER return;

    char filename[FILE_NAME_LENGTH];
    sprintf(filename, "%sMeta_Round_%03d.thu", _para.dstPrefix, _iter);

    bool flag;
    MPI_Status status;
    
    if (_commRank != 1)
        MPI_Recv(&flag, 1, MPI_C_BOOL, _commRank - 1, 0, MPI_COMM_WORLD, &status);

    FILE* file = (_commRank == 1)
               ? fopen(filename, "w")
               : fopen(filename, "a");

    size_t cls;
    vec4 quat;
    vec2 tran;
    RFLOAT df;

    // RFLOAT rVari, s0, s1, s;
    RFLOAT k1, k2, k3, s0, s1, s;

    FOR_EACH_2D_IMAGE
    {
        _par[l].rank1st(cls, quat, tran, df);

        //_par[l].vari(rVari, s0, s1, s);
        _par[l].vari(k1, k2, k3, s0, s1, s);

        /***
        rVari = 0;
        s0 = 0;
        s1 = 0;
        s = 0;
        ***/

        fprintf(file,
                "%18.9lf %18.9lf %18.9lf %18.9lf %18.9lf %18.9lf %18.9lf \
                 %s %s %18.9lf %18.9lf \
                 %6d %6lu \
                 %18.9lf %18.9lf %18.9lf %18.9lf \
                 %18.9lf %18.9lf %18.9lf \
                 %18.9lf %18.9lf %18.9lf %18.9lf \
                 %18.9lf %18.9lf \
                 %18.9lf\n",
                 _ctfAttr[l].voltage,
                 _ctfAttr[l].defocusU,
                 _ctfAttr[l].defocusV,
                 _ctfAttr[l].defocusTheta,
                 _ctfAttr[l].Cs,
                 _ctfAttr[l].amplitudeContrast,
                 _ctfAttr[l].phaseShift,
                 _db.path(_ID[l]).c_str(),
                 _db.micrographPath(_ID[l]).c_str(),
                 _db.coordX(_ID[l]),
                 _db.coordY(_ID[l]),
                 _groupID[l],
                 cls,
                 quat(0),
                 quat(1),
                 quat(2),
                 quat(3),
                 k1,
                 k2,
                 k3,
                 tran(0) - _offset[l](0),
                 tran(1) - _offset[l](1),
                 s0,
                 s1,
                 df,
                 s,
                 _par[l].compress());
    }

    fclose(file);

    if (_commRank != _commSize - 1)
        MPI_Send(&flag, 1, MPI_C_BOOL, _commRank + 1, 0, MPI_COMM_WORLD);
}

void Optimiser::saveBestProjections()
{
    IF_MASTER return;

    FFT fft;

    Image result(_para.size, _para.size, FT_SPACE);
    Image diff(_para.size, _para.size, FT_SPACE);
    char filename[FILE_NAME_LENGTH];

    size_t cls;
    mat22 rot2D;
    mat33 rot3D;
    vec2 tran;
    RFLOAT d;

    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            #pragma omp parallel for
            SET_0_FT(result);

            #pragma omp parallel for
            SET_0_FT(diff);

            if (_para.mode == MODE_2D)
            {
                _par[l].rank1st(cls, rot2D, tran, d);

                _model.proj(cls).projectMT(result, rot2D, tran);
            }
            else if (_para.mode == MODE_3D)
            {
                _par[l].rank1st(cls, rot3D, tran, d);

                _model.proj(cls).projectMT(result, rot3D, tran);
            }
            else
                REPORT_ERROR("INEXISTENT MODE");

            #pragma omp parallel for
            FOR_EACH_PIXEL_FT(diff)
                diff[i] = _img[l][i] - result[i] * REAL(_ctf[l][i]);

            sprintf(filename, "%sResult_%04d_Round_%03d.bmp", _para.dstPrefix, _ID[l], _iter);

            fft.bw(result);
            result.saveRLToBMP(filename);
            fft.fw(result);

            sprintf(filename, "%sDiff_%04d_Round_%03d.bmp", _para.dstPrefix, _ID[l], _iter);
            fft.bw(diff);
            diff.saveRLToBMP(filename);
            fft.fw(diff);
        }
    }
}

void Optimiser::saveImages()
{
    IF_MASTER return;

    FFT fft;

    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            sprintf(filename, "Fourier_Image_%04d.bmp", _ID[l]);

            _img[l].saveFTToBMP(filename, 0.01);

            sprintf(filename, "Image_%04d.bmp", _ID[l]);

            fft.bw(_img[l]);
            _img[l].saveRLToBMP(filename);
            fft.fw(_img[l]);
        }
    }
}

void Optimiser::saveCTFs()
{
    IF_MASTER return;

    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            sprintf(filename, "CTF_%04d.bmp", _ID[l]);

            _ctf[l].saveFTToBMP(filename, 0.01);
        }
    }
}

void Optimiser::saveMapHalf(const bool finished)
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    FFT fft;

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];

    for (int t = 0; t < _para.k; t++)
    {
        if (_para.mode == MODE_2D)
        {
            if (_commRank == HEMI_A_LEAD)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Saving Reference " << t;

                Image ref(_para.size,
                          _para.size,
                          FT_SPACE);

                SLC_EXTRACT_FT(ref, _model.ref(t), 0);

                /***
                if (finished)
                    sprintf(filename, "%sFT_Reference_%03d_A_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sFT_Reference_%03d_A_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveFTToBMP(filename, 0.001);
                ***/

                fft.bwMT(ref);

                if (finished)
                    sprintf(filename, "%sReference_%03d_A_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sReference_%03d_A_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveRLToBMP(filename);
            }
            else if (_commRank == HEMI_B_LEAD)
            {
                BLOG(INFO, "LOGGER_ROUND") << "Saving Reference " << t;

                Image ref(_para.size,
                          _para.size,
                          FT_SPACE);

                SLC_EXTRACT_FT(ref, _model.ref(t), 0);

                /***
                if (finished)
                    sprintf(filename, "%sFT_Reference_%03d_B_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sFT_Reference_%03d_B_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveFTToBMP(filename, 0.001);
                ***/

                fft.bwMT(ref);

                if (finished)
                    sprintf(filename, "%sReference_%03d_B_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sReference_%03d_B_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveRLToBMP(filename);
            }
        }
        else if (_para.mode == MODE_3D)
        {
            Volume lowPass(_para.size,
                           _para.size,
                           _para.size,
                           FT_SPACE);

            if (finished)
                fft.bwMT(_model.ref(t));
            else
            {
#ifdef OPTIMISER_SAVE_LOW_PASS_REFERENCE
                lowPassFilter(lowPass,
                              _model.ref(t),
                              (RFLOAT)_resReport / _para.size,
                              (RFLOAT)EDGE_WIDTH_FT / _para.size);
#else
                lowPass = _model.ref(t).copyVolume();
#endif

                fft.bwMT(lowPass);
            }

            if (_commRank == HEMI_A_LEAD)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Saving Reference " << t;

                if (finished)
                {
                    sprintf(filename, "%sReference_%03d_A_Final.mrc", _para.dstPrefix, t);

                    imf.readMetaData(_model.ref(t));
                    imf.writeVolume(filename, _model.ref(t), _para.pixelSize);

                    fft.fwMT(_model.ref(t));
                    _model.ref(t).clearRL();

                }
                else
                {
                    sprintf(filename, "%sReference_%03d_A_Round_%03d.mrc", _para.dstPrefix, t, _iter);

                    imf.readMetaData(lowPass);
                    imf.writeVolume(filename, lowPass, _para.pixelSize);
                }
            }
            else if (_commRank == HEMI_B_LEAD)
            {
                BLOG(INFO, "LOGGER_ROUND") << "Saving Reference " << t;

                if (finished)
                {
                    sprintf(filename, "%sReference_%03d_B_Final.mrc", _para.dstPrefix, t);

                    imf.readMetaData(_model.ref(t));
                    imf.writeVolume(filename, _model.ref(t), _para.pixelSize);

                    fft.fwMT(_model.ref(t));
                    _model.ref(t).clearRL();
                }
                else
                {
                    sprintf(filename, "%sReference_%03d_B_Round_%03d.mrc", _para.dstPrefix, t, _iter);

                    imf.readMetaData(lowPass);
                    imf.writeVolume(filename, lowPass, _para.pixelSize);
                }
            }
        }
    }
}

void Optimiser::saveMapJoin(const bool finished)
{
    FFT fft;

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];

    if (_para.mode == MODE_2D)
    {
        IF_MASTER
        {
            MLOG(INFO, "LOGGER_ROUND") << "Saving Stack of Reference(s)";

            if (finished)
                sprintf(filename, "%sReference_Final.mrcs", _para.dstPrefix);
            else
                sprintf(filename, "%sReference_Round_%03d.mrcs", _para.dstPrefix, _iter);

            imf.openStack(filename, _para.size, _para.k, _para.pixelSize);
        }

        for (int l = 0; l < _para.k; l++)
        {
            IF_MASTER
            {
                Image ref(_para.size, _para.size, FT_SPACE);

                Image A(_para.size, _para.size, FT_SPACE);
                Image B(_para.size, _para.size, FT_SPACE);

                MLOG(INFO, "LOGGER_ROUND") << "Receiving Reference " << l << " from Hemisphere A";

                MPI_Recv_Large(&A[0],
                               A.sizeFT(),
                               TS_MPI_DOUBLE_COMPLEX,
                               HEMI_A_LEAD,
                               l,
                               MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Receiving Reference " << l << " from Hemisphere B";

                MPI_Recv_Large(&B[0],
                               B.sizeFT(),
                               TS_MPI_DOUBLE_COMPLEX,
                               HEMI_B_LEAD,
                               l,
                               MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Averaging Two Hemispheres";
                FOR_EACH_PIXEL_FT(ref)
                    ref[i] = (A[i] + B[i]) / 2;

                fft.bwMT(ref);

                imf.writeStack(ref, l);
            }
            else
            {
                if ((_commRank == HEMI_A_LEAD) ||
                    (_commRank == HEMI_B_LEAD))
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Sending Reference "
                                                 << l
                                                 << " from Hemisphere A";

                    BLOG(INFO, "LOGGER_ROUND") << "Sending Reference "
                                                 << l
                                                 << " from Hemisphere B";

                    MPI_Ssend_Large(&_model.ref(l)[0],
                                    _model.ref(l).sizeFT(),
                                    TS_MPI_DOUBLE_COMPLEX,
                                    MASTER_ID,
                                    l,
                                    MPI_COMM_WORLD);
                }
            }
        }

        IF_MASTER imf.closeStack();
    }
    else if (_para.mode == MODE_3D)
    {
        // TODO
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

void Optimiser::saveFSC(const bool finished) const
{
    NT_MASTER return;

    char filename[FILE_NAME_LENGTH];

    if (finished)
        sprintf(filename, "%sFSC_Final.txt", _para.dstPrefix);
    else
        sprintf(filename, "%sFSC_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 1; i < _model.rU(); i++)
    {
        fprintf(file,
                "%05d   %10.6lf",
                i,
                1.0 / resP2A(i, _para.size, _para.pixelSize));
        for (int t = 0; t < _para.k; t++)
            fprintf(file,
                    "   %10.6lf",
                    (_model.fsc(t))(i));
        fprintf(file, "\n");
    }
    
    fclose(file);
}

void Optimiser::saveClassInfo(const bool finished) const
{
    NT_MASTER return;

    char filename[FILE_NAME_LENGTH];

    if (finished)
        sprintf(filename, "%sClass_Info_Final.txt", _para.dstPrefix);
    else
        sprintf(filename, "%sClass_Info_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int t = 0; t < _para.k; t++)
    {
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                t,
                1.0 / _model.resolutionA(t, _para.thresReportFSC),
                _cDistr(t));
    }

    fclose(file);
}

void Optimiser::saveSig() const
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    char filename[FILE_NAME_LENGTH];

    if (_commRank == HEMI_A_LEAD)
        sprintf(filename, "%sSig_A_Round_%03d.txt", _para.dstPrefix, _iter);
    else
        sprintf(filename, "%sSig_B_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 0; i < maxR(); i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _para.size, _para.pixelSize),
                _sig(_groupID[0] - 1, i));

    fclose(file);
}

void Optimiser::saveTau() const
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    char filename[FILE_NAME_LENGTH];

    if (_commRank == HEMI_A_LEAD)
        sprintf(filename, "%sTau_A_Round_%03d.txt", _para.dstPrefix, _iter);
    else if (_commRank == HEMI_B_LEAD)
        sprintf(filename, "%sTau_B_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 1; i < maxR() * _para.pf - 1; i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _para.size * _para.pf, _para.pixelSize),
                _model.tau(0)(i));

    fclose(file);
}


/**
 *  This function is add by huabin
 */
//Change by huabin doubleToRFLOAT
RFLOAT logDataVSPrior_m_huabin(const Complex* dat, const Complex* pri, const RFLOAT* ctf, const RFLOAT* sigRcp, const int m)
{

//Change by huabin doubleToRFLOAT
    RFLOAT result2 = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmpReal = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmpImag = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmp1Real = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmp1Imag = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmp2;
//Change by huabin doubleToRFLOAT
    RFLOAT tmp3;
    for (int i = 0; i < m; i++)
    {
            
        tmpReal = ctf[i] * pri[i].dat[0];
        tmpImag = ctf[i] * pri[i].dat[1];
        tmp1Real = dat[i].dat[0] - tmpReal;
        tmp1Imag = dat[i].dat[1] - tmpImag;

        tmp2 = tmp1Real * tmp1Real + tmp1Imag * tmp1Imag;
        tmp3 = tmp2 * sigRcp[i];
        
        result2 += tmp3;

    }

    return result2;
}

/**
 *  This function is add by huabin
 */
vec logDataVSPrior_m_n_huabin(const Complex* dat, const Complex* pri, const RFLOAT* ctf, const RFLOAT* sigRcp, const int n, const int m)
{


    vec result2 = vec::Zero(n);

//Change by huabin doubleToRFLOAT
    RFLOAT tmpCPMulReal  = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmpCPMulImag  = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmpDSubCPReal = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmpDSubCPImag = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmp1          = 0.0;
//Change by huabin doubleToRFLOAT
    RFLOAT tmp2          = 0.0;


    for(int i = 0; i < m; i ++)
    {
        for(int j = 0; j < n; j++)
        {
            int idx = i * n + j;

            tmpCPMulReal  = ctf[idx] * pri[i].dat[0];
            tmpCPMulImag  = ctf[idx] * pri[i].dat[1];

            tmpDSubCPReal = dat[idx].dat[0] - tmpCPMulReal; //temp.real
            tmpDSubCPImag = dat[idx].dat[1] - tmpCPMulImag;//temp.imag

            tmp1          = tmpDSubCPReal * tmpDSubCPReal + tmpDSubCPImag * tmpDSubCPImag; //tmp1
            tmp2          = tmp1 * sigRcp[idx];//temp2
            result2(j)    += tmp2;//result2
        }
    }
    
    return result2;
}

RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sigRcp,
                      const RFLOAT rU,
                      const RFLOAT rL)
{
    RFLOAT result = 0;

    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));
            if ((v < rU) &&
                (v >= rL))
            {
                int index = dat.iFTHalf(i, j);

                result += ABS2(dat.iGetFT(index)
                             - REAL(ctf.iGetFT(index))
                             * pri.iGetFT(index))
                        * sigRcp(v);
            }
        }
    }

    return result;
}

RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sigRcp,
                      const int* iPxl,
                      const int* iSig,
                      const int m)
{
    RFLOAT result = 0;

    for (int i = 0; i < m; i++)
        result += ABS2(dat.iGetFT(iPxl[i])
                     - REAL(ctf.iGetFT(iPxl[i]))
                     * pri.iGetFT(iPxl[i]))
                * sigRcp(iSig[i]);

    return result;
}

RFLOAT logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const RFLOAT* ctf,
                      const RFLOAT* sigRcp,
                      const int m)
{
    RFLOAT result = 0;

    for (int i = 0; i < m; i++)
        result += ABS2(dat[i] - ctf[i] * pri[i])
                * sigRcp[i];

    return result;
}

RFLOAT logDataVSPrior(const Complex* dat,
                      const Complex* pri,
                      const RFLOAT* frequency,
                      const RFLOAT* defocus,
                      const RFLOAT df,
                      const RFLOAT K1,
                      const RFLOAT K2,
                      const RFLOAT* sigRcp,
                      const int m)
{
    RFLOAT result = 0;

    for (int i = 0; i < m; i++)
    {
        RFLOAT ki = K1 * defocus[i] * df * TSGSL_pow_2(frequency[i])
                  + K2 * TSGSL_pow_4(frequency[i]);

        RFLOAT ctf = -w1 * sin(ki) + w2 * cos(ki);

        result += ABS2(dat[i] - ctf * pri[i])
                * sigRcp[i];

    }

    return result;
}

RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sigRcp,
                      const RFLOAT rU,
                      const RFLOAT rL)
{
    RFLOAT result = 0;

    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));
            if ((v < rU) &&
                (v >= rL))
            {
                int index = dat.iFTHalf(i, j);

                result += ABS2(dat.iGetFT(index)
                             - REAL(ctf.iGetFT(index))
                             * pri.iGetFT(index)
                             * tra.iGetFT(index))
                        * sigRcp(v);
            }
        }
    }

    return result;
}

RFLOAT logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sigRcp,
                      const int* iPxl,
                      const int* iSig,
                      const int m)
{
    RFLOAT result = 0;

    for (int i = 0; i < m; i++)
    {
        int index = iPxl[i];

        result += ABS2(dat.iGetFT(index)
                     - REAL(ctf.iGetFT(index))
                     * pri.iGetFT(index)
                     * tra.iGetFT(index))
                * sigRcp(iSig[i]);
    }

    return result;
}

vec logDataVSPrior(const vector<Image>& dat,
                   const Image& pri,
                   const vector<Image>& ctf,
                   const vector<int>& groupID,
                   const mat& sigRcp,
                   const RFLOAT rU,
                   const RFLOAT rL)
{
    int n = dat.size();

    vec result = vec::Zero(n);

    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));
            if ((v < rU) &&
                (v >= rL))
            {
                int index = dat[0].iFTHalf(i, j);

                for (int l = 0; l < n; l++)
                {
                    result(l) += ABS2(dat[l].iGetFT(index)
                                    - REAL(ctf[l].iGetFT(index))
                                    * pri.iGetFT(index))
                               * sigRcp(groupID[l] - 1, v);
                }
            }
        }
    }

    return result;
}

vec logDataVSPrior(const vector<Image>& dat,
                   const Image& pri,
                   const vector<Image>& ctf,
                   const vector<int>& groupID,
                   const mat& sigRcp,
                   const int* iPxl,
                   const int* iSig,
                   const int m)
{
    int n = dat.size();

    vec result = vec::Zero(n);

    for (int l = 0; l < n; l++)
    {
        int gL = groupID[l] - 1;

        const Image& datL = dat[l];
        const Image& ctfL = ctf[l];

        for (int i = 0; i < m; i++)
        {
            int index = iPxl[i];

            result(l) += ABS2(datL.iGetFT(index)
                            - REAL(ctfL.iGetFT(index))
                            * pri.iGetFT(index))
                       * sigRcp(gL, iSig[i]);
        }
    }

    return result;
}

/***
vec logDataVSPrior(const Complex* const* dat,
                   const Complex* pri,
                   const RFLOAT* const* ctf,
                   const RFLOAT* const* sigRcp,
                   const int n,
                   const int m)
{
    vec result = vec::Zero(n);

    for (int l = 0; l < n; l++)
        for (int i = 0; i < m; i++)
            result(l) += ABS2(dat[l][i]
                            - ctf[l][i]
                            * pri[i])
                       * sigRcp[l][i];
    
    return result;
}
***/

vec logDataVSPrior(const Complex* dat,
                   const Complex* pri,
                   const RFLOAT* ctf,
                   const RFLOAT* sigRcp,
                   const int n,
                   const int m)
{
    vec result = vec::Zero(n);

    // imageMajor

    /***
    for (int i = 0; i < n * m; i++)
        result(i / m) += ABS2(dat[i] - ctf[i] * pri[i % m])
#ifdef OPTIMISER_CTF_WRAP
                       * fabs(ctf[i])
#endif
                       * sigRcp[i];
    ***/

    // pixelMajor

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            size_t idx = i * n + j;

            result(j) += ABS2(dat[idx] - ctf[idx] * pri[i])
                       * sigRcp[idx];
        }

    return result;
}

RFLOAT dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sigRcp,
                   const RFLOAT rU,
                   const RFLOAT rL)
{
    return exp(logDataVSPrior(dat, pri, ctf, sigRcp, rU, rL));
}

RFLOAT dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& tra,
                   const Image& ctf,
                   const vec& sigRcp,
                   const RFLOAT rU,
                   const RFLOAT rL)
{
    return exp(logDataVSPrior(dat, pri, tra, ctf, sigRcp, rU, rL));
}

void scaleDataVSPrior(vec& sXA,
                      vec& sAA,
                      const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const RFLOAT rU,
                      const RFLOAT rL)
{
    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    for (int i = 0; i < rU; i++)
    {
        sXA(i) = 0;
        sAA(i) = 0;
    }
    
    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_PIXEL_R_FT(CEIL(rU) + 1)
    {
        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));
            if ((v < rU) &&
                (v >= rL))
            {
                int index = dat.iFTHalf(i, j);

                #pragma omp atomic
                sXA(v) += REAL(dat.iGetFT(index)
                             * pri.iGetFT(index)
                             * REAL(ctf.iGetFT(index)));

                #pragma omp atomic
                sAA(v) += REAL(pri.iGetFT(index)
                             * pri.iGetFT(index)
                             * TSGSL_pow_2(REAL(ctf.iGetFT(index))));
            }
        }
    }
}

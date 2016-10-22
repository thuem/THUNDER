/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Bing Li, Heng Guo
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "MLOptimiser.h"

void display(const MLOptimiserPara& para)
{
    printf("Number of Classes:                                     %12d\n", para.k);
    printf("Size of Image:                                         %12d\n", para.size);
    printf("Pixel Size (Angstrom):                                 %12.6lf\n", para.pixelSize); 
    printf("Radius of Mask on Images (Angstrom):                   %12.6lf\n", para.maskRadius);
    printf("Estimated Translation (Pixel):                         %12.6lf\n", para.transS);
    printf("Initial Resolution (Angstrom):                         %12.6lf\n", para.initRes);
    printf("Perform Global Search Under (Angstrom):                %12.6lf\n", para.globalSearchRes);
    printf("Symmetry:                                              %12s\n", para.sym);
    printf("Initial Model:                                         %12s\n", para.initModel);
    printf("Sqlite3 File Storing Paths and CTFs of Images:         %12s\n", para.db);
    
    printf("Perform Reference Mask:                                %12d\n", para.performMask);
    printf("Automask:                                              %12d\n", para.autoMask);
    printf("Mask:                                                  %12s\n", para.mask);

    printf("Perform Sharpening:                                    %12d\n", para.performSharpen);
    printf("Auto Estimate B-factor:                                %12d\n", para.estBFactor);
    printf("B-Factor:                                              %12.6lf\n", para.bFactor);

    printf("Max Number of Iteration:                               %12d\n", para.iterMax);
    printf("Padding Factor:                                        %12d\n", para.pf);
    printf("MKB Kernel Radius:                                     %12.6lf\n", para.a);
    printf("MKB Kernel Smooth Factor:                              %12.6lf\n", para.alpha);
    printf("Number of Sampling Points in Global Search:            %12d\n", para.mG);
    printf("Number of Sampling Points in Local Search:             %12d\n", para.mL);
    printf("Ignore Signal Under (Angstrom):                        %12.6lf\n", para.ignoreRes);
    printf("Correct Intensity Scale Using Signal Under (Angstrom): %12.6lf\n", para.sclCorRes);
    printf("FSC Threshold for Cutoff Frequency:                    %12.6lf\n", para.thresCutoffFSC);
    printf("FSC Threshold for Reporting Resolution:                %12.6lf\n", para.thresReportFSC);
    printf("Grouping when Calculating Sigma:                       %12d\n", para.groupSig);
    printf("Grouping when Correcting Intensity Scale:              %12d\n", para.groupScl);
    printf("Mask Images with Zero Noise:                           %12d\n", para.zeroMask);
}

MLOptimiser::MLOptimiser() {}

MLOptimiser::~MLOptimiser()
{
    clear();
}

MLOptimiserPara& MLOptimiser::para()
{
    return _para;
}

void MLOptimiser::setPara(const MLOptimiserPara& para)
{
    _para = para;
}

void MLOptimiser::init()
{
    MLOG(INFO, "LOGGER_INIT") << "Setting MPI Environment of _model";
    _model.setMPIEnv(_commSize, _commRank, _hemi);

    MLOG(INFO, "LOGGER_INIT") << "Setting up Symmetry";
    _sym.init(_para.sym);

    MLOG(INFO, "LOGGER_INIT") << "Passing Parameters to _model";
    _model.init(_para.k,
                _para.size,
                0,
                _para.pf,
                _para.pixelSize,
                _para.a,
                _para.alpha,
                &_sym);

    MLOG(INFO, "LOGGER_INIT") << "Setting Parameters: _r, _iter";

    _r = AROUND(resA2P(1.0 / _para.initRes, _para.size, _para.pixelSize)) + 1;
    _model.setR(_r);

    _iter = 0;

    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _para.ignoreRes
                              << " Angstrom will be Ingored during Comparison";

    _rL = resA2P(1.0 / _para.ignoreRes, _para.size, _para.pixelSize);

    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _rL
                              << " (Pixel) will be Ingored during Comparison";

    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _para.sclCorRes
                              << " Angstrom will be Used for Performing Intensity Scale Correction";

    _rS = AROUND(resA2P(1.0 / _para.sclCorRes, _para.size, _para.pixelSize)) + 1;

    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _rS
                              << " (Pixel) will be Used for Performing Intensity Scale Correction";

    MLOG(INFO, "LOGGER_INIT") << "Seting Frequency Upper Boudary during Global Search";

    _model.setRGlobal(AROUND(resA2P(1.0 / _para.globalSearchRes,
                             _para.size,
                             _para.pixelSize)) + 1);

    MLOG(INFO, "LOGGER_INIT") << "Global Search Resolution Limit : "
                              << _para.globalSearchRes
                              << " (Angstrom), "
                              << _model.rGlobal()
                              << " (Pixel)";

    MLOG(INFO, "LOGGER_INIT") << "Openning Database File";
    _exp.openDatabase(_para.db);

    MLOG(INFO, "LOGGER_INIT") << "Setting MPI Environment of _exp";
    _exp.setMPIEnv(_commSize, _commRank, _hemi);

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting ID of _exp";
    _exp.bcastID();

    MLOG(INFO, "LOGGER_INIT") << "Preparing Temporary File of _exp";
    _exp.prepareTmpFile();

    MLOG(INFO, "LOGGER_INIT") << "Scattering _exp";
    _exp.scatter();

    MLOG(INFO, "LOGGER_INIT") << "Appending Initial References into _model";
    initRef();

    MLOG(INFO, "LOGGER_INIT") << "Bcasting Total Number of 2D Images";
    bCastNPar();

    NT_MASTER
    {
        if (_para.performMask && !_para.autoMask)
        {
            ALOG(INFO, "LOGGER_INIT") << "Reading Mask";
            BLOG(INFO, "LOGGER_INIT") << "Reading Mask";

            initMask();
        }

        ALOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";
        BLOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";

        initID();

        ALOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";
        BLOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";

        allReduceN();

        ALOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere A: " << _N;
        BLOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere B: " << _N;

        initImg();

        ALOG(INFO, "LOGGER_INIT") << "Setting Parameters: _N";
        BLOG(INFO, "LOGGER_INIT") << "Setting Parameters: _N";

        ALOG(INFO, "LOGGER_INIT") << "Generating CTFs";
        BLOG(INFO, "LOGGER_INIT") << "Generating CTFs";

        initCTF();

        ALOG(INFO, "LOGGER_INIT") << "Initialising Switch";
        BLOG(INFO, "LOGGER_INIT") << "Initialising Switch";

        initSwitch();

        ALOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";
        BLOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";

        initParticles();
    }

    MLOG(INFO, "LOGGER_INIT") << "Broadacasting Information of Groups";

    bcastGroupInfo();

    NT_MASTER
    {
        ALOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";
        BLOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";

        _model.initProjReco();

        ALOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale";
        ALOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale";

        correctScale(true, false);

        ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";
        BLOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";

        initSigma();
    }
}

void MLOptimiser::expectation()
{
    IF_MASTER return;

    int nPxl;
    allocPreCal(nPxl, _r, _rL);

    int nPer = 0;

    if (_searchType == SEARCH_TYPE_GLOBAL)
    {
        // initialse a particle filter

        int nR = _para.mS;
        int nT = GSL_MAX_INT(100,
                             AROUND(M_PI
                                  * gsl_pow_2(_para.transS
                                            * gsl_cdf_chisq_Qinv(0.5, 2))
                                  * TRANS_SEARCH_FACTOR));

        Particle par;
        par.init(_para.transS, 0.01, &_sym);
        par.reset(nR, nT);

        // copy the particle filter to the rest of particle filters

        /***
        #pragma omp parallel for
        for (int i = 1; i < (int)_ID.size(); i++)
            _par[0].copy(_par[i]);
            ***/

        mat33 rot;
        vec2 t;

        // generate "translations"

        vector<Image> trans;
        trans.resize(nT);

        #pragma omp parallel for schedule(dynamic) private(t)
        for (unsigned int m = 0; m < (unsigned int)nT; m++)
        {
            trans[m].alloc(size(), size(), FT_SPACE);

            par.t(t, m);
                    
            translate(trans[m], _r, t(0), t(1));
        }
        
        // perform expectations

        //mat topW(_par[0].n(), _ID.size());

        mat topW(_para.mG, _ID.size());
        for (int i = 0; i < _para.mG; i++)
            for (int j = 0; j < (int)_ID.size(); j++)
                topW(i, j) = -DBL_MAX;

        umat iTopR(_para.mG, _ID.size());
        umat iTopT(_para.mG, _ID.size());

        _nR = 0;

        //#pragma omp parallel for schedule(dynamic) private(rot)
        for (unsigned int m = 0; m < (unsigned int)nR; m++)
        {
            Image imgRot(size(), size(), FT_SPACE);
            Image imgAll(size(), size(), FT_SPACE);

            // perform projection

            par.rot(rot, m * nT);

            //_model.proj(0).project(imgRot, rot);
            _model.proj(0).projectMT(imgRot, rot);

            for (unsigned int n = 0; n < (unsigned int)nT; n++)
            {
                // perform translation

                /***
                #pragma omp parallel for schedule(dynamic)
                IMAGE_FOR_EACH_PIXEL_FT(imgAll)
                ***/

                #pragma omp parallel for schedule(dynamic)
                IMAGE_FOR_PIXEL_R_FT(_r)
                {
                    if (QUAD(i, j) < gsl_pow_2(_r))
                    {
                        int index = imgAll.iFTHalf(i, j);
                        imgAll[index] = imgRot[index] * trans[n][index];
                    }
                }

                /***
                logW.row(m * nT + n).transpose() = logDataVSPrior(_img,
                                                                  imgAll,
                                                                  _ctf,
                                                                  _groupID,
                                                                  _sig,
                                                                  _r,
                                                                  _rL);
                                                                  ***/
                /***
                logW.row(m * nT + n).transpose() = logDataVSPrior(_img,
                                                                  imgAll,
                                                                  _ctf,
                                                                  _groupID,
                                                                  _sig,
                                                                  _iPxl,
                                                                  _iSig,
                                                                  nPxl);
                                                                  ***/

                vec dvp = logDataVSPrior(_img,
                                         imgAll,
                                         _ctf,
                                         _groupID,
                                         _sig,
                                         _iPxl,
                                         _iSig,
                                         nPxl);

                #pragma omp parallel for
                FOR_EACH_2D_IMAGE
                    recordTopK(topW.col(l).data(),
                               iTopR.col(l).data(),
                               iTopT.col(l).data(),
                               dvp(l),
                               m,
                               n,
                               _para.mG);


                /***
                #pragma omp parallel for
                FOR_EACH_2D_IMAGE
                {
                    logW(m * nT + n, l) = logDataVSPrior(_img[l],
                                                         imgAll,
                                                         _ctf[l],
                                                         _sig.row(_groupID[l] - 1).head(_r).transpose(),
                                                         _r);
                }
                ***/
            }

            //#pragma omp atomic
            _nR += 1;

            //#pragma omp critical
            if (_nR > (int)(nR / 10))
            {
                _nR = 0;

                nPer += 1;

                ALOG(INFO, "LOGGER_ROUND") << nPer * 10
                                           << "\% Initial Phase of Global Search Performed";
                BLOG(INFO, "LOGGER_ROUND") << nPer * 10
                                           << "\% Initial Phase of Global Search Performed";
            }
        }
        
        // process logW

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            //vec v = logW.col(l);
            vec v = topW.col(l);

            PROCESS_LOGW(v);

            topW.col(l) = v;
        }

        // reset weights of particle filter

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            /***
            for (int m = 0; m < _par[l].n(); m++)
                _par[l].mulW(logW(m, l), m);
            ***/

            _par[l].reset(_para.mG);

            vec4 quat;
            vec2 t;

            for (int m = 0; m < _para.mG; m++)
            {
                par.quaternion(quat, iTopR(m, l) * nT);
                par.t(t, iTopT(m, l));

                _par[l].setQuaternion(quat, m);
                _par[l].setT(t, m);

                _par[l].mulW(topW(m, l), m);
            }

            _par[l].normW();

            /***
            // sort
            _par[l].sort(_para.mG);
            ***/

            if (_ID[l] < 20)
            {
                char filename[FILE_NAME_LENGTH];
                snprintf(filename,
                         sizeof(filename),
                         "Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l]);
            }

            // shuffle
            _par[l].shuffle();

            // resample
            _par[l].resample(_para.mG);
        }

        ALOG(INFO, "LOGGER_ROUND") << "Initial Phase of Global Search Performed.";
        BLOG(INFO, "LOGGER_ROUND") << "Initial Phase of Global Search Performed.";
    }

    _nF = 0;
    _nI = 0;

    nPer = 0;

    #pragma omp parallel for schedule(dynamic)
    FOR_EACH_2D_IMAGE
    {
        Image image(size(), size(), FT_SPACE);

        // number of sampling for the next phase searching
        // int nSamplingNextPhase = 0;

        int nPhaseWithNoVariDecrease = 0;

        double tVariS0 = 5 * _para.transS;
        double tVariS1 = 5 * _para.transS;
        double rVari = 1;

        for (int phase = 0; phase < MAX_N_PHASE_PER_ITER; phase++)
        {
            /***
            int nR = 0;
            int nT = 0;
            ***/

            /***
            if (phase == 0)
            {
                if (_searchType == SEARCH_TYPE_GLOBAL)
                {
                    nR = _para.mG;
                    nT = GSL_MAX_INT(50,
                                     AROUND(M_PI
                                          * gsl_pow_2(_para.transS
                                                    * gsl_cdf_chisq_Qinv(0.5, 2))
                                          * TRANS_SEARCH_FACTOR));
                    
                    _par[l].reset(nR, nT);
                }
                else
                    _par[l].resample(_para.mL,
                                     ALPHA_LOCAL_SEARCH);
            }
            ***/

            if ((phase == 0) &&
                (_searchType == SEARCH_TYPE_LOCAL))
            {
                _par[l].resample(_para.mL,
                                 ALPHA_LOCAL_SEARCH);

                _par[l].perturb(PERTURB_FACTOR_L);
            }
            else
            {
                _par[l].perturb(PERTURB_FACTOR_S);
            }

            /***
            if ((_searchType == SEARCH_TYPE_LOCAL) &&
                (phase == 0))
            {
                // perturb with 5x confidence area
                _par[l].perturb(5);
            }
            else if (phase != 0)
            {
                // pertrub with 0.2x confidence area
                _par[l].perturb();
            }
            ***/

            vec logW(_par[l].n());
            mat33 rot;
            vec2 t;

            /***
            if ((_searchType == SEARCH_TYPE_GLOBAL) &&
                (phase == 0))
            {
                vector<Image> trans;
                trans.resize(nT);

                for (int m = 0; m < nT; m++)
                {
                    trans[m].alloc(size(), size(), FT_SPACE);

                    _par[l].t(t, m);
                    
                    translate(trans[m], _r, t(0), t(1));
                }

            }
            else
            {
            ***/
                for (int m = 0; m < _par[l].n(); m++)
                {
                    _par[l].rot(rot, m);
                    _par[l].t(t, m);
                    _model.proj(0).project(image, rot, t);

                    /***
                    logW(m) = logDataVSPrior(_img[l], // dat
                                             image, // pri
                                             _ctf[l], // ctf
                                             _sig.row(_groupID[l] - 1).head(_r).transpose(), // sig
                                             _r,
                                             _rL);
                                             ***/
                    logW(m) = logDataVSPrior(_img[l], // dat
                                             image, // pri
                                             _ctf[l], // ctf
                                             _sig.row(_groupID[l] - 1).head(_r).transpose(), // sig
                                             _iPxl,
                                             _iSig,
                                             nPxl);
                }
            /***
            }
            ***/

            /***
            logW.array() -= logW.maxCoeff(); // avoiding numerical error

            for (int m = 0; m < _par[l].n(); m++)
                _par[l].mulW(exp(logW(m)), m);
            ***/

                /***
                logW.array() -= logW.minCoeff();

                for (int m = 0; m < _par[l].n(); m++)
                    _par[l].mulW(logW(m), m);
                ***/

                /***
                logW.array() -= logW.maxCoeff();

                for (int m = 0; m < _par[l].n(); m++)
                    _par[l].mulW(logW(m) < -logThres ? 0 : logW(m) + logThres, m);
                ***/

            /***
            logW.array() -= logW.maxCoeff();
            logW.array() *= -1;
            logW.array() += 1;
            logW.array() = 1.0 / logW.array();
            logW.array() -= logW.minCoeff();
            ***/

            PROCESS_LOGW(logW);

            for (int m = 0; m < _par[l].n(); m++)
                _par[l].mulW(logW(m), m);

            /***
            for (int m = 0; m < _par[l].n(); m++)
                _par[l].mulW(1.0 / logW(m), m);
            ***/

            _par[l].normW();

            /***
            if ((_searchType == SEARCH_TYPE_GLOBAL) &&
                (phase == 0))
            {
                // sort
                _par[l].sort(_para.mG);

                // shuffle
                _par[l].shuffle();
            }
            ***/

            if (_ID[l] < 20)
            {
                char filename[FILE_NAME_LENGTH];
                snprintf(filename,
                         sizeof(filename),
                         "Particle_%04d_Round_%03d_%03d_%03d.par",
                         _ID[l],
                         _iter,
                         phase,
                         0);
                save(filename, _par[l]);
            }

            // Only after resampling, the current variance can be calculated
            // correctly.

            if (_searchType == SEARCH_TYPE_GLOBAL)
                _par[l].resample(_para.mG);
            else
                _par[l].resample(_para.mL);
            
            if (phase >= MIN_N_PHASE_PER_ITER)
            {
                double tVariS0Cur;
                double tVariS1Cur;
                double rVariCur;
                _par[l].vari(rVariCur, tVariS0Cur, tVariS1Cur);

                /***
                CLOG(INFO, "LOGGER_SYS") << "phase = " << phase;
                CLOG(INFO, "LOGGER_SYS") << "tVariS0 = " << tVariS0;
                CLOG(INFO, "LOGGER_SYS") << "tVariS1 = " << tVariS1;
                CLOG(INFO, "LOGGER_SYS") << "rVari = " << rVari;
                CLOG(INFO, "LOGGER_SYS") << "tVariS0Cur = " << tVariS0Cur;
                CLOG(INFO, "LOGGER_SYS") << "tVariS1Cur = " << tVariS1Cur;
                CLOG(INFO, "LOGGER_SYS") << "rVariCur = " << rVariCur;
                ***/

                if ((tVariS0Cur < tVariS0 * 0.9) ||
                    (tVariS1Cur < tVariS1 * 0.9) ||
                    (rVariCur < rVari * 0.9))
                {
                    // there is still room for searching
                    nPhaseWithNoVariDecrease = 0;
                }
                else
                {
                    // there is no improvement in this search
                    nPhaseWithNoVariDecrease += 1;
                }

                // make tVariS0, tVariS1, rVari the smallest variance ever got
                if (tVariS0Cur < tVariS0) tVariS0 = tVariS0Cur;
                if (tVariS1Cur < tVariS1) tVariS1 = tVariS1Cur;
                if (rVariCur < rVari) rVari = rVariCur;

                // break if in a few continuous searching, there is no improvement
                if (nPhaseWithNoVariDecrease == 3)
                {
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

        if (_ID[l] < 20)
        {
            char filename[FILE_NAME_LENGTH];
            snprintf(filename,
                     sizeof(filename),
                     "Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, _par[l]);
        }
    }

    freePreCal();
}

void MLOptimiser::maximization()
{
    if ((_searchType == SEARCH_TYPE_GLOBAL) &&
        (_para.groupScl))
    {
        ALOG(INFO, "LOGGER_ROUND") << "Re-balancing Intensity Scale for Each Group";
        BLOG(INFO, "LOGGER_ROUND") << "Re-balancing Intensity Scale for Each Group";

        correctScale(false, true);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Generate Sigma for the Next Iteration";
    BLOG(INFO, "LOGGER_ROUND") << "Generate Sigma for the Next Iteration";

    allReduceSigma(_para.groupSig);

    ALOG(INFO, "LOGGER_ROUND") << "Reconstruct Reference";
    BLOG(INFO, "LOGGER_ROUND") << "Reconstruct Reference";

    reconstructRef(_para.performMask);
}

void MLOptimiser::run()
{
    IF_MASTER display(_para);

    MLOG(INFO, "LOGGER_ROUND") << "Initialising MLOptimiser";

    init();

    MLOG(INFO, "LOGGER_ROUND") << "Saving Some Data";
    
    saveImages();
    saveBinImages();
    saveCTFs();
    saveLowPassImages();

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Entering Iteration";
    for (_iter = 0; _iter < _para.iterMax; _iter++)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter;

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type : Global Search";
        }
        else if (_searchType == SEARCH_TYPE_LOCAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type : Local Search";
        }
        else
        {
            MLOG(INFO, "LOGGER_ROUND") << "Search Type : Stop Search";
            MLOG(INFO, "LOGGER_ROUND") << "Exitting Searching";

            break;
        }

        MPI_Barrier(MPI_COMM_WORLD);

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

        MLOG(INFO, "LOGGER_ROUND") << "Saving Best Projections";
        saveBestProjections();

        MLOG(INFO, "LOGGER_ROUND") << "Calculating Variance of Rotation and Translation";

        refreshVariance();

        MLOG(INFO, "LOGGER_ROUND") << "Rotation Variance : " << _model.rVari();

        MLOG(INFO, "LOGGER_ROUND") << "Translation Variance : " << _model.tVariS0()
                                   << ", " << _model.tVariS1();

        MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Rotation Variance : "
                                   << _model.stdRVari();

        MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Translation Variance : "
                                   << _model.stdTVariS0()
                                   << ", "
                                   << _model.stdTVariS1();

        MLOG(INFO, "LOGGER_ROUND") << "Calculating Changes of Rotation between Iterations";
        refreshRotationChange();

        MLOG(INFO, "LOGGER_ROUND") << "Determining Which Images Unsuited for Reconstruction";
        refreshSwitch();

        MLOG(INFO, "LOGGER_ROUND") << "Average Rotation Change : " << _model.rChange();
        MLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Rotation Change : "
                                   << _model.stdRChange();

        /***
        NT_MASTER
        {
            _model.allReduceRChange(_par, _N);

            ALOG(INFO, "LOGGER_ROUND") << "Average Rotation Change : " << _model.rChange();
            BLOG(INFO, "LOGGER_ROUND") << "Average Rotation Change : " << _model.rChange();

            ALOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Rotation Change : "
                                       << _model.stdRChange();
            BLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Rotation Change : "
                                       << _model.stdRChange();
        }
        ***/

        MLOG(INFO, "LOGGER_ROUND") << "Calculating Tau";
        NT_MASTER
        {
            _model.refreshTau();
        }

        MLOG(INFO, "LOGGER_ROUND") << "Performing Maximization";
        maximization();

        MLOG(INFO, "LOGGER_ROUND") << "Saving Reference(s)";
        saveReference();

        MLOG(INFO, "LOGGER_ROUND") << "Calculating FSC(s)";
        _model.BcastFSC();

        MLOG(INFO, "LOGGER_ROUND") << "Calculating SNR(s)";
        _model.refreshSNR();

        MLOG(INFO, "LOGGER_ROUND") << "Saving FSC(s)";
        saveFSC();

        MLOG(INFO, "LOGGER_ROUND") << "Current Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1,
                                                   _para.size,
                                                   _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Recording Current Resolution";

        _resReport = _model.resolutionP(_para.thresReportFSC, false);

        MLOG(INFO, "LOGGER_ROUND") << "Current Resolution (Report): "
                                   << _resReport
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_resReport, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        _model.setRes(_resReport);

        _resCutoff = _model.resolutionP(_para.thresCutoffFSC, true);

        MLOG(INFO, "LOGGER_ROUND") << "Current Resolution (Cutoff): "
                                   << _resCutoff
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_resCutoff, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Updating Cutoff Frequency in Model";

        _model.updateR(_para.thresCutoffFSC);

        MLOG(INFO, "LOGGER_ROUND") << "Increasing Cutoff Frequency or Not: "
                                   << _model.increaseR()
                                   << ", as the Rotation Change is "
                                   << _model.rChange()
                                   << " and the Previous Rotation Change is "
                                   << _model.rChangePrev();

        if (_model.r() > _model.rT())
        {
            MLOG(INFO, "LOGGER_ROUND") << "Resetting Parameters Determining Increase Frequency";

            _model.resetRChange();
            _model.setNRChangeNoDecrease(0);
            _model.setNTopResNoImprove(0);
            _model.setIncreaseR(false);

            MLOG(INFO, "LOGGER_ROUND") << "Recording Current Highest Frequency";

            _model.setRT(_model.r());
        }

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

        MLOG(INFO, "LOGGER_ROUND") << "Updating Cutoff Frequency";
        _r = _model.r();

        MLOG(INFO, "LOGGER_ROUND") << "New Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        NT_MASTER
        {
            ALOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors";
            BLOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors";

            _model.refreshProj();

            ALOG(INFO, "LOGGER_ROUND") << "Refreshing Reconstructors";
            BLOG(INFO, "LOGGER_ROUND") << "Refreshing Reconstructors";

            _model.refreshReco();
        }
    }

    MLOG(INFO, "LOGGER_ROUND") << "Preparing to Reconstruct Reference(s) at Nyquist";

    MLOG(INFO, "LOGGER_ROUND") << "Resetting to Nyquist Limit";
    _model.setRU(maxR());

    MLOG(INFO, "LOGGER_ROUND") << "Refreshing Reconstructors";
    NT_MASTER _model.refreshReco();

    MLOG(INFO, "LOGGER_ROUND") << "Reconstructing References(s) at Nyquist";
    reconstructRef(_para.performMask);

    MLOG(INFO, "LOGGER_ROUND") << "Saving Final Reference(s)";
    saveReference(true);

    MLOG(INFO, "LOGGER_ROUND") << "Calculating Final FSC(s)";
    _model.BcastFSC();

    MLOG(INFO, "LOGGER_ROUND") << "Saving Final FSC(s)";
    saveFSC(true);

    if (_para.performSharpen)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Sharpening Reference(s)";

        if (_para.estBFactor)
            _model.sharpenUp(true);
        else
            _model.sharpenUp(-_para.bFactor / gsl_pow_2(_para.pixelSize), true);

        MLOG(INFO, "LOGGER_ROUND") << "Saving Sharp Reference(s)";
        saveSharpReference();
    }
}

void MLOptimiser::clear()
{
    _img.clear();
    _par.clear();
    _ctf.clear();
}

void MLOptimiser::bCastNPar()
{
    IF_MASTER _nPar = _exp.nParticle();

    MPI_Bcast(&_nPar, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
}

void MLOptimiser::allReduceN()
{
    IF_MASTER return;

    _N = _exp.nParticle();

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_N, 1, MPI_INT, MPI_SUM, _hemi);

    MPI_Barrier(_hemi);
}

int MLOptimiser::size() const
{
    return _para.size;
}

int MLOptimiser::maxR() const
{
    return size() / 2 - 1;
}

void MLOptimiser::bcastGroupInfo()
{
    ALOG(INFO, "LOGGER_INIT") << "Storing GroupID";

    _groupID.clear();

    NT_MASTER
    {
        sql::Statement stmt("select GroupID from particles where ID = ?", -1, _exp.expose());
        FOR_EACH_2D_IMAGE
        {
            stmt.bind_int(1, _ID[l]);
            while (stmt.step())
                _groupID.push_back(stmt.get_int(0));
            stmt.reset();
        }
    }

    MLOG(INFO, "LOGGER_INIT") << "Getting Number of Groups from Database";
    IF_MASTER _nGroup = _exp.nGroup();

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting Number of Groups";
    MPI_Bcast(&_nGroup, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Number of Group: " << _nGroup;

    ALOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Sigma";
    NT_MASTER _sig.resize(_nGroup, maxR() + 1);

    ALOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Intensity Scale";
    NT_MASTER _scale.resize(_nGroup);
}

void MLOptimiser::initRef()
{
    _model.appendRef(Volume());

    MLOG(INFO, "LOGGER_INIT") << "Read Initial Model from Hard-disk";

    Volume ref;

    ImageFile imf(_para.initModel, "rb");
    imf.readMetaData();
    imf.readVolume(ref);

    if ((ref.nColRL() != _para.size) ||
        (ref.nRowRL() != _para.size) ||
        (ref.nSlcRL() != _para.size))
    {
        CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Appending Reference"
                                  << ": size = " << _para.size
                                  << ", nCol = " << ref.nColRL()
                                  << ", nRow = " << ref.nRowRL()
                                  << ", nSlc = " << ref.nSlcRL();

        __builtin_unreachable();
    }
    
    MLOG(INFO, "LOGGER_INIT") << "Padding Initial Model";

    #pragma omp parallel for
    FOR_EACH_PIXEL_RL(ref)
        if (ref(i) < 0) ref(i) = 0;

    VOL_PAD_RL(_model.ref(0), ref, _para.pf);

    MLOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform";

    FFT fft;
    fft.fwMT(_model.ref(0));
    _model.ref(0).clearRL();
}

void MLOptimiser::initMask()
{
    ImageFile imf(_para.mask, "rb");
    imf.readMetaData();
    imf.readVolume(_mask);
}

void MLOptimiser::initID()
{
    _ID.clear();

    sql::Statement stmt("select ID from particles;", -1, _exp.expose());

    while (stmt.step())
        _ID.push_back(stmt.get_int(0));
}

void MLOptimiser::initImg()
{
    _img.clear();
    _img.resize(_ID.size());

    string imgName;

    sql::Statement stmt("select Name from particles where ID = ?", -1, _exp.expose());

    FOR_EACH_2D_IMAGE
    {
        stmt.bind_int(1, _ID[l]);
        // ILOG(INFO) << "Read 2D Image ID of Which is " << _ID[i];
        // get the filename of the image from database
        if (stmt.step())
            imgName = stmt.get_text(0);
        else
            CLOG(FATAL, "LOGGER_SYS") << "Database Changed";

        stmt.reset();

	    //Image& currentImg = _img[l];

        // read the image fromm hard disk
        if (imgName.find('@') == string::npos)
        {
            ImageFile imf(imgName.c_str(), "rb");
            imf.readMetaData();
            //imf.readImage(currentImg);
            imf.readImage(_img[l]);
        }
        else
        {
            int nSlc = atoi(imgName.substr(0, imgName.find('@')).c_str()) - 1;
            string filename = imgName.substr(imgName.find('@') + 1);

            ImageFile imf(filename.c_str(), "rb");
            imf.readMetaData();
            //imf.readImage(currentImg, nSlc);
            imf.readImage(_img[l], nSlc);
        }

        /***
        if ((currentImg.nColRL() != _para.size) ||
            (currentImg.nRowRL() != _para.size))
            ***/
        if ((_img[l].nColRL() != _para.size) ||
            (_img[l].nRowRL() != _para.size))
        {
            CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of 2D Images";
            __builtin_unreachable();
        }
    }

    ALOG(INFO, "LOGGER_INIT") << "Substructing Mean of Noise, Making the Noise Have Zero Mean";
    BLOG(INFO, "LOGGER_INIT") << "Substructing Mean of Noise, Making the Noise Have Zero Mean";

    substractBgImg();

    ALOG(INFO, "LOGGER_INIT") << "Performing Statistics of 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Performing Statistics of 2D Images";

    statImg();

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images Before Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images Before Normalising";

    displayStatImg();

    ALOG(INFO, "LOGGER_INIT") << "Masking on 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Masking on 2D Images";

    maskImg();

    ALOG(INFO, "LOGGER_INIT") << "Normalising 2D Images, Making the Noise Have Standard Deviation of 1";
    BLOG(INFO, "LOGGER_INIT") << "Normalising 2D Images, Making the Noise Have Standard Deviation of 1";

    normaliseImg();

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";

    displayStatImg();

    /***
    statImg();

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";

    displayStatImg();
    ***/

    ALOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform on 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform on 2D Images";

    fwImg();
}

void MLOptimiser::statImg()
{
    _stdN = 0;
    _stdD = 0;
    _stdS = 0;
    
    _stdStdN = 0;

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        #pragma omp atomic
        _stdN += bgStddev(0,
                         _img[l],
                         //size() * MASK_RATIO / 2);
                         _para.maskRadius / _para.pixelSize);

        #pragma omp atomic
        _stdD += stddev(0, _img[l]);

        #pragma omp atomic
        _stdStdN += gsl_pow_2(bgStddev(0,
                                       _img[l],
                                       //size() * MASK_RATIO / 2));
                                       _para.maskRadius / _para.pixelSize));
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdN, 1, MPI_DOUBLE, MPI_SUM, _hemi);
    MPI_Allreduce(MPI_IN_PLACE, &_stdD, 1, MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdStdN, 1, MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Barrier(_hemi);

    _stdN /= _N;
    _stdD /= _N;

    _stdStdN /= _N;

    _stdS = _stdD - _stdN;

    _stdStdN = sqrt(_stdStdN - gsl_pow_2(_stdN));
}

void MLOptimiser::displayStatImg()
{
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Noise  : " << _stdN;
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Data   : " << _stdD;
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Signal : " << _stdS;

    ALOG(INFO, "LOGGER_INIT") << "Standard Devation of Standard Deviation of Noise : "
                              << _stdStdN;

    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Noise  : " << _stdN;
    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Data   : " << _stdD;
    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Signal : " << _stdS;

    BLOG(INFO, "LOGGER_INIT") << "Standard Devation of Standard Deviation of Noise : "
                              << _stdStdN;
}

void MLOptimiser::substractBgImg()
{
    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        double bg = background(_img[l],
                               _para.maskRadius / _para.pixelSize,
                               //size() * MASK_RATIO / 2,
                               EDGE_WIDTH_RL);

        FOR_EACH_PIXEL_RL(_img[l])
            _img[l](i) -= bg;
    }
}

void MLOptimiser::maskImg()
{
    /***
    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
        softMask(_img[l],
                 _img[l],
                 _para.maskRadius / _para.pixelSize,
                 //size() * MASK_RATIO / 2,
                 EDGE_WIDTH_RL,
                 0,
                 0);
                 ***/
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
}

void MLOptimiser::normaliseImg()
{
    double scale = 1.0 / _stdN;

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
        SCALE_RL(_img[l], scale);

    _stdN *= scale;
    _stdD *= scale;
    _stdS *= scale;
}

void MLOptimiser::fwImg()
{
    // perform Fourier transform
    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        FFT fft;
        fft.fw(_img[l]);
        _img[l].clearRL();
    }
}

void MLOptimiser::bwImg()
{
    // perform inverse Fourier transform
    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        FFT fft;
        fft.bw(_img[l]);
        _img[l].clearFT();
    }
}

void MLOptimiser::initCTF()
{
    IF_MASTER return;

    _ctf.clear();

    // get CTF attributes from _exp
    CTFAttr ctfAttr;

    sql::Statement stmt(
            "select Voltage, DefocusU, DefocusV, DefocusAngle, CS from \
             micrographs, particles where \
             particles.micrographID = micrographs.ID and \
             particles.ID = ?;",
             -1,
             _exp.expose());

    FOR_EACH_2D_IMAGE
    {
        // get attributes of CTF from database
        stmt.bind_int(1, _ID[l]);

        if (stmt.step())
        {
            ctfAttr.voltage = stmt.get_double(0);
            ctfAttr.defocusU = stmt.get_double(1);
            ctfAttr.defocusV = stmt.get_double(2);
            ctfAttr.defocusAngle = stmt.get_double(3);
            ctfAttr.CS = stmt.get_double(4);
        }
        else 
        {
            CLOG(FATAL, "LOGGER_SYS") << "No Data";

            __builtin_unreachable();
        }

        stmt.reset();

        // append a CTF
        _ctf.push_back(Image(size(), size(), FT_SPACE));

        // initialise the CTF according to attributes given
        CTF(_ctf.back(),
            _para.pixelSize,
            ctfAttr.voltage,
            ctfAttr.defocusU,
            ctfAttr.defocusV,
            ctfAttr.defocusAngle,
            ctfAttr.CS);
    }
}

void MLOptimiser::initSwitch()
{
    IF_MASTER return;

    _switch.clear();
    _switch.resize(_ID.size());
}

void MLOptimiser::correctScale(const bool init,
                               const bool group)
{
    IF_MASTER return;

    refreshScale(init, group);

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        FOR_EACH_PIXEL_FT(_img[l])
            _img[l][i] /= _scale(_groupID[l] - 1);
    }

    if (!init)
    {
        #pragma omp parallel for
        for (int i = 0; i < _nGroup; i++)
            _sig.row(i) /= gsl_pow_2(_scale(i));
    }
}

void MLOptimiser::initSigma()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Image";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Average Image";

    Image avg = _img[0].copyImage();

    for (size_t l = 1; l < _ID.size(); l++)
    {
        #pragma omp parallel for
        ADD_FT(avg, _img[l]);
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &avg[0],
                  avg.sizeFT(),
                  MPI_DOUBLE_COMPLEX,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    #pragma omp parallel for
    SCALE_FT(avg, 1.0 / _N);

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";

    vec avgPs = vec::Zero(maxR());

    FOR_EACH_2D_IMAGE
    {
        vec ps(maxR());

        powerSpectrum(ps, _img[l], maxR());

        avgPs += ps;
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs.data(),
                  maxR(),
                  MPI_DOUBLE,
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
                               [](const Complex x)
                               {
                                   return REAL(x) + IMAG(x);
                               });
        psAvg(i) = gsl_pow_2(psAvg(i));
    }

    // avgPs -> average power spectrum
    // psAvg -> expectation of pixels
    ALOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";
    BLOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";

    _sig.leftCols(_sig.cols() - 1).rowwise() = (avgPs - psAvg).transpose() / 2;

    /***
    ALOG(INFO) << "Saving Initial Sigma";
    if (_commRank == HEMI_A_LEAD)
        _sig.save("Sigma_000.txt", raw_ascii);
        ***/
}

void MLOptimiser::initParticles()
{
    IF_MASTER return;

    _par.clear();
    _par.resize(_ID.size());

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
        _par[l].init(_para.transS,
                     0.01,
                     &_sym);
}

void MLOptimiser::refreshRotationChange()
{
    vec rc = vec::Zero(_nPar);

    NT_MASTER
    {
        FOR_EACH_2D_IMAGE
            rc(_ID[l] - 1) = _par[l].diffTopR();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  rc.data(),
                  rc.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);

    double mean, std;
    stat_MAS(mean, std, rc, _nPar);

    _model.setRChange(mean);
    _model.setStdRChange(std);
}

void MLOptimiser::refreshVariance()
{
    vec rv = vec::Zero(_nPar);
    vec t0v = vec::Zero(_nPar);
    vec t1v = vec::Zero(_nPar);

    NT_MASTER
    {
        double rVari, tVariS0, tVariS1;

        #pragma omp parallel for private(rVari, tVariS0, tVariS1)
        FOR_EACH_2D_IMAGE
        {
            _par[l].vari(rVari,
                         tVariS0,
                         tVariS1);

            rv(_ID[l] - 1) = rVari;
            t0v(_ID[l] - 1) = tVariS0;
            t1v(_ID[l] - 1) = tVariS1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  rv.data(),
                  rv.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Allreduce(MPI_IN_PLACE,
                  t0v.data(),
                  t0v.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Allreduce(MPI_IN_PLACE,
                  t1v.data(),
                  t1v.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);

    double mean, std;

    stat_MAS(mean, std, rv, _nPar);

    _model.setRVari(mean);
    _model.setStdRVari(std);

    stat_MAS(mean, std, t0v, _nPar);

    _model.setTVariS0(mean);
    _model.setStdTVariS0(std);

    stat_MAS(mean, std, t0v, _nPar);

    _model.setTVariS1(mean);
    _model.setStdTVariS1(std);
}

void MLOptimiser::refreshSwitch()
{
    int nFail = 0;

    NT_MASTER
    {
        double rVariThres = _model.rVari()
                          + SWITCH_FACTOR
                          * _model.stdRVari();

        double tVariS0Thres = _model.tVariS0()
                            + SWITCH_FACTOR
                            * _model.stdTVariS0();

        double tVariS1Thres = _model.tVariS1() +
                            + SWITCH_FACTOR
                            * _model.stdTVariS1();

        double rVari, tVariS0, tVariS1;

        #pragma omp parallel for private(rVari, tVariS0, tVariS1)
        FOR_EACH_2D_IMAGE
        {
            _par[l].vari(rVari,
                         tVariS0,
                         tVariS1);

            if ((rVari > rVariThres) ||
                (tVariS0 > tVariS0Thres) ||
                (tVariS1 > tVariS1Thres))
            {
                _switch[l] = false;
                #pragma omp atomic
                nFail += 1;
            }
            else
                _switch[l] = true;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  &nFail,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << (double)nFail / _nPar * 100
                               << "\% of Images are Unsuited for Reconstruction";
}

void MLOptimiser::refreshScale(const bool init,
                               const bool group)
{
    IF_MASTER return;

    if (_rS > _r) MLOG(FATAL, "LOGGER_SYS") << "_rS is Larger than _r";

    mat mXA = mat::Zero(_nGroup, _rS);
    mat mAA = mat::Zero(_nGroup, _rS);

    vec sXA = vec::Zero(_rS);
    vec sAA = vec::Zero(_rS);

    Image img(size(), size(), FT_SPACE);

    mat33 rot;
    vec2 tran;

    FOR_EACH_2D_IMAGE
    {
        if (init)
        {
            randRotate3D(rot);

            _model.proj(0).projectMT(img, rot);
        }
        else
        {
            if (!_switch[l]) continue;

            _par[l].rank1st(rot, tran);

            _model.proj(0).projectMT(img, rot, tran);
        }

        scaleDataVSPrior(sXA,
                         sAA,
                         _img[l],
                         img,
                         _ctf[l],
                         _rS,
                         _rL);

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

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_ROUND") << "Accumulating Intensity Scale Information";
    BLOG(INFO, "LOGGER_ROUND") << "Accumulating Intensity Scale Information";

    MPI_Allreduce(MPI_IN_PLACE,
                  mXA.data(),
                  mXA.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  mAA.data(),
                  mAA.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    if (group)
    {
        for (int i = 0; i < _nGroup; i++)
        {
            double sum = 0;
            int count = 0;

            for (int r = 0; r < _rS; r++)
                if (r > _rL)
                {
                    sum += mXA(i, r) / mAA(i, r);
                    count += 1;
                }

            _scale(i) = sum / count;
        }
    }
    else
    {
        double sum = 0;
        int count = 0;

        for (int r = 0; r < _rS; r++)
            if (r > _rL)
            {
                sum += mXA(0, r) / mAA(0, r);
                count += 1;
            }
        
        for (int i = 0; i < _nGroup; i++)
            _scale(i) = sum / count;
    }
    
    ALOG(INFO, "LOGGER_ROUND") << "Average Intensity Scale: " << _scale.mean();
    BLOG(INFO, "LOGGER_ROUND") << "Average Intensity Scale: " << _scale.mean();

    ALOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Intensity Scale: "
                               << gsl_stats_sd(_scale.data(), 1, _scale.size());
    BLOG(INFO, "LOGGER_ROUND") << "Standard Deviation of Intensity Scale: "
                               << gsl_stats_sd(_scale.data(), 1, _scale.size());

    /***
        for (int i = 0; i < _nGroup; i++)
            CLOG(INFO, "LOGGER_ROUND") << "Group "
                                       << i
                                       << ": Scale = "
                                       << _scale(i);
                                       ***/
}

void MLOptimiser::allReduceSigma(const bool group)
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_ROUND") << "Clearing Up Sigma";
    BLOG(INFO, "LOGGER_ROUND") << "Clearing Up Sigma";

    // set re-calculating part to zero
    _sig.leftCols(_r).setZero();
    _sig.rightCols(1).setZero();

    ALOG(INFO, "LOGGER_ROUND") << "Recalculating Sigma";
    BLOG(INFO, "LOGGER_ROUND") << "Recalculating Sigma";

    Image img(size(), size(), FT_SPACE);

    mat33 rot;
    vec2 tran;

    FOR_EACH_2D_IMAGE
    {
        if (!_switch[l]) continue;

        vec sig(_r);

        _par[l].rank1st(rot, tran);

        // calculate differences

        _model.proj(0).projectMT(img, rot, tran);

        #pragma omp parallel for
        FOR_EACH_PIXEL_FT(img)
            img[i] *= REAL(_ctf[l][i]);

        #pragma omp parallel for
        NEG_FT(img);
        #pragma omp parallel for
        ADD_FT(img, _img[l]);

        powerSpectrum(sig, img, _r);

        if (group)
        {
            _sig.row(_groupID[l] - 1).head(_r) += sig.transpose() / 2;

            _sig(_groupID[l] - 1, _sig.cols() - 1) += 1;
        }
        else
        {
            _sig.row(0).head(_r) += sig.transpose() / 2;

            _sig(0, _sig.cols() - 1) += 1;
        }
    }

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_ROUND") << "Averaging Sigma of Images Belonging to the Same Group";
    BLOG(INFO, "LOGGER_ROUND") << "Averaging Sigma of Images Belonging to the Same Group";

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.data(),
                  _r * _nGroup,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.col(_sig.cols() - 1).data(),
                  _nGroup,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    if (group)
    {
        for (int i = 0; i < _sig.rows(); i++)
            _sig.row(i).head(_r) /= _sig(i, _sig.cols() - 1);
    }
    else
    {
        _sig.row(0).head(_r) /= _sig(0, _sig.cols() - 1);

        for (int i = 1; i < _sig.rows(); i++)
            _sig.row(i).head(_r) = _sig.row(0).head(_r);
    }
}

void MLOptimiser::reconstructRef(const bool mask)
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_ROUND") << "Inserting High Probability 2D Images into Reconstructor";
    BLOG(INFO, "LOGGER_ROUND") << "Inserting High Probability 2D Images into Reconstructor";

    FOR_EACH_2D_IMAGE
    {
        if (!_switch[l]) continue;

        mat33 rot;
        vec2 tran;
        
        _par[l].rank1st(rot, tran);

        _model.reco(0).insert(_img[l], _ctf[l], rot, tran, 1);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Reconstructing References for Next Iteration";
    BLOG(INFO, "LOGGER_ROUND") << "Reconstructing References for Next Iteration";

    _model.reco(0).reconstruct(_model.ref(0));

    if (_genMask)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Generating Automask";
        BLOG(INFO, "LOGGER_ROUND") << "Generating Automask";

        _mask.alloc(_para.pf * _para.size,
                    _para.pf * _para.size,
                    _para.pf * _para.size,
                    RL_SPACE);

        Volume lowPassRef(_para.pf * _para.size,
                          _para.pf * _para.size,
                          _para.pf * _para.size,
                          RL_SPACE);

        lowPassRef = _model.ref(0).copyVolume();

        FFT fft;

        fft.fwMT(lowPassRef);

        lowPassFilter(lowPassRef,
                      lowPassRef,
                      _para.pixelSize / GEN_MASK_RES,
                      (double)EDGE_WIDTH_FT / _para.pf / _para.size);

        fft.bwMT(lowPassRef);

        autoMask(_mask,
                 lowPassRef,
                 GEN_MASK_EXT,
                 GEN_MASK_EDGE_WIDTH,
                 _para.size * 0.5);

        saveMask();

        _genMask = false;
    }

    if (mask && !_mask.isEmptyRL())
    {
        ALOG(INFO, "LOGGER_ROUND") << "Performing Reference Masking";
        BLOG(INFO, "LOGGER_ROUND") << "Performing Reference Masking";

        softMask(_model.ref(0), _model.ref(0), _mask, 0);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Fourier Transforming References";
    BLOG(INFO, "LOGGER_ROUND") << "Fourier Transforming References";

    FFT fft;
    fft.fwMT(_model.ref(0));
    _model.ref(0).clearRL();
}

void MLOptimiser::allocPreCal(int& nPxl,
                              const double rU,
                              const double rL)
{
    IF_MASTER return;

    _iPxl = new int[_img[0].sizeFT()];
    _iSig = new int[_img[0].sizeFT()];

    double rU2 = gsl_pow_2(rU);
    double rL2 = gsl_pow_2(rL);

    nPxl = 0;

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        double u = QUAD(i, j);

        if ((u < rU2) && (u > rL2))
        {
            int v = AROUND(NORM(i, j));

            if (v < rU)
            {
                _iPxl[nPxl] = _img[0].iFTHalf(i, j);
                _iSig[nPxl] = v;

                nPxl++;
            }
        }
    }
}

void MLOptimiser::freePreCal()
{
    IF_MASTER return;

    delete[] _iPxl;
    delete[] _iSig;
}

void MLOptimiser::saveBestProjections()
{
    IF_MASTER return;

    FFT fft;

    Image result(_para.size, _para.size, FT_SPACE);
    Image diff(_para.size, _para.size, FT_SPACE);
    char filename[FILE_NAME_LENGTH];

    mat33 rot;
    vec2 tran;

    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            SET_0_FT(result);
            SET_0_FT(diff);

            _par[l].rank1st(rot, tran);

            _model.proj(0).project(result, rot, tran);

            FOR_EACH_PIXEL_FT(diff)
                diff[i] = _img[l][i] - result[i] * REAL(_ctf[l][i]);

            sprintf(filename, "Result_%04d_Round_%03d.bmp", _ID[l], _iter);

            fft.bw(result);
            result.saveRLToBMP(filename);
            fft.fw(result);

            sprintf(filename, "Diff_%04d_Round_%03d.bmp", _ID[l], _iter);
            fft.bw(diff);
            diff.saveRLToBMP(filename);
            fft.fw(diff);
        }
    }
}

void MLOptimiser::saveImages()
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

void MLOptimiser::saveBinImages()
{
    IF_MASTER return;

    FFT fft;

    Image bin;

    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            sprintf(filename, "Image_Bin_8_%04d.bmp", _ID[l]);

            fft.bw(_img[l]);
            binning(bin, _img[l], 8);
            fft.fw(_img[l]);

            bin.saveRLToBMP(filename);
        }
    }
            
}

void MLOptimiser::saveCTFs()
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

void MLOptimiser::saveLowPassImages()
{
    IF_MASTER return;

    FFT fft;

    Image img(size(), size(), FT_SPACE);
    SET_0_FT(img);

    char filename[FILE_NAME_LENGTH];

    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            lowPassFilter(img, _img[l], _para.pixelSize / 40, 1.0 / _para.size);

            sprintf(filename, "Image_LowPass_%04d.bmp", _ID[l]);

            fft.bw(img);
            img.saveRLToBMP(filename);
            fft.fw(img);
        }
    }
}

void MLOptimiser::saveReference(const bool finished)
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    Volume lowPass(_para.size * _para.pf,
                   _para.size * _para.pf,
                   _para.size * _para.pf,
                   FT_SPACE);

    FFT fft;

    if (finished)
        fft.bwMT(_model.ref(0));
    else
    {
        lowPassFilter(lowPass,
                      _model.ref(0),
                      (double)_resReport / _para.size,
                      (double)EDGE_WIDTH_FT / _para.size);
        fft.bwMT(lowPass);
    }

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];

    Volume result;

    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Saving Reference(s)";

        if (finished)
        {
            VOL_EXTRACT_RL(result, _model.ref(0), 1.0 / _para.pf);

            fft.fwMT(_model.ref(0));
            _model.ref(0).clearRL();

            sprintf(filename, "Reference_A_Final.mrc");
        }
        else
        {
            VOL_EXTRACT_RL(result, lowPass, 1.0 / _para.pf);
            sprintf(filename, "Reference_A_Round_%03d.mrc", _iter);
        }

        imf.readMetaData(result);

        imf.writeVolume(filename, result);
    }
    else if (_commRank == HEMI_B_LEAD)
    {
        BLOG(INFO, "LOGGER_ROUND") << "Saving Reference(s)";

        if (finished)
        {
            VOL_EXTRACT_RL(result, _model.ref(0), 1.0 / _para.pf);

            fft.fwMT(_model.ref(0));
            _model.ref(0).clearRL();

            sprintf(filename, "Reference_B_Final.mrc");
        }
        else
        {
            VOL_EXTRACT_RL(result, lowPass, 1.0 / _para.pf);
            sprintf(filename, "Reference_B_Round_%03d.mrc", _iter);
        }

        imf.readMetaData(result);

        imf.writeVolume(filename, result);
    }
}

void MLOptimiser::saveSharpReference()
{
    NT_MASTER return;

    FFT fft;

    fft.bwMT(_model.ref(0));

    ImageFile imf;
    imf.readMetaData(_model.ref(0));
    imf.writeVolume("Reference_Sharp.mrc", _model.ref(0));
}

void MLOptimiser::saveMask()
{
    ImageFile imf;
    char filename[FILE_NAME_LENGTH];

    Volume mask;

    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Saving Mask(s)";

        VOL_EXTRACT_RL(mask, _mask, 1.0 / _para.pf);

        imf.readMetaData(mask);
        sprintf(filename, "Mask_A.mrc");
        imf.writeVolume(filename, mask);
    }
    else if (_commRank == HEMI_B_LEAD)
    {
        BLOG(INFO, "LOGGER_ROUND") << "Saving Mask(s)";

        VOL_EXTRACT_RL(mask, _mask, 1.0 / _para.pf);

        imf.readMetaData(mask);
        sprintf(filename, "Mask_B.mrc");
        imf.writeVolume(filename, mask);
    }
}

void MLOptimiser::saveFSC(const bool finished) const
{
    NT_MASTER return;

    char filename[FILE_NAME_LENGTH];

    vec fsc = _model.fsc(0);

    if (finished)
        sprintf(filename, "FSC_Final.txt");
    else
        sprintf(filename, "FSC_Round_%03d.txt", _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 1; i < fsc.size(); i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _para.size * _para.pf, _para.pixelSize),
                fsc(i));

    fclose(file);
}

int searchPlace(double* topW,
                const double w,
                const int l,
                const int r)
{
    if (l < r)
    {
        if (topW[(l + r) / 2] < w)
            return searchPlace(topW, w, l, (l + r) / 2);
        else
            return searchPlace(topW, w, (l + r) / 2 + 1, r);
    }
    else
        return r;
}

void recordTopK(double* topW,
                unsigned int* iTopR,
                unsigned int* iTopT,
                const double w,
                const unsigned int iR,
                const unsigned int iT,
                const int k)
{
    int place = searchPlace(topW, w, 0, k);

    for (int i = k - 1; i > place; i--)
    {
        topW[i] = topW[i - 1];

        iTopR[i] = iTopR[i - 1];
        iTopT[i] = iTopT[i - 1];
    }

    topW[place] = w;

    iTopR[place] = iR;
    iTopT[place] = iT;
}

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sig,
                      const double rU,
                      const double rL)
{
    double result = 0;

    double rU2 = gsl_pow_2(rU);
    double rL2 = gsl_pow_2(rL);

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        double u = QUAD(i, j);

        if ((u < rU2) && (u > rL2))
        {
            int v = AROUND(NORM(i, j));
            if (v < rU)
            {
                int index = dat.iFTHalf(i, j);

                result += ABS2(dat.iGetFT(index)
                             - REAL(ctf.iGetFT(index))
                             * pri.iGetFT(index))
                        / (-2 * sig[v]);
            }
        }
    }

    return result;
}

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sig,
                      const int* iPxl,
                      const int* iSig,
                      const int m)
{
    double result = 0;

    for (int i = 0; i < m; i++)
        result += ABS2(dat.iGetFT(iPxl[i])
                     - REAL(ctf.iGetFT(iPxl[i]))
                     * pri.iGetFT(iPxl[i]))
                / (-2 * sig(iSig[i]));

    return result;
}

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sig,
                      const double rU,
                      const double rL)
{
    double result = 0;

    double rU2 = gsl_pow_2(rU);
    double rL2 = gsl_pow_2(rL);

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        double u = QUAD(i, j);

        if ((u < rU2) && (u > rL2))
        {
            int v = AROUND(NORM(i, j));
            if (v < rU)
            {
                int index = dat.iFTHalf(i, j);

                result += ABS2(dat.iGetFT(index)
                             - REAL(ctf.iGetFT(index))
                             * pri.iGetFT(index)
                             * tra.iGetFT(index))
                        / (-2 * sig[v]);
            }
        }
    }

    return result;
}

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& tra,
                      const Image& ctf,
                      const vec& sig,
                      const int* iPxl,
                      const int* iSig,
                      const int m)
{
    double result = 0;

    for (int i = 0; i < m; i++)
        result += ABS2(dat.iGetFT(iPxl[i])
                     - REAL(ctf.iGetFT(iPxl[i]))
                     * pri.iGetFT(iPxl[i])
                     * tra.iGetFT(iPxl[i]))
                / (-2 * sig(iSig[i]));

    return result;
}

vec logDataVSPrior(const vector<Image>& dat,
                   const Image& pri,
                   const vector<Image>& ctf,
                   const vector<int>& groupID,
                   const mat& sig,
                   const double rU,
                   const double rL)
{
    int n = dat.size();

    vec result = vec::Zero(n);

    double rU2 = gsl_pow_2(rU);
    double rL2 = gsl_pow_2(rL);

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        double u = QUAD(i, j);

        if ((u < rU2) && (u > rL2))
        {
            int v = AROUND(NORM(i, j));
            if (v < rU)
            {
                int index = dat[0].iFTHalf(i, j);

                #pragma omp parallel for
                for (int l = 0; l < n; l++)
                {
                    result(l) += ABS2(dat[l].iGetFT(index)
                                    - REAL(ctf[l].iGetFT(index))
                                    * pri.iGetFT(index))
                               / (-2 * sig(groupID[l] - 1, v));
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
                   const mat& sig,
                   const int* iPxl,
                   const int* iSig,
                   const int m)
{
    int n = dat.size();

    vec result = vec::Zero(n);

    #pragma omp parallel for
    for (int l = 0; l < n; l++)
    {
        for (int i = 0; i < m; i++)
            result(l) += ABS2(dat[l].iGetFT(iPxl[i])
                            - REAL(ctf[l].iGetFT(iPxl[i]))
                            * pri.iGetFT(iPxl[i]))
                       / (-2 * sig(groupID[l] - 1, iSig[i]));
                         
    }

    return result;
}

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sig,
                   const double rU,
                   const double rL)
{
    return exp(logDataVSPrior(dat, pri, ctf, sig, rU, rL));
}

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& tra,
                   const Image& ctf,
                   const vec& sig,
                   const double rU,
                   const double rL)
{
    return exp(logDataVSPrior(dat, pri, tra, ctf, sig, rU, rL));
}

void scaleDataVSPrior(vec& sXA,
                      vec& sAA,
                      const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const double rU,
                      const double rL)
{
    double rU2 = gsl_pow_2(rU);
    double rL2 = gsl_pow_2(rL);

    for (int i = 0; i < rU; i++)
    {
        sXA(i) = 0;
        sAA(i) = 0;
    }
    
    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_PIXEL_R_FT(CEIL(rU) + 1)
    {
        double u = QUAD(i, j);

        if ((u < rU2) && (u > rL2))
        {
            int v = AROUND(NORM(i, j));
            if (v < rU)
            {
                int index = dat.iFTHalf(i, j);

                #pragma omp critical
                sXA(v) += REAL(dat.iGetFT(index)
                             * pri.iGetFT(index)
                             * REAL(ctf.iGetFT(index)));

                #pragma omp critical
                sAA(v) += REAL(pri.iGetFT(index)
                             * pri.iGetFT(index)
                             * gsl_pow_2(REAL(ctf.iGetFT(index))));
            }
        }
    }
}

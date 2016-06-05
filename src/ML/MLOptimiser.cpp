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
    _r = MIN(16, MAX(MAX_GAP, _para.size / 16));
    _iter = 0;
    _model.setR(_r);

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

    NT_MASTER
    {
        ALOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";
        initID();

        ALOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";
        initImg();

        ALOG(INFO, "LOGGER_INIT") << "Setting Parameters: _N";
        allReduceN();
        ALOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere A: " << _N;
        BLOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere B: " << _N;

        /***
        ALOG(INFO) << "Applying Low Pass Filter on Initial References";
        _model.lowPassRef(_r, EDGE_WIDTH_FT);
        ***/

        ALOG(INFO, "LOGGER_INIT") << "Seting maxRadius of _model";
        _model.setR(_r);

        ALOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";
        _model.initProjReco();

        ALOG(INFO, "LOGGER_INIT") << "Generating CTFs";
        initCTF();

        ALOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";
        initParticles();
    }

    MLOG(INFO, "LOGGER_INIT") << "Broadacasting Information of Groups";
    bcastGroupInfo();

    NT_MASTER
    {
        ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";
        initSigma();
    }
}

void MLOptimiser::expectation()
{
    IF_MASTER return;

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        Image image(size(), size(), FT_SPACE);

        for (int phase = 0; phase < MAX_N_PHASE_PER_ITER; phase++)
        {
            if ((_iter != 0) || (phase != 0))
            {
                if (_iter < N_ITER_TOTAL_GLOBAL_SEARCH)
                {
                    if (phase == 0)
                        _par[l].resample(_para.m * _para.mf,
                                         ALPHA_TOTAL_GLOBAL_SEARCH);
                    else
                        _par[l].resample(_para.m,
                                         ALPHA_SEARCH_BG);
                }
                else if (_iter < N_ITER_TOTAL_GLOBAL_SEARCH
                               + N_ITER_PARTIAL_GLOBAL_SEARCH)
                {
                    if (phase == 0)
                        _par[l].resample(_para.m * _para.mf,
                                         (ALPHA_GLOBAL_SEARCH_MAX
                                        - ALPHA_GLOBAL_SEARCH_MIN)
                                       * (N_ITER_TOTAL_GLOBAL_SEARCH 
                                        + N_ITER_PARTIAL_GLOBAL_SEARCH
                                        - _iter - 1)
                                       / (N_ITER_PARTIAL_GLOBAL_SEARCH - 1)
                                       + ALPHA_GLOBAL_SEARCH_MIN);
                    else
                        _par[l].resample(_para.m,
                                         ALPHA_SEARCH_BG);
                }
                else
                {
                    if (phase == 0)
                        _par[l].resample(_para.m,
                                         ALPHA_LOCAL_SEARCH);
                    else
                        _par[l].resample();
                }
            }

            /***
            double nt = (phase == N_PHASE_PER_ITER - 1)
                      ? 2 * TOP_K
                      : _par[l].n() / 10;
            ***/
            double nt = _par[l].n() / 10;

            int nSearch = 0;
            do
            {
                // perturbation
                _par[l].perturb();

                vec logW(_par[l].n());
                mat33 rot;
                vec2 t;
                for (int m = 0; m < _par[l].n(); m++)
                {
                    _par[l].rot(rot, m);
                    _par[l].t(t, m);
                    _model.proj(0).project(image, rot, t);

                    logW[m] = logDataVSPrior(_img[l], // data
                                             image, // prior
                                             _ctf[l], // ctf
                                             _sig.row(_groupID[l] - 1).head(_r).transpose(),
                                             _r);
                }

                logW.array() -= logW.maxCoeff(); // avoiding numerical error

                for (int m = 0; m < _par[l].n(); m++)
                    _par[l].mulW(exp(logW(m)), m);
                    // _par[l].mulW(logW(m) > -20 ? exp(logW(m)) : 0, m);

                _par[l].normW();

                if (_ID[l] < 100)
                {
                    char filename[FILE_NAME_LENGTH];
                    snprintf(filename,
                             sizeof(filename),
                             "Particle_%04d_Round_%03d_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase,
                             nSearch);
                    save(filename, _par[l]);
                }

                nSearch++;
            } while ((_par[l].neff() > nt) &&
                     (nSearch < MAX_N_SEARCH_PER_PHASE));

            if (nSearch == MAX_N_SEARCH_PER_PHASE) break;
        }
    }
}

void MLOptimiser::maximization()
{
    ALOG(INFO, "LOGGER_ROUND") << "Generate Sigma for the Next Iteration";
    allReduceSigma();

    ALOG(INFO, "LOGGER_ROUND") << "Reconstruct Reference";
    reconstructRef();
}

void MLOptimiser::run()
{
    MLOG(INFO, "LOGGER_ROUND") << "Initialising MLOptimiser";

    init();

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Entering Iteration";
    for (_iter = 0; _iter < _para.iterMax; _iter++)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter;

        MLOG(INFO, "LOGGER_ROUND") << "Performing Expectation";
        expectation();

        MLOG(INFO, "LOGGER_ROUND") << "Performing Maximization";
        maximization();

        MLOG(INFO, "LOGGER_ROUND") << "Calculating FSC";
        _model.BcastFSC();

        MLOG(INFO, "LOGGER_ROUND") << "Calculating SNR";
        _model.refreshSNR();

        MLOG(INFO, "LOGGER_ROUND") << "Recording Current Resolution";
        _res = _model.resolutionP();
        MLOG(INFO, "LOGGER_ROUND") << "Current Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1,
                                                   _para.size,
                                                   _para.pixelSize)
                                   << " (Angstrom)";
        MLOG(INFO, "LOGGER_ROUND") << "Current Resolution: "
                                   << _res
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_res, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Updating Cutoff Frequency: ";
        _model.updateR();
        _r = _model.r();

        MLOG(INFO, "LOGGER_ROUND") << "New Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        NT_MASTER
        {
            ALOG(INFO, "LOGGER_ROUND") << "Refreshing Projectors";
            _model.refreshProj();

            ALOG(INFO, "LOGGER_ROUND") << "Refreshing Reconstructors";
            _model.refreshReco();
        }

        // save the result of last projection
        if (_iter == _para.iterMax - 1)
        {
            saveBestProjections();
            saveImages();
        } 
    }
}

void MLOptimiser::clear()
{
    _img.clear();
    _par.clear();
    _ctf.clear();
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
    IF_MASTER
        _nGroup = _exp.nGroup();

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting Number of Groups";
    MPI_Bcast(&_nGroup, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);

    ALOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Sigma";
    NT_MASTER _sig.resize(_nGroup, maxR() + 1);
}

void MLOptimiser::initRef()
{
    _model.appendRef(Volume());

    ALOG(INFO, "LOGGER_INIT") << "Read Initial Model from Hard-disk";

    ImageFile imf(_para.initModel, "rb");
    imf.readMetaData();
    imf.readVolume(_model.ref(0));

    ALOG(INFO, "LOGGER_INIT") << "Size of the Initial Model is: "
                              << _model.ref(0).nColRL()
                              << " X "
                              << _model.ref(0).nRowRL()
                              << " X "
                              << _model.ref(0).nSlcRL();

    ALOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform";

    FFT fft;
    fft.fw(_model.ref(0));
    _model.ref(0).clearRL();
    /***
    fft.bw(_model.ref(0));
    fft.fw(_model.ref(0));
    _model.ref(0).clearRL();
    ***/
}

void MLOptimiser::initID()
{
    sql::Statement stmt("select ID from particles;", -1, _exp.expose());
    while (stmt.step())
        _ID.push_back(stmt.get_int(0));
}

void MLOptimiser::initImg()
{
    FFT fft;
    _img.clear();
    _img.resize(_ID.size());

    std::string imgName;

    sql::Statement stmt("select Name from particles where ID = ?", -1, _exp.expose());
    FOR_EACH_2D_IMAGE
    {
        stmt.bind_int(1, _ID[l]);
        // ILOG(INFO) << "Read 2D Image ID of Which is " << _ID[i];
        // get the filename of the image from database
        if (stmt.step())
            imgName = stmt.get_text(0);
        else
            throw std::runtime_error("Database changed");
        stmt.reset();
        // read the image fromm hard disk
	    Image& currentImg = _img[l];
        ImageFile imf(imgName.c_str(), "rb");
        imf.readMetaData();
        imf.readImage(currentImg);

        if ((currentImg.nColRL() != _para.size) ||
            (currentImg.nRowRL() != _para.size))
            LOG(FATAL) << "Incorrect Size of 2D Images";

        /***
        // apply a soft mask on it
        softMask(currentImg,
                 currentImg,
                 _para.size / 4,
                 EDGE_WIDTH_RL);
                 ***/

        /***
        sprintf(imgName, "%04dMasked.bmp", _ID[i]);
        _img[i].saveRLToBMP(imgName);
        ***/

        // perform Fourier Transform
        fft.fw(currentImg);
        currentImg.clearRL();
    }
}

void MLOptimiser::initCTF()
{
    IF_MASTER return;

    // get CTF attributes from _exp
    CTFAttr ctfAttr;

    sql::Statement stmt(
            "select Voltage, DefocusU, DefocusV, DefocusAngle, CS from \
             micrographs, particles where \
             particles.micrographID = micrographs.ID and \
             particles.ID = ?;", -1, _exp.expose());
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
        } else {
            throw std::runtime_error("No data");
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

void MLOptimiser::initSigma()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Image";

    Image avg = _img[0].copyImage();

    for (size_t l = 1; l < _ID.size(); l++)
        ADD_FT(avg, _img[l]);

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &avg[0],
                  avg.sizeFT(),
                  MPI_DOUBLE_COMPLEX,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    SCALE_FT(avg, 1.0 / _N);

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";

    // vec avgPs(maxR(), fill::zeros);
    vec avgPs = vec::Zero(maxR());
    vec ps(maxR());
    FOR_EACH_2D_IMAGE
    {
        powerSpectrum(ps, _img[l], maxR());
        avgPs += ps;
        // cblas_daxpy(maxR(), 1, ps.memptr(), 1, avgPs.memptr(), 1);
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

    vec psAvg(maxR());
    // powerSpectrum(psAvg, avg, maxR());
    for (int i = 0; i < maxR(); i++)
        psAvg(i) = ringAverage(i, avg, [](const Complex x){ return REAL(x) + IMAG(x); });

    // avgPs -> average power spectrum
    // psAvg -> expectation of pixels
    ALOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";

    _sig.leftCols(_sig.cols() - 1).rowwise() = (avgPs - psAvg).transpose() / 2;
    // _sig.head_cols(_sig.n_cols - 1).each_row() = (avgPs - psAvg).t() / 2;
    // _sig.head_cols(_sig.n_cols - 1).each_row() = (avgPs - psAvg).t();

    /***
    ALOG(INFO) << "Saving Initial Sigma";
    if (_commRank == HEMI_A_LEAD)
        _sig.save("Sigma_000.txt", raw_ascii);
        ***/
}

void MLOptimiser::initParticles()
{
    IF_MASTER return;

    FOR_EACH_2D_IMAGE
        _par.push_back(Particle());

    FOR_EACH_2D_IMAGE
        _par[l].init(_para.m * _para.mf,
                     _para.maxX,
                     _para.maxY,
                     &_sym);

    /***
    FOR_EACH_2D_IMAGE
    {
        _par.push_back(Particle());
        _par.back().init(_para.m,
                         _para.maxX,
                         _para.maxY,
                         &_sym);
    }
    ***/
}


void MLOptimiser::allReduceSigma()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_ROUND") << "Clear Up Sigma";

    // set re-calculating part to zero
    _sig.leftCols(_r).setZero();
    _sig.rightCols(1).setZero();

    ALOG(INFO, "LOGGER_ROUND") << "Recalculate Sigma";
    // loop over 2D images
    FOR_EACH_2D_IMAGE
    {
        // sort weights in particle and store its indices
        uvec iSort = _par[l].iSort();

        Coordinate5D coord;
        double w;
        Image img(size(), size(), FT_SPACE);
        vec sig(_r);

        // loop over sampling points with top K weights
        for (int m = 0; m < TOP_K; m++)
        {
            // get coordinate
            _par[l].coord(coord, iSort[m]);
            // get weight
            w = _par[l].w(iSort[m]);

            // calculate differences
            _model.proj(0).project(img, coord);
            FOR_EACH_PIXEL_FT(img)
                img[i] *= REAL(_ctf[l][i]);
            // MUL_FT(img, _ctf[l]);
            NEG_FT(img);
            ADD_FT(img, _img[l]);

            powerSpectrum(sig, img, _r);

            // sum up the results from top K sampling points
            // TODO Change it to w
            _sig.row(_groupID[l] - 1).head(_r) += (1.0 / TOP_K) * sig.transpose() / 2;
        }

        _sig(_groupID[l] - 1, _sig.cols() - 1) += 1;
        // _sig.row(_groupID[l] - 1).tail(1) += 1;
    }

    ALOG(INFO, "LOGGER_ROUND") << "Averaging Sigma of Images Belonging to the Same Group";

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.data(),
                  _r * _nGroup,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.col(_sig.cols() - 1).data(),
                  // _sig.colptr(_sig.n_cols - 1),
                  _nGroup,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    /***
    // TODO: there is something wrong here! FIX IT!
    _sig.each_row([this](rowvec& x){ x.head(_r) /= x(x.n_elem - 1); });
    ***/
    for (int i = 0; i < _sig.rows(); i++)
        _sig.row(i).head(_r) /= _sig(i, _sig.cols() - 1);

    /***
    ALOG(INFO) << "Saving Sigma";
    if (_commRank == HEMI_A_LEAD)
    {
        char filename[FILE_NAME_LENGTH];
        sprintf(filename, "Sigma_%03d.txt", _iter + 1);
        _sig.save(filename, raw_ascii);
    }
    ***/
}

void MLOptimiser::reconstructRef()
{
    IF_MASTER return;

    Image img(size(), size(), FT_SPACE);

    ALOG(INFO, "LOGGER_ROUND") << "Inserting High Probability 2D Images into Reconstructor";

    FOR_EACH_2D_IMAGE
    {
        /***
        ILOG(INFO) << "Inserting Particle "
                   << _ID[l]
                   << " into Reconstructor";
                   ***/

        // reduce the CTF effect
        reduceCTF(img, _img[l], _ctf[l]);
        // reduceCTF(img, _img[l], _ctf[l], _r);

        uvec iSort = _par[l].iSort();

        mat33 rot;
        vec2 t;
        // Coordinate5D coord;
        double w;
        for (int m = 0; m < TOP_K; m++)
        {
            // get coordinate
            // _par[l].coord(coord, iSort[m]);
            _par[l].rot(rot, iSort[m]);
            _par[l].t(t, iSort[m]);
            // get weight
            w = _par[l].w(iSort[m]);

            // TODO: _model.reco(0).insert(_img[l], coord, w);
            _model.reco(0).insert(_img[l], rot, t, 1);
        }
    }

    ALOG(INFO, "LOGGER_ROUND") << "Reconstructing References for Next Iteration";

    _model.reco(0).reconstruct(_model.ref(0));

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];
    Volume result;
    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Saving References";

        VOL_EXTRACT_RL(result, _model.ref(0), 1.0 / _para.pf);

        imf.readMetaData(result);
        sprintf(filename, "Reference_A_Round_%03d.mrc", _iter);
        imf.writeVolume(filename, result);
    }
    else if (_commRank == HEMI_B_LEAD)
    {
        BLOG(INFO, "LOGGER_ROUND") << "Saving References";

        VOL_EXTRACT_RL(result, _model.ref(0), 1.0 / _para.pf);

        imf.readMetaData(result);
        sprintf(filename, "Reference_B_Round_%03d.mrc", _iter);
        imf.writeVolume(filename, result);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Fourier Transforming References";

    FFT fft;
    fft.fw(_model.ref(0));
    _model.ref(0).clearRL();
}

void MLOptimiser::saveBestProjections()
{
    FFT fft;

    Image result(_para.size, _para.size, FT_SPACE);
    Coordinate5D coord;
    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        SET_0_FT(result);

        uvec iSort = _par[l].iSort();
        _par[l].coord(coord, iSort[0]);

        _model.proj(0).project(result, coord);

        sprintf(filename, "Result_%04d.bmp", _ID[l]);

        fft.bw(result);
        result.saveRLToBMP(filename);
        fft.fw(result);
    }
}

void MLOptimiser::saveImages()
{
    FFT fft;

    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        sprintf(filename, "Image_%04d.bmp", _ID[l]);

        fft.bw(_img[l]);
        _img[l].saveRLToBMP(filename);
        fft.fw(_img[l]);
    }
}

double logDataVSPrior(const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const vec& sig,
                      const int r)
{
    double result = 0;

    IMAGE_FOR_EACH_PIXEL_FT(pri)
    {
        int u = AROUND(NORM(i, j));

        if ((FREQ_DOWN_CUTOFF < u) &&
            (u < r))
            result += ABS2(dat.getFT(i, j)
                         - REAL(ctf.getFT(i, j))
                         * pri.getFT(i, j))
                        / (-2 * sig[u]);
    }

    return result;
}

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sig,
                   const int r)
{
    return exp(logDataVSPrior(dat, pri, ctf, sig, r));
}

/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Bing Li, Heng Guo
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
    MLOG(INFO) << "Setting MPI Environment of _model";
    _model.setMPIEnv(_commSize, _commRank, _hemi);

    MLOG(INFO) << "Setting up Symmetry";
    _sym.init(_para.sym);

    MLOG(INFO) << "Passing Parameters to _model";
    _model.init(_para.k,
                _para.size,
                0,
                _para.pf,
                _para.pixelSize,
                _para.a,
                _para.alpha,
                &_sym);

    MLOG(INFO) << "Setting Parameters: _r, _iter";
    _r = _para.size / 16;
    _iter = 0;
    _model.setR(_r);

    MLOG(INFO) << "Openning Database File";
    _exp.openDatabase(_para.db);

    MLOG(INFO) << "Setting MPI Environment of _exp";
    _exp.setMPIEnv(_commSize, _commRank, _hemi);

    MLOG(INFO) << "Broadcasting ID of _exp";
    _exp.bcastID();

    MLOG(INFO) << "Preparing Temporary File of _exp";
    _exp.prepareTmpFile();

    MLOG(INFO) << "Scattering _exp";
    _exp.scatter();

    NT_MASTER
    {
        ALOG(INFO) << "Appending Initial References into _model";
        initRef();

        ALOG(INFO) << "Initialising IDs of 2D Images";
        initID();

        ALOG(INFO) << "Initialising 2D Images";
        initImg();

        ALOG(INFO) << "Setting Parameters: _N";
        allReduceN();
        ALOG(INFO) << "Number of Images in Hemisphere A: " << _N;
        BLOG(INFO) << "Number of Images in Hemisphere B: " << _N;

        /***
        ALOG(INFO) << "Applying Low Pass Filter on Initial References";
        _model.lowPassRef(_r, EDGE_WIDTH_FT);
        ***/

        ALOG(INFO) << "Seting maxRadius of _model";
        _model.setR(_r);

        ALOG(INFO) << "Setting Up Projectors and Reconstructors of _model";
        _model.initProjReco();

        ALOG(INFO) << "Generating CTFs";
        initCTF();
    
        ALOG(INFO) << "Initialising Particle Filters";
        initParticles();
    }

    MLOG(INFO) << "Broadacasting Information of Groups";
    bcastGroupInfo();

    NT_MASTER
    {
        ALOG(INFO) << "Estimating Initial Sigma";
        initSigma();
    }
}

void MLOptimiser::expectation()
{
    IF_MASTER return;

    Image image(size(),
                size(),
                FT_SPACE);

    FOR_EACH_2D_IMAGE
    {
        ILOG(INFO) << "Performing Expectation on Image " << _ID[l]
                   << " with Radius of " << _r;

        /***
        if (_par[l].neff() < 1.05)
        {
            ILOG(WARNING) << "Round " << _iter
                          << ": Resetting Particle due to Extreme Small neff";
            _par[l].reset();
        }
        ***/
        if (_par[l].neff() < _par[l].n() / 5)
        {
            ILOG(INFO) << "Round " << _iter
                       << ": Resampling Particle " << _ID[l]
                       << " for neff = " << _par[l].neff();

            // _par[l].resample(GSL_MAX_INT(_para.m, _par[l].n() / 2));
            _par[l].resample();
            // _par[l].perturb();
        }
        else 
            _par[l].perturb();

        for (int m = 0; m < _par[l].n(); m++)
        {
            Coordinate5D coord;
            _par[l].coord(coord, m);
            _model.proj(0).project(image, coord);

            double w = dataVSPrior(image,
                                   _img[l],
                                   _ctf[l],
                                   _sig.row(_groupID[l] - 1).head(_r).t(),
                                   _r);

            _par[l].mulW(w, m);
        }
        _par[l].normW();

        // calculate current variance
        _par[l].calVari();

        double k0, k1, k2, s0, s1, rho;
        _par[l].vari(k0, k1, k2, s0, s1, rho);
        ILOG(INFO) << "Round " << _iter
                   << ": Information of Particle " << _ID[l]
                   << ", Neff " << _par[l].neff()
                   << ", k0 " << k0
                   << ", k1 " << k1
                   << ", k2 " << k2
                   << ", s0 " << s0
                   << ", s1 " << s1
                   << ", rho " << rho;

        char filename[FILE_NAME_LENGTH];
        sprintf(filename, "Particle_%04d_Round_%03d.par", _ID[l], _iter);
        save(filename, _par[l]);
    }
}

void MLOptimiser::maximization()
{
    /***
    ALOG(INFO) << "Generate Sigma for the Next Iteration";
    allReduceSigma();
    ***/

    /***
    ALOG(INFO) << "Reconstruct Reference";
    reconstructRef();
    ***/
}

void MLOptimiser::run()
{
    MLOG(INFO) << "Initialising MLOptimiser";

    init();

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO) << "Entering Iteration";
    for (_iter = 0; _iter < _para.iterMax; _iter++)
    {
        MLOG(INFO) << "Round " << _iter;

        MLOG(INFO) << "Performing Expectation";
        expectation();

        MLOG(INFO) << "Performing Maximization";
        maximization();

        MLOG(INFO) << "Calculating FSC";
        _model.BcastFSC();

        MLOG(INFO) << "Calculating SNR";
        _model.refreshSNR();

        MLOG(INFO) << "Recording Current Resolution";
        _res = _model.resolutionP();
        MLOG(INFO) << "Current Cutoff Frequency: "
                   << _r - 1
                   << " (Spatial), "
                   << 1.0 / resP2A(_r - 1, _para.size, _para.pixelSize)
                   << " (Angstrom)";
        MLOG(INFO) << "Current Resolution: "
                   << _res
                   << " (Spatial), "
                   << 1.0 / resP2A(_res, _para.size, _para.pixelSize)
                   << " (Angstrom)";

        MLOG(INFO) << "Updating Cutoff Frequency: ";
        _model.updateR();
        _r = _model.r();
        MLOG(INFO) << "New Cutoff Frequency: "
                   << _r - 1
                   << " (Spatial), "
                   << 1.0 / resP2A(_r - 1, _para.size, _para.pixelSize)
                   << " (Angstrom)";

        /***
        NT_MASTER
        {
            ALOG(INFO) << "Refreshing Projectors";
            _model.refreshProj();

            ALOG(INFO) << "Refreshing Reconstructors";
            _model.refreshReco();
        }
        ***/
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
    ALOG(INFO) << "Storing GroupID";
    NT_MASTER
    {
        char sql[SQL_COMMAND_LENGTH];
    
        FOR_EACH_2D_IMAGE
        {
            sprintf(sql, "select GroupID from particles where ID = %d;", _ID[l]);
            _exp.execute(sql,
                         SQLITE3_CALLBACK
                         {
                             ((vector<int>*)data)->push_back(atoi(values[0]));
                             return 0;
                         },
                         &_groupID); 
        }
    }

    MLOG(INFO) << "Getting Number of Groups from Database";
    IF_MASTER
        _exp.execute("select count(*) from groups;",
                     SQLITE3_CALLBACK
                     {
                         *((int*)data) = atoi(values[0]);
                         return 0;
                     },
                     &_nGroup);

    MLOG(INFO) << "Broadcasting Number of Groups";
    MPI_Bcast(&_nGroup, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
    
    ALOG(INFO) << "Setting Up Space for Storing Sigma";
    NT_MASTER _sig.set_size(_nGroup, maxR() + 1);
}

void MLOptimiser::initRef()
{
    _model.appendRef(Volume());

    ALOG(INFO) << "Read Initial Model from Hard-disk";

    ImageFile imf(_para.initModel, "rb");
    imf.readMetaData();
    imf.readVolume(_model.ref(0));

    ALOG(INFO) << "Size of the Initial Model is: "
               << _model.ref(0).nColRL()
               << " X "
               << _model.ref(0).nRowRL()
               << " X "
               << _model.ref(0).nSlcRL();

    ALOG(INFO) << "Performing Fourier Transform";

    FFT fft;
    fft.fw(_model.ref(0));
    _model.ref(0).clearRL();
}

void MLOptimiser::initID()
{
    char sql[] = "select ID from particles;";
    _exp.execute(sql,
                 SQLITE3_CALLBACK
                 {
                    ((vector<int>*)data)->push_back(atoi(values[0]));
                    return 0;
                 },
                 &_ID);
}

void MLOptimiser::initImg()
{
    FFT fft;

    char sql[SQL_COMMAND_LENGTH];
    char imgName[FILE_NAME_LENGTH];

    FOR_EACH_2D_IMAGE
    {
        // ILOG(INFO) << "Read 2D Image ID of Which is " << _ID[i];

        _img.push_back(Image());

        // get the filename of the image from database
        sprintf(sql, "select Name from particles where ID = %d;", _ID[l]);
        _exp.execute(sql,
                     SQLITE3_CALLBACK
                     {
                         sprintf((char*)data, "%s", values[0]); 
                         return 0;
                     },
                     imgName);

        // read the image fromm hard disk
        ImageFile imf(imgName, "rb");
        imf.readMetaData();
        imf.readImage(_img.back());

        if ((_img.back().nColRL() != _para.size) ||
            (_img.back().nRowRL() != _para.size))
            LOG(FATAL) << "Incorrect Size of 2D Images";

        // apply a soft mask on it
        softMask(_img.back(),
                 _img.back(),
                 _para.size / 4,
                 EDGE_WIDTH_RL);

        /***
        sprintf(imgName, "%04dMasked.bmp", _ID[i]);
        _img[i].saveRLToBMP(imgName);
        ***/

        // perform Fourier Transform
        fft.fw(_img.back());
        _img.back().clearRL();
    }
}

void MLOptimiser::initCTF()
{
    IF_MASTER return;

    // get CTF attributes from _exp
    char sql[SQL_COMMAND_LENGTH];

    CTFAttr ctfAttr;

    FOR_EACH_2D_IMAGE
    {
        // get attributes of CTF from database
        sprintf(sql,
                "select Voltage, DefocusU, DefocusV, DefocusAngle, CS from \
                 micrographs, particles where \
                 particles.micrographID = micrographs.ID and \
                 particles.ID = %d;",
                _ID[l]);
        _exp.execute(sql,
                     SQLITE3_CALLBACK
                     {
                        ((CTFAttr*)data)->voltage = atof(values[0]);
                        ((CTFAttr*)data)->defocusU = atof(values[1]);
                        ((CTFAttr*)data)->defocusV = atof(values[2]);
                        ((CTFAttr*)data)->defocusAngle = atof(values[3]);
                        ((CTFAttr*)data)->CS = atof(values[4]);
                        return 0;
                     },
                     &ctfAttr);

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

    ALOG(INFO) << "Calculating Average Image";

    Image avg = _img[0];

    for (int l = 1; l < _ID.size(); l++)
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

    ALOG(INFO) << "Calculating Average Power Spectrum";

    vec avgPs(maxR(), fill::zeros);
    vec ps(maxR());
    FOR_EACH_2D_IMAGE
    {
        powerSpectrum(ps, _img[l], maxR());
        avgPs += ps;
        // cblas_daxpy(maxR(), 1, ps.memptr(), 1, avgPs.memptr(), 1);
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs.memptr(),
                  maxR(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    avgPs /= _N;

    // ALOG(INFO) << "Average Power Spectrum is " << endl << avgPs;

    // ALOG(INFO) << "Calculating Power Spectrum of Average Image";

    ALOG(INFO) << "Calculating Expectation for Initializing Sigma";

    vec psAvg(maxR());
    // powerSpectrum(psAvg, avg, maxR());
    for (int i = 0; i < maxR(); i++)
        psAvg(i) = ringAverage(i, avg, [](const Complex x){ return REAL(x) + IMAG(x); });

    // ALOG(INFO) << "Power Spectrum of Average Image is " << endl << psAvg;
    
    // avgPs -> average power spectrum
    // psAvg -> expectation of pixels
    ALOG(INFO) << "Substract avgPs and psAvg for _sig";

    _sig.head_cols(_sig.n_cols - 1).each_row() = (avgPs - psAvg).t() / 2;
    // _sig.head_cols(_sig.n_cols - 1).each_row() = (avgPs - psAvg).t();

    ALOG(INFO) << "Saving Initial Sigma";
    if (_commRank == HEMI_A_LEAD)
        _sig.save("Sigma_000.txt", raw_ascii);
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

    ALOG(INFO) << "Clear Up Sigma";

    // set re-calculating part to zero
    _sig.head_cols(_r).zeros();
    _sig.tail_cols(1).zeros();

    ALOG(INFO) << "Recalculate Sigma";
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
            MUL_FT(img, _ctf[l]);
            NEG_FT(img);
            ADD_FT(img, _img[l]);

            powerSpectrum(sig, img, _r);

            // sum up the results from top K sampling points
            // TODO Change it to w
            _sig.row(_groupID[l] - 1).head(_r) += (1.0 / TOP_K) * sig.t() / 2;
        }
        
        _sig.row(_groupID[l] - 1).tail(1) += 1;
    }

    ALOG(INFO) << "Averaging Sigma of Images Belonging to the Same Group";

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.memptr(),
                  _r * _nGroup,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.colptr(_sig.n_cols - 1),
                  _nGroup,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    _sig.each_row([this](rowvec& x){ x.head(_r) /= x(x.n_elem - 1); });

    ALOG(INFO) << "Saving Sigma";
    if (_commRank == HEMI_A_LEAD)
    {
        char filename[FILE_NAME_LENGTH];
        sprintf(filename, "Sigma_%03d.txt", _iter + 1);
        _sig.save(filename, raw_ascii);
    }
}

void MLOptimiser::reconstructRef()
{
    IF_MASTER return;

    Image img(size(), size(), FT_SPACE);

    ALOG(INFO) << "Inserting High Probability 2D Images into Reconstructor";

    FOR_EACH_2D_IMAGE
    {
        ILOG(INFO) << "Inserting Particle "
                   << _ID[l]
                   << " into Reconstructor";

        // reduce the CTF effect
        reduceCTF(img, _img[l], _ctf[l], _r);

        uvec iSort = _par[l].iSort();

        Coordinate5D coord;
        double w;
        for (int m = 0; m < TOP_K; m++)
        {
            // get coordinate
            _par[l].coord(coord, iSort[m]);
            // get weight
            w = _par[l].w(iSort[m]);

            // TODO: _model.reco(0).insert(_img[l], coord, w);
            _model.reco(0).insert(_img[l], coord, 1);
        }
    }

    ALOG(INFO) << "Reconstructing References for Next Iteration";

    _model.reco(0).reconstruct(_model.ref(0));

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];
    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO) << "Saving References";

        imf.readMetaData(_model.ref(0));
        sprintf(filename, "Reference_A_Round_%03d.mrc", _iter);
        imf.writeVolume(filename, _model.ref(0));
    }
    else if (_commRank == HEMI_B_LEAD)
    {
        BLOG(INFO) << "Saving References";

        imf.readMetaData(_model.ref(0));
        sprintf(filename, "Reference_B_Round_%03d.mrc", _iter);
        imf.writeVolume(filename, _model.ref(0));
    }

    ALOG(INFO) << "Fourier Transforming References";

    FFT fft;
    fft.fw(_model.ref(0));
    _model.ref(0).clearRL();
}

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sig,
                   const int r)
{
    double result = 1;
    // int counter = 0;

    IMAGE_FOR_EACH_PIXEL_FT(pri)
    {
        // adding /u for compensate
        int u = AROUND(NORM(i, j));
        if ((FREQ_DOWN_CUTOFF < u) &&
            (u < r))
        {
            result *= exp(ABS2(dat.getFT(i, j)
                             - ctf.getFT(i, j)
                             * pri.getFT(i, j))
                            / (-2 * sig(u)))
                    / (2 * M_PI * sig(u));
                    // / (2 * M_PI * u);
            // counter++;
        }
    }

    return result;

    // return exp(result);
}

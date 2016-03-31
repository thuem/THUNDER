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
    _model.init(0,
                _para.pf,
                _para.pixelSize,
                _para.a,
                _para.alpha,
                &_sym);

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

        ALOG(INFO) << "Setting Parameters: _N, _r, _iter";
        allReduceN();
        _r = maxR() / 8; // start from 1 / 8 of highest frequency
        _iter = 0;
        ALOG(INFO) << "_N = " << _N
                   << ", _r = " << _r
                   << ", _iter = " << _iter;

        ALOG(INFO) << "Applying Low Pass Filter on Initial References";
        _model.lowPassRef(_r, EDGE_WIDTH_FT);

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
        ILOG(INFO) << "Performing Expectation on Particle " << _ID[l];

        stringstream ss;
        ss << "Particle" << _ID[l] << ".par";
        FILE* file = fopen(ss.str().c_str(), "w");
        ss.str("");

        if (_par[l].neff() < _par[l].N() / 3)
            _par[l].resample();
        else 
            _par[l].perturb();

        for (int m = 0; m < _par[l].N(); m++)
        {
            Coordinate5D coord;
            _par[l].coord(coord, m);
            _model.proj(0).project(image, coord);

            double w = dataVSPrior(image,
                                   _img[l],
                                   _ctf[l],
                                   _sig.col(_groupID[l] - 1).head(_r),
                                   _r);

            _par[l].mulW(w, m);
        }
        _par[l].normW();

        // Save particles
        vec4 q;
        vec2 t;
        for (int m = 0; m < _par[l].N(); m++)
        {
            _par[l].quaternion(q, m);
            _par[l].t(t, m);
            fprintf(file, "%f %f %f %f %f %f %10f\n",
                          q(0),q(1),q(2),q(3),
                          t(0), t(1),
                          _par[l].w(m));
        }
        fclose(file);
    }
}

void MLOptimiser::maximization()
{
    ALOG(INFO) << "Generate Sigma for the Next Iteration";
    allReduceSigma();

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
    for (int l = 0; l < _para.iterMax; l++)
    {
        MLOG(INFO) << "Round " << l;

        MLOG(INFO) << "Performing Expectation";
        expectation();

        MLOG(INFO) << "Performing Maximization";
        maximization();

        /***
        // calculate FSC
        _model.BcastFSC();

        // record current resolution
        _res = _model.resolutionP();

        // update the radius of frequency for computing
        _model.updateR();
        _r = _model.r() / _para.pf;
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
    return _img[0].nColRL();
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
    NT_MASTER _sig.set_size(1 + maxR(), _nGroup);
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

    // perform fourier transformation
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

        // apply a soft mask on it
        softMask(_img.back(),
                 _img.back(),
                 _img.back().nColRL() / 4,
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
        cblas_daxpy(maxR(), 1, ps.memptr(), 1, avgPs.memptr(), 1);
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

    ALOG(INFO) << "Calculating Power Spectrum of Average Image";

    vec psAvg(maxR());
    powerSpectrum(psAvg, avg, maxR());

    // ALOG(INFO) << "Power Spectrum of Average Image is " << endl << psAvg;
    
    // avgPs -> average power spectrum
    // psAvg -> power spectrum of average image
    ALOG(INFO) << "Substract avgPs and psAvg for _sig";

    _sig.head_rows(_sig.n_rows - 1).each_col() = (avgPs - psAvg) / 2;

    // ALOG(INFO) << "Initial Sigma is " << endl << _sig[0];
}

void MLOptimiser::initParticles()
{
    IF_MASTER return;

    FOR_EACH_2D_IMAGE
        _par.push_back(Particle());

    FOR_EACH_2D_IMAGE
        _par[l].init(_para.m,
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

    // set to 0
    _sig.zeros();

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
            _sig.col(_groupID[l] - 1).head(_r) += w * sig;
            _sig.col(_groupID[l] - 1).tail(1) += 1;
        }
    }

    ALOG(INFO) << "Averaging Sigma of Images Belonging to the Same Group";

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _sig.memptr(),
                  _sig.n_elem,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    _sig.each_col([](vec& x){ x /= x(x.n_elem - 1); });

    /***
    // average images belonging to the same group
     
    vector<vec>  groupPowerSpectrum;
    vector<int>  groupSize;
    
    char  sql[1024] = "";

    double*      pAllSigma;
    double*      pMySigma;

    int i, j;

    // all reduce sigma
    int  count;
    
    count = MAX_GROUPID * _r


    pMySigma  = (double *)malloc( sizeof(double) * count );
    if (pMySigma ==NULL )
    {
        REPORT_ERROR("Fail to allocate memory for storing sigma");
        return ;
    }
    pAllSigma = (double *)malloc( sizeof(double) * count );
    if (pAllSigma ==NULL )
    {
        free(pMySigma );
        REPORT_ERROR("Fail to allocate memory for storing sigma");
        return ;
    }
    memset(pMySigma,  0, sizeof(double) * count);
    memset(pAllSigma, 0, sizeof(double) * count);

    groupID.clear();

    MPI_Allreduce( pMySigma, pAllSigma  , count , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for ( i = 0; i < _img.size(); i++)
    {
        for (j=0; j< MAX_POWER_SPECTRUM; j++)
        {
            if (groupSize[i] == 0)
               _sig[i](j)=0;
            else
               _sig[i](j) = *( pMySigma+ groupID[i] * MAX_POWER_SPECTRUM + j) / groupSize[i];
        }
    }    

    free(pMySigma);
    free(pAllSigma);
    ***/
}

void MLOptimiser::reconstructRef()
{
    FOR_EACH_2D_IMAGE
    {
        uvec iSort = _par[l].iSort();

        Coordinate5D coord;
        double w;
        for (int m = 0; m < TOP_K; m++)
        {
            // get coordinate
            _par[l].coord(coord, iSort[m]);
            // get weight
            w = _par[l].w(iSort[m]);

            _model.reco(0).insert(_img[l], coord, w);
        }
    }

    _model.reco(0).reconstruct(_model.ref(0));
}

double dataVSPrior(const Image& dat,
                   const Image& pri,
                   const Image& ctf,
                   const vec& sig,
                   const int r)
{
    double result = 0;

    IMAGE_FOR_EACH_PIXEL_FT(pri)
    {
        int u = AROUND(NORM(i, j));
        if (u < r)
            result += ABS2(dat.getFT(i, j)
                         - ctf.getFT(i, j)
                         * pri.getFT(i, j))
                    / (-2 * sig(u));
    }

    return exp(result);
}

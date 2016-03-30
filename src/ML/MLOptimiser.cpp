/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang
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
    
        ALOG(INFO) << "Estimating Initial Sigma";
        initSigma();

        ALOG(INFO) << "Initialising Particle Filters";
        initParticles();
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
                                   _sig[l],
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
            fprintf(file, "%f %f %f %f, %f %f, %10f\n",
                          q(0),q(1),q(2),q(3),
                          t(0), t(1),
                          _par[l].w(m));
        }
        fclose(file);
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
    MLOG(INFO) << "Initialising MLOptimiser";

    init();

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO) << "Entering Iteration";
    for (int l = 0; l < _para.iterMax; l++)
    {
        MLOG(INFO) << "Round " << l;

        MLOG(INFO) << "Performing Expectation";
        expectation();

        /***
        maximization();

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

    FOR_EACH_2D_IMAGE
        _sig.push_back(vec(maxR()));

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

    _sig[0] = (avgPs - psAvg) / 2;
    FOR_EACH_2D_IMAGE
        _sig[l] = _sig[0];

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

    // loop over 2D images
    FOR_EACH_2D_IMAGE
    {
        // reset sigma to 0
        _sig[l].zeros();

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
            _sig[l] += w * sig;
        }

        // TODO
        // fetch groupID of img[i]
        // average images belonging to the same group
    }
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
        double u = NORM(i, j);
        if (u < r )
            result += ABS2(dat.getFT(i, j)
                         - ctf.getFT(i, j)
                         * pri.getFT(i, j))
                    / (-2 * sig(AROUND(u)));
    }

    return exp(result);
}

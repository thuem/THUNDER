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
        ALOG(INFO) << "Setting up Symmetry";
        _sym.init(_para.sym);

        ALOG(INFO) << "Appending Initial References into _model";
        /***
        // append initial references into _model
        Volume ref;
        // TODO: read in ref
        _model.appendRef(ref);
        ***/

        ALOG(INFO) << "Initialising IDs of 2D Images";
        initID();

        ALOG(INFO) << "Initialising 2D Images";
        /***
        // read in images from hard disk
        // apply soft mask to the images
        // perform Fourier transform
        initImg();
        ***/

        ALOG(INFO) << "Applying Low Pass Filter on Initial References";
        // apply low pass filter on initial references
        /***
        _model.lowPassRef(_r, EDGE_WIDTH_FT);

        // set paramters: _N, _r, _iter
        allReduceN();
        _r = maxR() / 8; // start from 1 / 8 of highest frequency
        _iter = 0;

        // genereate corresponding CTF
        initCTF();
    
        // estimate initial sigma values
        initSigma();

        // initialise a particle filter for each 2D image
        initParticles();
        ***/
    }
}

void MLOptimiser::expectation()
{
    Image image(size(),
                size(),
                FT_SPACE);

    for (int i = 0; i < _img.size(); i++)
    {
        stringstream ss;
        ss << "Particle" << i << ".par";
        FILE* file = fopen(ss.str().c_str(), "w");
        ss.str("");

        if (_par[i].neff() < _par[i].N() / 3)
            _par[i].resample();
        else 
            _par[i].perturb();

        for (int j = 0; j < _par[i].N(); j++)
        {
            Coordinate5D coord;
            _par[i].coord(coord, j);
            _model.proj(0).project(image, coord);

            double w = dataVSPrior(image,
                                   _img[i],
                                   _ctf[i],
                                   _sig[i],
                                   _r);

            _par[i].mulW(w, j);
        }
        _par[i].normW();

        // Save particles
        vec4 q;
        vec2 t;
        for (int k = 0; k < _par[i].N(); k++)
        {
            _par[i].quaternion(q, k);
            _par[i].t(t, k);
            fprintf(file, "%f %f %f %f, %f %f, %f\n",
                          q(0),q(1),q(2),q(3),
                          t(0), t(1),
                          _par[i].w(k));
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

    MLOG(INFO) << "Entering Iteration";
    for (int i = 0; i < _para.iterMax; i++)
    {
        MLOG(INFO) << "Round " << i;

        expectation();

        maximization();

        /* calculate FSC */
        _model.BcastFSC();

        /* record current resolution */
        _res = _model.resolutionP();

        /* update the radius of frequency for computing */
        _model.updateR();
        _r = _model.r() / _para.pf;
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
    if (_commRank == MASTER_ID) return;

    _N = _exp.nParticle();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &_N, 1, MPI_INT, MPI_SUM, _hemi);

    MPI_Barrier(MPI_COMM_WORLD);
}

int MLOptimiser::size() const
{
    return _img[0].nColRL();
}

int MLOptimiser::maxR() const
{
    return size() / 2 - 1;
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
    char sql[SQL_COMMAND_LENGTH];
    
    for (int i = 0; i < _ID.size(); i++)
    {
        // TODO: read the image from hard disk
        /***
        = "select Name from particles;";
        char imgName[FILE_NAME_LENGTH]
        _exp.execute(sql,
                 SQLITE3_CALLBACK
                 {
                    ((vector<int>*)data)-push_back(atoi(values[0]));
                    return 0;
                 },
                 &_ID);
                 ***/

        // TODO: apply a soft mask on it
        
        // TODO: perform Fourier Transform
    }
}

void MLOptimiser::initCTF()
{
    // get CTF attributes from _exp
    char sql[SQL_COMMAND_LENGTH];

    CTFAttr ctfAttr;

    for (int i = 0; i < _ID.size(); i++)
    {
        // get attributes of CTF from database
        sprintf(sql,
                "select (Voltage, DefocusU, DefocusV, DefocusAngle, CS) from \
                 micrographs, particles where \
                 particles.micrographID = micrographs.ID and \
                 particles.ID = %d;",
                _ID[i]);
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
        CTF(*_ctf.end(),
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
    for (int i = 1; i < _img.size(); i++)
        ADD_FT(avg, _img[i]);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  &avg[0],
                  avg.sizeFT(),
                  MPI_DOUBLE_COMPLEX,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(MPI_COMM_WORLD);

    SCALE_RL(avg, 1.0 / _N);

    ALOG(INFO) << "Calculating Average Power Spectrum";

    vec avgPs(maxR(), fill::zeros);
    vec ps(maxR());
    for (int i = 0; i < _img.size(); i++)
    {
        powerSpectrum(ps, _img[i], maxR());
        cblas_daxpy(maxR(), 1, ps.memptr(), 1, avgPs.memptr(), 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs.memptr(),
                  maxR(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(MPI_COMM_WORLD);

    avgPs /= _N;

    ALOG(INFO) << "Calculating Power Spectrum of Average Image";

    vec psAvg(maxR());
    powerSpectrum(psAvg, avg, maxR());
    
    /* avgPs -> average power spectrum
     * psAvg -> power spectrum of average image */
    ALOG(INFO) << "Substract avgPs and psAvg for _sig";

    _sig[0] = (avgPs - psAvg) / 2;
    for (int i = 1; i < _sig.size(); i++)
        _sig[i] = _sig[0];
}

void MLOptimiser::initParticles()
{
    for (int i = 0; i < _img.size(); i++)
    {
        _par.push_back(Particle());
        _par.end()->init(_para.m,
                         _para.maxX,
                         _para.maxY,
                         &_sym);
    }
}


#define  MAX_GROUPID          300
#define  MAX_POWER_SPECTRUM   100

void MLOptimiser::allReduceSigma()
{


    vector<vec>  groupPowerSpectrum;
    vector<int>  groupSize;
    
    char  sql[1024] = "";

    double*      pAllSigma;
    double*      pMySigma;

    int i, j;

    // all reduce sigma
    int  count;
    
    count = MAX_GROUPID * _r  /* MAX_POWER_SPECTRUM */; 


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

    // loop over 2D images
    for (int i = 0; i < _img.size(); i++)
    {
        // reset sigma to 0
        _sig[i].zeros();

        // sort weights in particle and store its indices
        uvec iSort = _par[i].iSort();

        Coordinate5D coord;
        double w;
        Image img(size(), size(), FT_SPACE);
        vec sig(_r);

        // loop over sampling points with top K weights
        for (int j = 0; j < TOP_K; j++)
        {
            // get coordinate
            _par[i].coord(coord, iSort[j]);
            // get weight
            w = _par[i].w(iSort[j]);

            // calculate differences
            _model.proj(0).project(img, coord);
            MUL_FT(img, _ctf[i]);
            NEG_FT(img);
            ADD_FT(img, _img[i]);

            powerSpectrum(sig, img, _r);

            // sum up the results from top K sampling points
            _sig[i] += w * sig;
        }

        // TODO
        // fetch groupID of img[i] -> _par[i]
        //int  groudId=0;

        sprintf(sql, "select GroupID from particles where ID= %d ;", _ID[i] );
        _exp.execute(sql,
                     SQLITE3_CALLBACK
                     {
                        *((int*)data)= atoi(values[0]);
                        return 0;
                     },
                     &groupID[i]); 

        for (j=0; j< MAX_POWER_SPECTRUM; j++)
        {
            pMySigma[groupID[i] * MAX_POWER_SPECTRUM + j] += _sig[i][j];
        };

        // average images belonging to the same group
    }
    
   

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

}


void MLOptimiser::reconstructRef()
{
    // TODO
}

double dataVSPrior(const Image& A,
                   const Image& B,
                   const Image& ctf,
                   const vec& sig,
                   const int r)
{
    // Todo
}


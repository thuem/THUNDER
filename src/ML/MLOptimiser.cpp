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

void MLOptimiser::init()
{
    // set MPI environment of _model
    _model.setMPIEnv(_commSize, _commRank, _hemi);

    // set paramters: _N, _r, _iter
    allReduceN();
    _r = maxR() / 8; // start from 1 / 8 of highest frequency
    _iter = 0;

    // initialise symmetry
    _sym.init(_para.sym);

    // append initial references into _model
    Volume ref;
    // TODO: read in ref
    _model.appendRef(ref);

    // apply low pass filter on initial references
    _model.lowPassRef(_r, EDGE_WIDTH_FT);

    // read in images from hard disk
    // apply soft mask to the images
    // perform Fourier transform
    initImg();

    // genereate corresponding CTF
    initCTF();
    
    // estimate initial sigma values
    initSigma();

    // initialise a particle filter for each 2D image
    initParticles();
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
    if (_commRank == MASTER_ID) return;

    /* calculate average image */

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

    /* calculate average power spectrum */

    double* avgPs = new double[maxR()];
    memset(avgPs, 0, maxR() * sizeof(double));

    double* ps = new double[maxR()];
    vec vps(ps, maxR(), false, true);
    for (int i = 0; i < _img.size(); i++)
    {
        powerSpectrum(vps, _img[i], maxR());
        cblas_daxpy(maxR(), 1, ps, 1, avgPs, 1);
    }
    delete[] ps;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs,
                  maxR(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(MPI_COMM_WORLD);

    vec vAvgPs(avgPs, maxR());
    delete[] avgPs;
    vAvgPs /= _N;

    vec vPsAvg(maxR());
    powerSpectrum(vPsAvg, avg, maxR());
    
    /* vAvgPs -> average power spectrum
     * vPsAvg -> power spectrum of average image */
    _sig[0] = (vAvgPs - vPsAvg) / 2;
    for (int i = 1; i < _sig.size(); i++)
        _sig[i] = _sig[0];
}

void MLOptimiser::initParticles()
{
    for (int i = 0; i < _img.size(); i++)
    {
        _par.push_back(Particle());
        _par.end()->init(_para.M,
                         _para.maxX,
                         _para.maxY,
                         &_sym);
    }
}

void MLOptimiser::allReduceSigma()
{
    int  k;
    int  i, j;
    double  sigma;
    double  phi;
    double  ctf;
    double  p;
    double  v;
    double  Ka;
    double  noise;
    double  deltaPhi =0;

    sigma = 0;
    deltaPhi = 0.1 ;  // ????
    noise =0;

    #if 0
    for (k=0; k< K();  k++ )
    {
        for (phi = 0; phi < 2*PI; phi += deltaPhi )
        {
            temp = 0;
            for ( l=1; l< L();  l++ )
            {
                p = Pij;
                v = Vkl;                 
            }
            ctf = CTF(i,j);
            temp = Xij - ctf * (p * v);    
            temp = temp * temp;

            ka = Ka(i, k, phi);
            sigma += temp * Ka * deltaPhi;

        }
    }
    #endif 
    sigma = sigma /2;


    // TODO
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


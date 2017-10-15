/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Siyuan Ren
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

Particle::Particle()
{
    defaultInit();
}

Particle::Particle(const int mode,
                   const int nC,
                   const int nR,
                   const int nT,
                   const int nD,
                   const double transS,
                   const double transQ,
                   const Symmetry* sym)
{
    defaultInit();

    init(mode, nC, nR, nT, nD, transS, transQ, sym);
}

Particle::~Particle()
{
    clear();
}

void Particle::init(const int mode,
                    const double transS,
                    const double transQ,
                    const Symmetry* sym)
{
    clear();

    _mode = mode;

    _transS = transS;
    _transQ = transQ;

    _sym = sym;
}

void Particle::init(const int mode,
                    const int nC,
                    const int nR,
                    const int nT,
                    const int nD,
                    const double transS,
                    const double transQ,
                    const Symmetry* sym)
{
    init(mode, transS, transQ, sym);

    _nC = nC;

    _nR = nR;

    _nT = nT;

    _nD = nD;

    _c.resize(_nC);
    _r.resize(_nR, 4);
    _t.resize(_nT, 2);
    _d.resize(_nD);

    _wC.resize(_nC);
    _wR.resize(_nR);
    _wT.resize(_nT);
    _wD.resize(_nD);

    reset();
}

void Particle::reset()
{
    gsl_rng* engine = get_random_engine();

    // initialise class distribution

    /***
    for (int i = 0; i < _nC; i++)
        _c(i) = gsl_rng_uniform_int(engine, _nC);
    ***/

    for (int i = 0; i < _nC; i++)
        _c(i) = i;

    // initialise rotation distribution

    switch (_mode)
    {
        // rotation, MODE_2D, sample from von Mises Distribution with kappa = 0
        case MODE_2D:
            sampleVMS(_r, vec4(1, 0, 0, 0), 0, _nR);
            break;

        // rotation, MODE_3D, sample from Angular Central Gaussian Distribution
        // with identity matrix
        case MODE_3D:
            sampleACG(_r, 1, 1, _nR);
            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
            break;
    }


    // initialise translation distribution

    for (int i = 0; i < _nT; i++)
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &_t(i, 0),
                                   &_t(i, 1));

    // initialise defocus distribution

    _d = vec::Constant(_nD, 1);

    // initialise weight

    _wC = vec::Constant(_nC, 1.0 / _nC);
    _wR = vec::Constant(_nR, 1.0 / _nR);
    _wT = vec::Constant(_nT, 1.0 / _nT);
    _wD = vec::Constant(_nD, 1.0 / _nD);

    // symmetrise

    if (_mode == MODE_3D) symmetrise();
}

/***
void Particle::reset(const int m,
                     const int n)
{
    _m = m;

    _n = n;

    _c.resize(_n);
    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _d.resize(_n);

    _w.resize(_n);

    _cDistr.resize(_m);

    reset();
}
***/

void Particle::reset(const int nC,
                     const int nR,
                     const int nT,
                     const int nD)
{
    init(_mode, nC, nR, nT, nD, _transS, _transQ, _sym);
    /***
    gsl_rng* engine = get_random_engine();

    _m = m;

    _c.resize(_n);
    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _d.resize(_n);

    _w.resize(_n);

    _cDistr.resize(_m);

    uvec c(m);

    // sample from 0 to (m - 1)
    for (int i = 0; i < m; i++)
        c(i) = i;

    mat4 r(nR, 4);

    switch (_mode)
    {
        case MODE_2D:
            // sample from von Mises Distribution with kappa = 0
            sampleVMS(r, vec4(1, 0, 0, 0), 0, nR);
            break;

        case MODE_3D:
            // sample from Angular Central Gaussian Distribution with identity matrix
            sampleACG(r, 1, 1, nR);
            break;

        default:
            REPORT_ERROR("INEXISTENT MODE");
            break;
    }

    mat2 t(nT, 2);

#ifdef PARTICLE_TRANS_INIT_GAUSSIAN
    // sample from 2D Gaussian Distribution
    for (int i = 0; i < nT; i++)
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &t(i, 0),
                                   &t(i, 1));
#endif

#ifdef PARTICLE_TRANS_INIT_FLAT
    // sample for 2D Flat Distribution in a Circle
    for (int i = 0; i < nT; i++)
    {
        double r = gsl_ran_flat(engine, 0, _transS);
        double t = gsl_ran_flat(engine, 0, 2 * M_PI);

        t(i, 0) = r * cos(t);
        t(i, 1) = r * sin(t);
    }
#endif

    for (int k = 0; k < m; k++)
        for (int j = 0; j < nR; j++)
            for (int i = 0; i < nT; i++)
            {
                _c(k * nR * nT + j * nT + i) = c(k);

                _r.row(k * nR * nT + j * nT + i) = r.row(j);

                _t.row(k * nR * nT + j * nT + i) = t.row(i);

                _d(k * nR * nT + j * nT + i) = 1;

                _w(k * nR * nT + j * nT + i) = 1.0 / _n;
            }

    if (_mode == MODE_3D) symmetrise();
    ***/
}

void Particle::initD(const int nD,
                     const double sD)
{
    gsl_rng* engine = get_random_engine();

    _nD = nD;

    _d.resize(nD);

    for (int i = 0; i < _nD; i++)
        _d(i) = 1 + gsl_ran_gaussian(engine, sD);

    _wD = vec::Constant(_nD, 1.0 / _nD);
}

int Particle::mode() const { return _mode; }

void Particle::setMode(const int mode) { _mode = mode; }

int Particle::nC() const { return _nC; }

void Particle::setNC(const int nC) { _nC = nC; }

int Particle::nR() const { return _nR; }

void Particle::setNR(const int nR) { _nR = nR; }

int Particle::nT() const { return _nT; }

void Particle::setNT(const int nT) { _nT = nT; }

int Particle::nD() const { return _nD; }

void Particle::setND(const int nD) { _nD = nD; }

double Particle::transS() const { return _transS; }

void Particle::setTransS(const double transS) { _transS = transS; }

double Particle::transQ() const { return _transQ; }

void Particle::setTransQ(const double transQ) { _transQ = transQ; }

uvec Particle::c() const { return _c; }

void Particle::setC(const uvec& c) { _c = c; }

mat4 Particle::r() const { return _r; }

void Particle::setR(const mat4& r) { _r = r; }

mat2 Particle::t() const { return _t; }

void Particle::setT(const mat2& t) { _t = t; }

vec Particle::d() const { return _d; }

void Particle::setD(const vec& d) { _d = d; }

vec Particle::wC() const { return _wC; }

void Particle::setWC(const vec& wC) { _wC = wC; }

vec Particle::wR() const { return _wR; }

void Particle::setWR(const vec& wR) { _wR = wR; }

vec Particle::wT() const { return _wT; }

void Particle::setWT(const vec& wT) { _wT = wT; }

vec Particle::wD() const { return _wD; }

void Particle::setWD(const vec& wD) { _wD = wD; }

const Symmetry* Particle::symmetry() const { return _sym; }

void Particle::setSymmetry(const Symmetry* sym) { _sym = sym; }

void Particle::load(const int nR,
                    const int nT,
                    const int nD,
                    const vec4& quat,
                    const double stdR,
                    const vec2& tran,
                    const double stdTX,
                    const double stdTY,
                    const double d,
                    const double stdD)
{
    _nC = 1;
    _nR = nR;
    _nT = nT;
    _nD = nD;

    _c.resize(1);
    _wC.resize(1);

    _c(0) = 0;

    _wC(0) = 1;

    _topCPrev = 0;
    _topC = 0;

    _r.resize(_nR, 4);
    _t.resize(_nT, 2);
    _d.resize(_nD);

    _wR.resize(_nR);
    _wT.resize(_nT);
    _wD.resize(_nD);

    gsl_rng* engine = get_random_engine();

    // load the rotation
    
    _k0 = 1;
    
    _k1 = gsl_pow_2(stdR);
    
    _topRPrev = quat;
    _topR = quat;

    // mat4 p(_nR, 4);

    sampleACG(_r, _k0, _k1, _nR);

    //sampleACG(p, 1, gsl_pow_2(stdR), _nR);

    if (_mode == MODE_3D) symmetrise();
    
    for (int i = 0; i < _nR; i++)
    {
        vec4 pert = _r.row(i).transpose();

        vec4 part;

        /***
        if (gsl_ran_flat(engine, -1, 1) >= 0)
            quaternion_mul(part, quat, pert);
        else
            quaternion_mul(part, -quat, pert);
        ***/

        if (gsl_ran_flat(engine, -1, 1) >= 0)
            quaternion_mul(part, pert, quat);
        else
            quaternion_mul(part, pert, -quat);

        _r.row(i) = part.transpose();

        _wR(i) = 1.0 / _nR;
    }

    // load the translation

    _s0 = stdTX;
    _s1 = stdTY;

    _topTPrev = tran;
    _topT = tran;

    for (int i = 0; i < _nT; i++)
    {
       gsl_ran_bivariate_gaussian(engine,
                                  _s0,
                                  _s1,
                                  0,
                                  &_t(i, 0),
                                  &_t(i, 1));

       _t(i, 0) += tran(0);
       _t(i, 1) += tran(1);

       _wT(i) = 1.0 / _nT;
    }

    // load the defocus factor

    _s = stdD;
    
    _topDPrev = d;
    _topD = d;

    for (int i = 0; i < _nD; i++)
    {
        _d(i) = d + gsl_ran_gaussian(engine, _s);
        _wD(i) = 1.0 / _nD;
    }

}

void Particle::vari(double& k0,
                    double& k1,
                    double& s0,
                    double& s1,
                    double& rho,
                    double& s) const
{
    k0 = _k0;
    k1 = _k1;
    s0 = _s0;
    s1 = _s1;
    rho = _rho;
    s = _s;
}

void Particle::vari(double& rVari,
                    double& s0,
                    double& s1,
                    double& s) const
{
    switch (_mode)
    {
        case MODE_2D:
            rVari = 1.0 / (1 + _k); // TODO: it is a approximation
            break;

        case MODE_3D:
            /***
            if (_k0 == 0) CLOG(FATAL, "LOGGER_SYS") << "k0 = 0";
            if (gsl_isnan(_k0)) CLOG(FATAL, "LOGGER_SYS") << "k0 NAN";
            if (gsl_isnan(_k1)) CLOG(FATAL, "LOGGER_SYS") << "k1 NAN";
            ***/
            // more cencentrate, smaller rVari, bigger _k0 / _k1;

            rVari = sqrt(_k1) / sqrt(_k0);

            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
            break;
    }

    s0 = _s0;
    s1 = _s1;

    s = _s;
}

double Particle::compress() const
{
    // return _transS / sqrt(_s0 * _s1);

    // return pow(_k0 / _k1, 1.5);

    return sqrt(_k0) / sqrt(_k1);

    // return _k0 / _k1;

    // return pow(_k0 / _k1, 1.5) * gsl_pow_2(_transS) / _s0 / _s1;

    //return gsl_pow_2(_transS) / _s0 / _s1;
}

double Particle::wC(const int i) const
{
    return _wC(i);
}

void Particle::setWC(const double wC,
                     const int i)
{
    _wC(i) = wC;
}

void Particle::mulWC(const double wC,
                     const int i)
{
    _wC(i) *= wC;
}

double Particle::wR(const int i) const
{
    return _wR(i);
}

void Particle::setWR(const double wR,
                     const int i)
{
    _wR(i) = wR;
}

void Particle::mulWR(const double wR,
                     const int i)
{
    _wR(i) *= wR;
}

double Particle::wT(const int i) const
{
    return _wT(i);
}

void Particle::setWT(const double wT,
                     const int i)
{
    _wT(i) = wT;
}

void Particle::mulWT(const double wT,
                     const int i)
{
    _wT(i) *= wT;
}

double Particle::wD(const int i) const
{
    return _wD(i);
}

void Particle::setWD(const double wD,
                     const int i)
{
    _wD(i) = wD;
}

void Particle::mulWD(const double wD,
                     const int i)
{
    _wD(i) *= wD;
}

void Particle::normW()
{
    _wC /= _wC.sum();
    _wR /= _wR.sum();
    _wT /= _wT.sum();
    _wD /= _wD.sum();
}

/***
void Particle::coord(Coordinate5D& dst,
                     const int i) const
{
    vec4 quat = _r.row(i).transpose();
    angle(dst.phi,
          dst.theta,
          dst.psi,
          quat);

    dst.x = _t(i, 0);
    dst.y = _t(i, 1);
}
***/

void Particle::c(unsigned int& dst,
                 const int i) const
{
    dst = _c(i);
}

void Particle::setC(const unsigned int src,
                    const int i)
{
    _c(i) = src;
}

void Particle::rot(mat22& dst,
                   const int i) const
{
    rotate2D(dst, vec2(_r(i, 0), _r(i, 1)));
}

void Particle::rot(mat33& dst,
                   const int i) const
{
    rotate3D(dst, _r.row(i).transpose());
}

void Particle::t(vec2& dst,
                 const int i) const
{
    dst = _t.row(i).transpose();
}

void Particle::setT(const vec2& src,
                    const int i)
{
    _t.row(i) = src.transpose();
}

void Particle::quaternion(vec4& dst,
                          const int i) const
{
    dst = _r.row(i).transpose();
}

void Particle::setQuaternion(const vec4& src,
                             const int i) 
{
    _r.row(i) = src.transpose();
}

void Particle::d(double& d,
                 const int i) const
{
    d = _d(i);
}

void Particle::setD(const double d,
                    const int i)
{
    _d(i) = d;
}

double Particle::k0() const
{
    return _k0;
}

void Particle::setK0(const double k0)
{
    _k0 = k0;
}

double Particle::k1() const
{
    return _k1;
}

void Particle::setK1(const double k1)
{
    _k1 = k1;
}

double Particle::s0() const
{
    return _s0;
}

void Particle::setS0(const double s0)
{
    _s0 = s0;
}

double Particle::s1() const
{
    return _s1;
}

void Particle::setS1(const double s1)
{
    _s1 = s1;
}

/***
void Particle::calClassDistr()
{
    _cDistr.setZero();

    for (int i = 0; i < _n; i++)
        _cDistr(_c(i)) += 1;
}
***/

void Particle::calVari(const ParticleType pt)
{

    if (pt == PAR_C)
    {
        CLOG(WARNING, "LOGGER") << "NO NEED TO CALCULATE VARIANCE IN CLASS";
    }
    else if (pt == PAR_R)
    {
        if (_mode == MODE_2D)
            inferVMS(_k, _r);
        else if (_mode == MODE_3D)
            inferACG(_k0, _k1, _r);
        else
            REPORT_ERROR("INEXISTENT MODE");

        _k1 /= _k0;

        _k1 = GSL_MIN_DBL(_k1, 1);

        _k0 = 1;
    }
    else if (pt == PAR_T)
    {
#ifdef PARTICLE_CAL_VARI_TRANS_ZERO_MEAN
        _s0 = gsl_stats_sd_m(_t.col(0).data(), 1, _t.rows(), 0);
        _s1 = gsl_stats_sd_m(_t.col(1).data(), 1, _t.rows(), 0);
#else
        _s0 = gsl_stats_sd(_t.col(0).data(), 1, _t.rows());
        _s1 = gsl_stats_sd(_t.col(1).data(), 1, _t.rows());
#endif

        _rho = 0;
    }
    else if (pt == PAR_D)
    {
        _s = gsl_stats_sd(_d.data(), 1, _d.size());
    }
}

void Particle::perturb(const double pf,
                       const ParticleType pt)
{
    if (pt == PAR_C)
    {
        CLOG(WARNING, "LOGGER_SYS") << "NO NEED TO PERFORM PERTURBATION IN CLASS";
    }
    else if (pt == PAR_R)
    {
        mat4 d(_nR, 4);

        if (_mode == MODE_2D)
            sampleVMS(d, vec4(1, 0, 0, 0), _k / pf, _nR);
        else if (_mode == MODE_3D)
            sampleACG(d, _k0, GSL_MIN_DBL(_k0, pow(pf, 2) * _k1), _nR);

        vec4 mean;

        inferACG(mean, _r);

        vec4 quat;

        for (int i = 0; i < _nR; i++)
        {
            quat = _r.row(i).transpose();

            quaternion_mul(quat, quaternion_conj(mean), quat);

            _r.row(i) = quat.transpose();
        }

        vec4 pert;
           
        for (int i = 0; i < _nR; i++)
        {
            quat = _r.row(i).transpose();
            pert = d.row(i).transpose();
            quaternion_mul(quat, pert, quat);
            _r.row(i) = quat.transpose();
        }

        if (_mode == MODE_3D) symmetrise();

        for (int i = 0; i < _nR; i++)
        {
            quat = _r.row(i).transpose();

            quaternion_mul(quat, mean, quat);

            _r.row(i) = quat.transpose();
        }
    }
    else if (pt == PAR_T)
    {
        gsl_rng* engine = get_random_engine();

        for (int i = 0; i < _nT; i++)
        {
            double x, y;

            gsl_ran_bivariate_gaussian(engine, _s0, _s1, _rho, &x, &y);

            _t(i, 0) += x * pf;
            _t(i, 1) += y * pf;
        }

        reCentre();
    }
    else if (pt == PAR_D)
    {
        gsl_rng* engine = get_random_engine();

        for (int i = 0; i < _nD; i++)
            _d(i) += gsl_ran_gaussian(engine, _s) * pf;
    }
}

/***
void Particle::perturb(const double pfR,
                       const double pfT,
                       const double pfD)
{
#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Rotation Perturbation";
#endif

    perturb(pfR, PAR_R);

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Translation Perturbation";
#endif

    perturb(pfT, PAR_T);

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Defocus Factor Perturbation";
#endif

    perturb(pfD, PAR_D);
}
***/

void Particle::resample(const int n,
                        const ParticleType pt)
{
    gsl_rng* engine = get_random_engine();

    uvec rank = iSort(pt);

    if (pt == PAR_C)
    {
        c(_topC, rank(0));

        shuffle(pt);

        /***
        if (n != 1)
        {
            REPORT_ERROR("ONLY KEEP ONE CLASS");
            abort();
        }

        _nC = 1;

        _c.resize(1);
        _c(0) = _topC;
        
        _wC.resize(1);
        _wC(0) = 1;
        ***/

        vec cdf = cumsum(_wC);
        
        _nC = n;
        _wC.resize(_nC);

        uvec c(_nC);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nC);  

        int i = 0;
        for (int j = 0; j < _nC; j++)
        {
            double uj = u0 + j * 1.0 / _nC;

            while (uj > cdf[i])
                i++;

            c(j) = _c(i);
        
            _wC(j) = 1.0 / _nC;
        }

        _c = c;
    }
    else if (pt == PAR_R)
    {
        quaternion(_topR, rank(0));

        shuffle(pt);

        vec cdf = cumsum(_wR);

        _nR = n;
        _wR.resize(_nR);

        mat4 r(_nR, 4);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nR);  

        int i = 0;
        for (int j = 0; j < _nR; j++)
        {
            double uj = u0 + j * 1.0 / _nR;

            while (uj > cdf[i])
                i++;
        
            r.row(j) = _r.row(i);

            _wR(j) = 1.0 / _nR;
        }

        _r = r;
    }
    else if (pt == PAR_T)
    {
        t(_topT, rank(0));

        shuffle(pt);

        vec cdf = cumsum(_wT);

        _nT = n;
        _wT.resize(_nT);

        mat2 t(_nT, 2);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nT);  

        int i = 0;
        for (int j = 0; j < _nT; j++)
        {
            double uj = u0 + j * 1.0 / _nT;

            while (uj > cdf[i])
                i++;
        
            t.row(j) = _t.row(i);

            _wT(j) = 1.0 / _nT;
        }

        _t = t;
    }
    else if (pt == PAR_D)
    {
        d(_topD, rank(0));

        shuffle(pt);

        vec cdf = cumsum(_wD);

        _nD = n;
        _wD.resize(_nD);

        vec d(_nD);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nD);  

        int i = 0;
        for (int j = 0; j < _nD; j++)
        {
            double uj = u0 + j * 1.0 / _nD;

            while (uj > cdf[i])
                i++;

            d(j) = _d(i);

            _wD(j) = 1.0 / _nD;
        }

        _d = d;
    }
}

/***
void Particle::resample(const int nR,
                        const int nT,
                        const int nD)
{
    resample(_nC, PAR_C);
    resample(nR, PAR_R);
    resample(nT, PAR_T);
    resample(nD, PAR_D);
}
***/

/***
void Particle::resample()
{
    resample(_nR, _nT, _nD);
}
***/

/***
void Particle::resample(const double alpha)
{
    resample(_n, alpha);
}

void Particle::resample(const int n,
                        const double alpha)
{
#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Recording the Current Most Likely Coordinate";
#endif

    uvec rank = iSort();

    c(_topC, rank(0));
    quaternion(_topR, rank(0));
    t(_topT, rank(0));
    d(_topD, rank(0));

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Performing Shuffling";
#endif

    shuffle();

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Performing Resampling";
#endif

    vec cdf = cumsum(_w);

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Recording New Number of Sampling Points";
#endif

    _n = n;
    _w.resize(n);

    // number of global sampling points
    int nG = AROUND(alpha * n);

    // number of local sampling points
    int nL = n - nG;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Allocating Temporary Storage";
#endif

    uvec c(n);
    mat4 r(n, 4);
    mat2 t(n, 2);
    vec d(n);
    
#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points";
#endif

    gsl_rng* engine = get_random_engine();

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Class";
#endif

    for (int i = 0; i < nG; i++)
        c(i) = gsl_rng_uniform_int(engine, _m);

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Rotation";
#endif

    switch (_mode)
    {
        case MODE_2D:
            sampleVMS(r, vec4(1, 0, 0, 0), 0, nG);
            break;

        case MODE_3D:
            sampleACG(r, 1, 1, nG);
            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Translation";
#endif
    
    for (int i = 0; i < nG; i++)
    {
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &t(i, 0),
                                   &t(i, 1));
                
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Defocus Factor";
#endif

    for (int i = 0; i < nG; i++)
        _d(i) = 1;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Weights for Global Sampling Points";
#endif

    for (int i = 0; i < nG; i++)
        _w(i) = 1.0 / n;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Local Sampling Points";
#endif

    double u0 = gsl_ran_flat(engine, 0, 1.0 / nL);  

    int i = 0;
    for (int j = 0; j < nL; j++)
    {
        double uj = u0 + j * 1.0 / nL;

        while (uj > cdf[i])
            i++;
        
        c(nG + j) = _c(i);

        r.row(nG + j) = _r.row(i);

        t.row(nG + j) = _t.row(i);

        d(nG + j) = _d(i);

        _w(nG + j) = 1.0 / n;
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Recording Results";
#endif

    _c.resize(n);
    _c = c;

    _r.resize(n, 4);
    _r = r;

    _t.resize(n, 2);
    _t = t;

    _d.resize(n);
    _d = d;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Symmetrizing";
#endif
    
    if (_mode == MODE_3D) symmetrise();
}
***/

/***
double Particle::neff() const
{
    return 1.0 / _w.squaredNorm();
}

void Particle::segment(const double thres)
{
    uvec order = iSort();

    double s = 0;
    int i;

    for (i = 0; i < _n; i++)
    {
        s += _w(order(i));

        if (s > thres)
            break;
    }

    int n = _n;

    sort(i + 1);

    resample(n);
}

void Particle::flatten(const double thres)
{
    uvec order = iSort();

    double s = 0;
    int i;

    for (i = 0; i < _n; i++)
    {
        s += _w(order(i));

        if (s > thres)
            break;
    }

    int n = _n;

    sort(i + 1);

    for (int i = 0; i < _n; i++)
        _w(i) = 1.0 / _n;

    resample(n);
}

void Particle::sort()
{
    sort(_n);
}

void Particle::sort(const int n)
{
    if (n > _n)
        REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

    uvec order = iSort();

    uvec c(n);
    mat4 r(n, 4);
    mat2 t(n, 2);
    vec d(n);
    vec w(n);

    for (int i = 0; i < n; i++)
    {
        c(i) = _c(order(i));
        r.row(i) = _r.row(order(i));
        t.row(i) = _t.row(order(i));
        d(i) = _d(order(i));
        w(i) = _w(order(i));
    }

    _n = n;

    _c = c;
    _r = r;
    _t = t;
    _d = d;
    _w = w;

    // normalise weight again
    normW();
}
***/

void Particle::sort(const int n,
                    const ParticleType pt)
{
    uvec order = iSort(pt);

    if (pt == PAR_C)
    {
        if (n > _nC)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        uvec c(n);
        vec wC(n);

        for (int i = 0; i < n; i++)
        {
            c(i) = _c(order(i));
            wC(i) = _wC(order(i));
        }

        _nC = n;
        
        _c = c;
        _wC = wC;
    }
    else if (pt == PAR_R)
    {
        if (n > _nR)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        mat4 r(n, 4);
        vec wR(n);

        for (int i = 0; i < n; i++)
        {
            r.row(i) = _r.row(order(i));
            wR(i) = _wR(order(i));
        }

        _nR = n;
        
        _r = r;
        _wR = wR;
    }
    else if (pt == PAR_T)
    {
        if (n > _nT)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        mat2 t(n, 2);
        vec wT(n);

        for (int i = 0; i < n; i++)
        {
            t.row(i) = _t.row(order(i));
            wT(i) = _wT(order(i));
        }

        _nT = n;
        
        _t = t;
        _wT = wT;
    }
    else if (pt == PAR_D)
    {
        if (n > _nD)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        vec d(n);
        vec wD(n);

        for (int i = 0; i < n; i++)
        {
            d(i) = _d(order(i));
            wD(i) = _wD(order(i));
        }

        _nD = n;
        
        _d = d;
        _wD = wD;
    }
}

void Particle::sort(const int nC,
                    const int nR,
                    const int nT,
                    const int nD)
{
    sort(nC, PAR_C);
    sort(nR, PAR_R);
    sort(nT, PAR_T);
    sort(nD, PAR_D);
}

void Particle::sort()
{
    sort(_nC, _nR, _nT, _nD);
}

uvec Particle::iSort(const ParticleType pt) const
{
    if (pt == PAR_C)
        return index_sort_descend(_wC);
    else if (pt == PAR_R)
        return index_sort_descend(_wR);
    else if (pt == PAR_T)
        return index_sort_descend(_wT);
    else if (pt == PAR_D)
        return index_sort_descend(_wD);
    else abort();
}

bool Particle::diffTopC()
{
    bool diff = (_topCPrev == _topC);

    _topCPrev = _topC;

    return diff;
}

double Particle::diffTopR()
{
    double diff = 1 - fabs(_topRPrev.dot(_topR));

    _topRPrev = _topR;

    return diff;
}

double Particle::diffTopT()
{
    double diff = (_topTPrev - _topT).norm();

    _topTPrev = _topT;

    return diff;
}

double Particle::diffTopD()
{
    double diff = fabs(_topDPrev - _topD);

    _topDPrev = _topD;

    return diff;
}

void Particle::rank1st(unsigned int& cls) const
{
    cls = _topC;
}

void Particle::rank1st(vec4& quat) const
{
    quat = _topR;
}

void Particle::rank1st(mat22& rot) const
{
    vec4 quat;
    rank1st(quat);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rank1st(mat33& rot) const
{
    vec4 quat;
    rank1st(quat);

    rotate3D(rot, quat);
}

void Particle::rank1st(vec2& tran) const
{
    tran = _topT;
}

void Particle::rank1st(double& df) const
{
    df = _topD;
}

void Particle::rank1st(unsigned int& cls,
                       vec4& quat,
                       vec2& tran,
                       double& df) const
{
    cls = _topC;
    quat = _topR;
    tran = _topT;
    df = _topD;
}

void Particle::rank1st(unsigned int& cls,
                       mat22& rot,
                       vec2& tran,
                       double& df) const
{
    vec4 quat;
    rank1st(cls, quat, tran, df);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rank1st(unsigned int& cls,
                       mat33& rot,
                       vec2& tran,
                       double& df) const
{
    vec4 quat;
    rank1st(cls, quat, tran, df);

    rotate3D(rot, quat);
}

void Particle::rand(unsigned int& cls) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _nC);

    c(cls, u);
}

void Particle::rand(vec4& quat) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _nR);

    quaternion(quat, u);
}

void Particle::rand(mat22& rot) const
{
    vec4 quat;
    rand(quat);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rand(mat33& rot) const
{
    vec4 quat;
    rand(quat);

    rotate3D(rot, quat);
}

void Particle::rand(vec2& tran) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _nT);

    t(tran, u);
}

void Particle::rand(double& df) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _nD);

    d(df, u);
}

void Particle::rand(unsigned int& cls,
                    vec4& quat,
                    vec2& tran,
                    double& df) const
{
    rand(cls);
    rand(quat);
    rand(tran);
    rand(df);
    /***
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _n);

    c(cls, u);
    quaternion(quat, u);
    t(tran, u);
    d(df, u);
    ***/
}

void Particle::rand(unsigned int& cls,
                    mat22& rot,
                    vec2& tran,
                    double& df) const
{
    vec4 quat;
    rand(cls, quat, tran, df);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rand(unsigned int& cls,
                    mat33& rot,
                    vec2& tran,
                    double& df) const
{
    vec4 quat;
    rand(cls, quat, tran, df);

    rotate3D(rot, quat);
}

void Particle::shuffle(const ParticleType pt)
{
    gsl_rng* engine = get_random_engine();

    if (pt == PAR_C)
    {
        // CLOG(WARNING, "LOGGER_SYS") << "NO NEED TO PERFORM SHUFFLE IN CLASS";

        uvec s = uvec(_nC);

        for (int i = 0; i < _nC; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nC, sizeof(unsigned int));

        uvec c(_nC);

        vec wC(_nC);

        for (int i = 0; i < _nC; i++)
        {
            c(s(i)) = _c(i);
            wC(s(i)) = _wC(i);
        }

        _c = c;
        _wC = wC;
    }
    else if (pt == PAR_R)
    {
        uvec s = uvec(_nR);

        for (int i = 0; i < _nR; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nR, sizeof(unsigned int));

        mat4 r(_nR, 4);
        vec wR(_nR);

        for (int i = 0; i < _nR; i++)
        {
            r.row(s(i)) = _r.row(i);
            wR(s(i)) = _wR(i);
        }

        _r = r;
        _wR = wR;
    }
    else if (pt == PAR_T)
    {
        uvec s = uvec(_nT);

        for (int i = 0; i < _nT; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nT, sizeof(unsigned int));

        mat2 t(_nT, 2);
        vec wT(_nT);

        for (int i = 0; i < _nT; i++)
        {
            t.row(s(i)) = _t.row(i);
            wT(s(i)) = _wT(i);
        }

        _t = t;
        _wT = wT;
    }
    else if (pt == PAR_D)
    {
        uvec s = uvec(_nD);

        for (int i = 0; i < _nD; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nD, sizeof(unsigned int));

        vec d(_nD);
        vec wD(_nD);

        for (int i = 0; i < _nD; i++)
        {
            d(s(i)) = _d(i);
            wD(s(i)) = _wD(i);
        }

        _d = d;
        _wD = wD;
    }
}

void Particle::shuffle()
{
    shuffle(PAR_R);
    shuffle(PAR_T);
    shuffle(PAR_D);
    /***
    uvec s = uvec(_n);

    for (int i = 0; i < _n; i++) s(i) = i;

    gsl_rng* engine = get_random_engine();

    gsl_ran_shuffle(engine, s.data(), _n, sizeof(unsigned int));

    uvec c(_n);
    mat4 r(_n, 4);
    mat2 t(_n, 2);
    vec d(_n);
    vec w(_n);

    for (int i = 0; i < _n; i++)
    {
        c(s(i)) = _c(i);
        r.row(s(i)) = _r.row(i);
        t.row(s(i)) = _t.row(i);
        d(s(i)) = _d(i);
        w(s(i)) = _w(i);
    }

    _c = c;
    _r = r;
    _t = t;
    _d = d;
    _w = w;
    ***/
}

void Particle::copy(Particle& that) const
{
    that.setNC(_nC);
    that.setNR(_nR);
    that.setNT(_nT);
    that.setND(_nD);
    that.setTransS(_transS);
    that.setTransQ(_transQ);
    that.setC(_c);
    that.setR(_r);
    that.setT(_t);
    that.setD(_d);
    that.setWC(_wC);
    that.setWR(_wR);
    that.setWT(_wT);
    that.setWD(_wD);
    that.setSymmetry(_sym);
}

Particle Particle::copy() const
{
    Particle that;

    copy(that);

    return that;
}

void Particle::symmetrise()
{
    if (_sym == NULL) return;

    if (asymmetry(*_sym)) return;

    /***
    vec4 mean;

    inferACG(mean, _r);
    ***/

    vec4 quat;

    for (int i = 0; i < _nR; i++)
    {
        vec4 quat = _r.row(i).transpose();

        // quaternion_mul(quat, quaternion_conj(mean), quat);

        symmetryCounterpart(quat, *_sym);

        // quaternion_mul(quat, mean, quat);

        _r.row(i) = quat.transpose();
    }
}

void Particle::reCentre()
{
    double transM = _transS * gsl_cdf_chisq_Qinv(_transQ, 2);

    gsl_rng* engine = get_random_engine();

    for (int i = 0; i < _nT; i++)
        if (NORM(_t(i, 0), _t(i, 1)) > transM)
        {
            // _t.row(i) *= transM / NORM(_t(i, 0), _t(i, 1));

            gsl_ran_bivariate_gaussian(engine,
                                       _transS,
                                       _transS,
                                       0,
                                       &_t(i, 0),
                                       &_t(i, 1));
        }
}

void Particle::clear() {}

void display(const Particle& par)
{
    unsigned int c;
    vec4 q;
    vec2 t;
    double d;

    FOR_EACH_PAR(par)
    {
        par.c(c, iC);
        par.quaternion(q, iR);
        par.t(t, iT);
        par.d(d, iD);
        printf("%03d %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
               c,
               q(0), q(1), q(2), q(3),
               t(0), t(1),
               d,
               par.wC(iC) * par.wR(iR) * par.wT(iT) * par.wD(iD));
    }
}

void save(const char filename[],
          const Particle& par)
{
    FILE* file = fopen(filename, "w");

    unsigned int c;
    vec4 q;
    vec2 t;
    double d;

    FOR_EACH_PAR(par)
    {
        par.c(c, iC);
        par.quaternion(q, iR);
        par.t(t, iT);
        par.d(d, iD);
        fprintf(file,
                "%03d %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
                c,
                q(0), q(1), q(2), q(3),
                t(0), t(1),
                d,
                par.wC(iC) * par.wR(iR) * par.wT(iT) * par.wD(iD));
    }

    fclose(file);
}

void save(const char filename[],
          const Particle& par,
          const ParticleType pt)
{
    FILE* file = fopen(filename, "w");

    if (pt == PAR_C)
    {
        unsigned int c;

        FOR_EACH_C(par)
        {
            par.c(c, iC);

            fprintf(file,
                    "%03d %.24lf\n",
                    c,
                    par.wC(iC));
        }
    }
    else if (pt == PAR_R)
    {
        vec4 q;
 
        FOR_EACH_R(par)
        {
            par.quaternion(q, iR);

            fprintf(file,
                    "%15.9lf %15.9lf %15.9lf %15.9lf %.24lf\n",
                    q(0), q(1), q(2), q(3),
                    par.wR(iR));
        }
    }
    else if (pt == PAR_T)
    {
        vec2 t;

        FOR_EACH_T(par)
        {
            par.t(t, iT);

            fprintf(file,
                    "%15.9lf %15.9lf %.24lf\n",
                    t(0), t(1),
                    par.wT(iT));
        }
    }
    else if (pt == PAR_D)
    {
        double d;

        FOR_EACH_D(par)
        {
            par.d(d, iD);

            fprintf(file,
                   "%15.9lf %.24lf\n",
                   d,
                   par.wD(iD));
        }
    }

    fclose(file);
}
/***
void load(Particle& par,
          const char filename[])
{
    FILE* file = fopen(filename, "r");

    int c;
    vec4 q;
    vec2 t;
    double d;
    double w;

    char buf[FILE_LINE_LENGTH];

    int nLine = 0;
    while (fgets(buf, FILE_LINE_LENGTH, file)) nLine++;

    par.reset(1, nLine);

    rewind(file);

    for (int i = 0; i < nLine; i++) 
    {
        fscanf(file,
               "%d %lf %lf %lf %lf %lf %lf %lf %lf",
               &c,
               &q(0), &q(1), &q(2), &q(3),
               &t(0), &t(1),
               &d,
               &w);

        par.setC(c, i);
        par.setQuaternion(q, i);
        par.setT(t, i);
        par.setD(d, i);
        par.setW(w, i);
    }

    fclose(file);
}
***/

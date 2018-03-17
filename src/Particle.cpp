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

    defaultInit();

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

    _uC.resize(_nC);
    _uR.resize(_nR);
    _uT.resize(_nT);
    _uD.resize(_nD);

    reset();
}

void Particle::reset()
{
    gsl_rng* engine = get_random_engine();

    // initialise class distribution

    for (int i = 0; i < _nC; i++)
        _c(i) = i;

    // initialise rotation distribution

    switch (_mode)
    {
        // rotation, MODE_2D, sample from von Mises Distribution with k = 1
        case MODE_2D:

            sampleVMS(_r, dvec4(1, 0, 0, 0), 1, _nR);

            break;

        // rotation, MODE_3D, sample from Angular Central Gaussian Distribution
        // with identity matrix
        case MODE_3D:
            
            sampleACG(_r, 1, 1, 1, _nR);

            break;

        default:

            REPORT_ERROR("INEXISTENT MODE");

            abort();

            break;
    }


    // initialise translation distribution

#ifdef PARTICLE_TRANS_INIT_GAUSSIAN
    // sample from 2D Gaussian Distribution
    for (int i = 0; i < _nT; i++)
        gsl_ran_bivariate_gaussian(engine, _transS, _transS, 0, &_t(i, 0), &_t(i, 1));
#endif

#ifdef PARTICLE_TRANS_INIT_FLAT
    // sample for 2D Flat Distribution in a Square
    for (int i = 0; i < _nT; i++)
    {
        _t(i, 0) = gsl_ran_flat(engine,
                                -gsl_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 2) * _transS,
                                gsl_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 2) * _transS);
        _t(i, 1) = gsl_ran_flat(engine,
                                -gsl_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 2) * _transS,
                                gsl_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 2) * _transS);
    }
#endif

    // initialise defocus distribution

    _d = dvec::Constant(_nD, 1);

    // initialise weight

    _wC = dvec::Constant(_nC, 1.0 / _nC);
    _wR = dvec::Constant(_nR, 1.0 / _nR);
    _wT = dvec::Constant(_nT, 1.0 / _nT);
    _wD = dvec::Constant(_nD, 1.0 / _nD);

    _uC = dvec::Constant(_nC, 1.0 / _nC);
    _uR = dvec::Constant(_nR, 1.0 / _nR);
    _uT = dvec::Constant(_nT, 1.0 / _nT);
    _uD = dvec::Constant(_nD, 1.0 / _nD);

#ifdef PARTICLE_TRANS_INIT_GAUSSIAN
#ifdef PARTILCE_BALANCE_WEIGHT
    balanceWeight(PAR_T);
#endif
#endif

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

    dmat4 r(nR, 4);

    switch (_mode)
    {
        case MODE_2D:
            // sample from von Mises Distribution with kappa = 0
            sampleVMS(r, dvec4(1, 0, 0, 0), 0, nR);
            break;

        case MODE_3D:
            // sample from Angular Central Gaussian Distribution with identity matrix
            sampleACG(r, 1, 1, nR);
            break;

        default:
            REPORT_ERROR("INEXISTENT MODE");
            break;
    }

    dmat2 t(nT, 2);

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

#ifdef PARTICLE_DEFOCUS_INIT_GAUSSIAN
    for (int i = 0; i < _nD; i++)
        _d(i) = 1 + gsl_ran_gaussian(engine, sD);
#endif

#ifdef PARTICLE_DEFOCUS_INIT_FLAT
    for (int i = 0; i < _nD; i++)
        _d(i) = 1 + gsl_ran_flat(engine,
                                 -gsl_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 1) * sD,
                                 gsl_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 1) * sD);
#endif

    _wD = dvec::Constant(_nD, 1.0 / _nD);

    _uD = dvec::Constant(_nD, 1.0 / _nD);

#ifdef PARTICLE_DEFOCUS_INIT_GAUSSIAN
#ifdef PARTILCE_BALANCE_WEIGHT
    balanceWeight(PAR_D);
#endif
#endif
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

dmat4 Particle::r() const { return _r; }

void Particle::setR(const dmat4& r) { _r = r; }

dmat2 Particle::t() const { return _t; }

void Particle::setT(const dmat2& t) { _t = t; }

dvec Particle::d() const { return _d; }

void Particle::setD(const dvec& d) { _d = d; }

dvec Particle::wC() const { return _wC; }

void Particle::setWC(const dvec& wC) { _wC = wC; }

dvec Particle::wR() const { return _wR; }

void Particle::setWR(const dvec& wR) { _wR = wR; }

dvec Particle::wT() const { return _wT; }

void Particle::setWT(const dvec& wT) { _wT = wT; }

dvec Particle::wD() const { return _wD; }

void Particle::setWD(const dvec& wD) { _wD = wD; }

dvec Particle::uC() const { return _uC; }

void Particle::setUC(const dvec& uC) { _uC = uC; }

dvec Particle::uR() const { return _uR; }

void Particle::setUR(const dvec& uR) { _uR = uR; }

dvec Particle::uT() const { return _uT; }

void Particle::setUT(const dvec& uT) { _uT = uT; }

dvec Particle::uD() const { return _uD; }

void Particle::setUD(const dvec& uD) { _uD = uD; }

const Symmetry* Particle::symmetry() const { return _sym; }

void Particle::setSymmetry(const Symmetry* sym) { _sym = sym; }

void Particle::load(const int nR,
                    const int nT,
                    const int nD,
                    const dvec4& q,
                    const double k1,
                    const double k2,
                    const double k3,
                    const dvec2& t,
                    const double s0,
                    const double s1,
                    const double d,
                    const double s)
{
    _nC = 1;
    _nR = nR;
    _nT = nT;
    _nD = nD;

    _c.resize(1);
    _wC.resize(1);
    _uC.resize(1);

    _c(0) = 0;
    _wC(0) = 1;
    _uC(0) = 1;

    _topCPrev = 0;
    _topC = 0;

    _r.resize(_nR, 4);
    _t.resize(_nT, 2);
    _d.resize(_nD);

    _wR.resize(_nR);
    _wT.resize(_nT);
    _wD.resize(_nD);

    _uR.resize(_nR);
    _uT.resize(_nT);
    _uD.resize(_nD);

    gsl_rng* engine = get_random_engine();

    // load the rotation

    _k1 = k1;
    _k2 = k2;
    _k3 = k3;
    
    // _k0 = 1;
    
    // _k1 = gsl_pow_2(stdR);
    
    _topRPrev = q;
    _topR = q;

    // dmat4 p(_nR, 4);

    // sampleACG(_r, _k0, _k1, _nR);

    sampleACG(_r, _k1, _k2, _k3, _nR);

    //sampleACG(p, 1, gsl_pow_2(stdR), _nR);

    if (_mode == MODE_3D) symmetrise();
    
    for (int i = 0; i < _nR; i++)
    {
        dvec4 pert = _r.row(i).transpose();

        dvec4 part;

        /***
        if (gsl_ran_flat(engine, -1, 1) >= 0)
            quaternion_mul(part, quat, pert);
        else
            quaternion_mul(part, -quat, pert);
        ***/

        if (gsl_ran_flat(engine, -1, 1) >= 0)
            quaternion_mul(part, pert, q);
        else
            quaternion_mul(part, pert, -q);

        _r.row(i) = part.transpose();

        _wR(i) = 1.0 / _nR;

        _uR(i) = 1.0 / _nR;
    }

    // load the translation

    _s0 = s0;
    _s1 = s1;

    _topTPrev = t;
    _topT = t;

    for (int i = 0; i < _nT; i++)
    {
       gsl_ran_bivariate_gaussian(engine,
                                  _s0,
                                  _s1,
                                  0,
                                  &_t(i, 0),
                                  &_t(i, 1));

       _t(i, 0) += t(0);
       _t(i, 1) += t(1);

       _wT(i) = 1.0 / _nT;

       _uT(i) = 1.0 / _nT;
    }

    // load the defocus factor

    _s = s;
    
    _topDPrev = d;
    _topD = d;

    for (int i = 0; i < _nD; i++)
    {
        _d(i) = d + gsl_ran_gaussian(engine, _s);

        _wD(i) = 1.0 / _nD;
        _uD(i) = 1.0 / _nD;
    }

}

void Particle::vari(double& k1,
                    double& k2,
                    double& k3,
                    double& s0,
                    double& s1,
                    double& s) const
{
    k1 = _k1;
    k2 = _k2;
    k3 = _k3;
    s0 = _s0;
    s1 = _s1;
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

            rVari = _k1;

            break;

        case MODE_3D:
            /***
            if (_k0 == 0) CLOG(FATAL, "LOGGER_SYS") << "k0 = 0";
            if (gsl_isnan(_k0)) CLOG(FATAL, "LOGGER_SYS") << "k0 NAN";
            if (gsl_isnan(_k1)) CLOG(FATAL, "LOGGER_SYS") << "k1 NAN";
            ***/
            // more cencentrate, smaller rVari, bigger _k0 / _k1;

            // rVari = sqrt(_k1) / sqrt(_k0);
            rVari = pow(_k1 * _k2 * _k3, 1.0 / 6);

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

double Particle::variR() const
{
    if (_mode == MODE_2D)
    {
        return _k1;
    }
    else if (_mode == MODE_3D)
    {
        return pow(_k1 * _k2 * _k3, 1.0 / 6);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

double Particle::variT() const
{
    mat22 A;

    A << gsl_pow_2(_s0), _rho,
         _rho, gsl_pow_2(_s1);

    SelfAdjointEigenSolver<mat22> eigenSolver(A);

    return sqrt(eigenSolver.eigenvalues()[0]
              * eigenSolver.eigenvalues()[1]);
}

double Particle::variD() const
{
    return _s;
}

double Particle::compressR() const
{
    // return _transS / sqrt(_s0 * _s1);

    // return pow(_k0 / _k1, 1.5);

    // return sqrt(_k0) / sqrt(_k1);

    if (_mode == MODE_2D)
    {
        return 1.0 / _k1;
    }
    else if (_mode == MODE_3D)
    {
        return pow(_k1 * _k2 * _k3, -1.0 / 6);
        //return pow(_k1 * _k2 * _k3, -1.0 / 3);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    // return pow(_k1 * _k2 * _k3, -1.0 / 3);

    // return _k0 / _k1;

    // return pow(_k0 / _k1, 1.5) * gsl_pow_2(_transS) / _s0 / _s1;

    //return gsl_pow_2(_transS) / _s0 / _s1;
}

double Particle::compressT() const
{
    mat22 A;

    A << gsl_pow_2(_s0), _rho,
         _rho, gsl_pow_2(_s1);

    SelfAdjointEigenSolver<mat22> eigenSolver(A);

    return 1.0 / sqrt(eigenSolver.eigenvalues()[0]
                    * eigenSolver.eigenvalues()[1]);
}

double Particle::compressD() const
{
    return 1.0 / _s;
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

double Particle::uC(const int i) const
{
    return _uC(i);
}

void Particle::setUC(const double uC,
                     const int i)
{
    _uC(i) = uC;
}

double Particle::uR(const int i) const
{
    return _uR(i);
}

void Particle::setUR(const double uR,
                     const int i)
{
    _uR(i) = uR;
}

double Particle::uT(const int i) const
{
    return _uT(i);
}

void Particle::setUT(const double uT,
                     const int i)
{
    _uT(i) = uT;
}

double Particle::uD(const int i) const
{
    return _uD(i);
}

void Particle::setUD(const double uD,
                     const int i)
{
    _uD(i) = uD;
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
    dvec4 quat = _r.row(i).transpose();
    angle(dst.phi,
          dst.theta,
          dst.psi,
          quat);

    dst.x = _t(i, 0);
    dst.y = _t(i, 1);
}
***/

void Particle::c(size_t& dst,
                 const int i) const
{
    dst = _c(i);
}

void Particle::setC(const size_t src,
                    const int i)
{
    _c(i) = src;
}

void Particle::rot(dmat22& dst,
                   const int i) const
{
    rotate2D(dst, dvec2(_r(i, 0), _r(i, 1)));
}

void Particle::rot(dmat33& dst,
                   const int i) const
{
    rotate3D(dst, _r.row(i).transpose());
}

void Particle::t(dvec2& dst,
                 const int i) const
{
    dst = _t.row(i).transpose();
}

void Particle::setT(const dvec2& src,
                    const int i)
{
    _t.row(i) = src.transpose();
}

void Particle::quaternion(dvec4& dst,
                          const int i) const
{
    dst = _r.row(i).transpose();
}

void Particle::setQuaternion(const dvec4& src,
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

/***
double Particle::k0() const
{
    return _k0;
}

void Particle::setK0(const double k0)
{
    _k0 = k0;
}
***/

double Particle::k1() const
{
    return _k1;
}

void Particle::setK1(const double k1)
{
    _k1 = k1;
}

double Particle::k2() const
{
    return _k2;
}

void Particle::setK2(const double k2)
{
    _k2 = k2;
}

double Particle::k3() const
{
    return _k3;
}

void Particle::setK3(const double k3)
{
    _k3 = k3;
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

double Particle::rho() const
{
    return _rho;
}

void Particle::setRho(const double rho)
{
    _rho = rho;
}

double Particle::s() const
{
    return _s;
}

void Particle::setS(const double s)
{
    _s = s;
}

/***
void Particle::calClassDistr()
{
    _cDistr.setZero();

    for (int i = 0; i < _n; i++)
        _cDistr(_c(i)) += 1;
}
***/

void Particle::calRank1st(const ParticleType pt)
{
    uvec rank = iSort(pt);

    if (pt == PAR_C)
        c(_topC, rank(0));
    else if (pt == PAR_R)
        quaternion(_topR, rank(0));
    else if (pt == PAR_T)
        t(_topT, rank(0));
    else if (pt == PAR_D)
        d(_topD, rank(0));
}

void Particle::calVari(const ParticleType pt)
{

    if (pt == PAR_C)
    {
        CLOG(WARNING, "LOGGER") << "NO NEED TO CALCULATE VARIANCE IN CLASS";
    }
    else if (pt == PAR_R)
    {
        if (_mode == MODE_2D)
        {
            inferVMS(_k1, _r);
        }
        else if (_mode == MODE_3D)
        {
            dvec4 mean;

#ifdef PARTICLE_ROT_MEAN_USING_STAT
            inferACG(mean, _r);
#else
            mean = _topR;
#endif

            dvec4 quat;

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, quaternion_conj(mean), quat);

                _r.row(i) = quat.transpose();
            }

            symmetrise(); // TODO

            inferACG(_k1, _k2, _k3, _r);

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, mean, quat);

                _r.row(i) = quat.transpose();
            }
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
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

        _rho = gsl_stats_covariance(_t.col(0).data(),
                                    1,
                                    _t.col(1).data(),
                                    1,
                                    _t.rows());
        // _rho = gsl_stats_(_t.col(0).data(), 1, _t.col(1).data(), 1, _t.rows());

        /***
#ifdef PARTICLE_RHO
        _rho = gsl_stats_correlation(_t.col(0).data(), 1, _t.col(1).data(), 1, _t.rows());
#else
        _rho = 0;
#endif
        ***/
    }
    else if (pt == PAR_D)
    {
#ifdef PARTICLE_CAL_VARI_DEFOCUS_ZERO_MEAN
        _s = gsl_stats_sd_m(_d.data(), 1, _d.size(), 0);
#else
        _s = gsl_stats_sd(_d.data(), 1, _d.size());
#endif
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
        dmat4 d(_nR, 4);

        if (_mode == MODE_2D)
        {
            sampleVMS(d, dvec4(1, 0, 0, 0), GSL_MIN_DBL(1, _k1 * pf), _nR);

            dvec4 quat;

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, quat, d.row(i).transpose());

                _r.row(i) = quat.transpose();
            }
        }
        else if (_mode == MODE_3D)
        {
#ifdef PARTICLE_ROTATION_KAPPA
            double kappa = GSL_MIN_DBL(1, GSL_MAX_DBL(_k1, GSL_MAX_DBL(_k2, _k3)));
            sampleACG(d,
                      gsl_pow_2(pf) * kappa,
                      gsl_pow_2(pf) * kappa,
                      gsl_pow_2(pf) * kappa,
                      _nR);
#else
            sampleACG(d,
                      gsl_pow_2(pf) * GSL_MIN_DBL(1, _k1),
                      gsl_pow_2(pf) * GSL_MIN_DBL(1, _k2),
                      gsl_pow_2(pf) * GSL_MIN_DBL(1, _k3),
                      _nR);
#endif

            dvec4 mean;

#ifdef PARTICLE_ROT_MEAN_USING_STAT
            inferACG(mean, _r);
#else
            mean = _topR;
#endif

            dvec4 quat;

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, quaternion_conj(mean), quat);

                _r.row(i) = quat.transpose();
            }

            dvec4 pert;
           
            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();
                pert = d.row(i).transpose();
                quaternion_mul(quat, pert, quat);
                _r.row(i) = quat.transpose();
            }

            symmetrise();

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, mean, quat);

                _r.row(i) = quat.transpose();
            }
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

#ifdef PARTILCE_BALANCE_WEIGHT
        balanceWeight(PAR_R);
#endif
    }
    else if (pt == PAR_T)
    {
        gsl_rng* engine = get_random_engine();

        for (int i = 0; i < _nT; i++)
        {
            double x, y;

#ifdef PARTICLE_TRANSLATION_S
            double s = GSL_MAX_DBL(_s0, _s1);
            //gsl_ran_bivariate_gaussian(engine, s, s, 0, &x, &y);
            gsl_ran_bivariate_gaussian(engine, s, s, _rho / s / s, &x, &y);
#else
            //gsl_ran_bivariate_gaussian(engine, _s0, _s1, 0, &x, &y);
            gsl_ran_bivariate_gaussian(engine, _s0, _s1, _rho / _s0 / _s1, &x, &y);
#endif

            _t(i, 0) += x * pf;
            _t(i, 1) += y * pf;
        }

#ifdef PARTICLE_RECENTRE

        reCentre();

#endif

#ifdef PARTILCE_BALANCE_WEIGHT
        balanceWeight(PAR_T);
#endif
    }
    else if (pt == PAR_D)
    {
        gsl_rng* engine = get_random_engine();

        for (int i = 0; i < _nD; i++)
            _d(i) += gsl_ran_gaussian(engine, _s) * pf;

#ifdef PARTILCE_BALANCE_WEIGHT
        balanceWeight(PAR_D);
#endif
    }
}

void Particle::resample(const int n,
                        const ParticleType pt)
{
    gsl_rng* engine = get_random_engine();

    if (pt == PAR_C)
    {
        shuffle(pt);

        uvec rank = iSort(pt);

        c(_topC, rank(0));

        for (int i = 0; i < _nC; i++)
            _wC(i) *= _uC(i);

        _wC /= _wC.sum();

        dvec cdf = d_cumsum(_wC);
        
        cdf /= cdf(_nC - 1);

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

#ifdef PARTICLE_PRIOR_ONE
            _wC(j) = 1.0 / _uC(i);
#else
            _wC(j) = 1.0 / _nC;
#endif
        }

        _c = c;

        _uC.resize(_nC);
    }
    else if (pt == PAR_R)
    {
        shuffle(pt);

        uvec rank = iSort(pt);

        quaternion(_topR, rank(0));

        for (int i = 0; i < _nR; i++)
            _wR(i) *= _uR(i);

        _wR /= _wR.sum();

        dvec cdf = d_cumsum(_wR);

        cdf /= cdf(_nR - 1);

        _nR = n;
        _wR.resize(_nR);

        dmat4 r(_nR, 4);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nR);  

        int i = 0;
        for (int j = 0; j < _nR; j++)
        {
            double uj = u0 + j * 1.0 / _nR;

            while (uj > cdf[i])
                i++;
        
            r.row(j) = _r.row(i);

#ifdef PARTICLE_PRIOR_ONE
            _wR(j) = 1.0 / _uR(i);
#else
            _wR(j) = 1.0 / _nR;
#endif
        }

        _r = r;

        _uR.resize(_nR);
    }
    else if (pt == PAR_T)
    {
        shuffle(pt);

        uvec rank = iSort(pt);

        t(_topT, rank(0));

        for (int i = 0; i < _nT; i++)
            _wT(i) *= _uT(i);

        _wT /= _wT.sum();

        dvec cdf = d_cumsum(_wT);

        cdf /= cdf(_nT - 1);

        _nT = n;
        _wT.resize(_nT);

        dmat2 t(_nT, 2);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nT);  

        int i = 0;
        for (int j = 0; j < _nT; j++)
        {
            double uj = u0 + j * 1.0 / _nT;

            while (uj > cdf[i])
                i++;
        
            t.row(j) = _t.row(i);

#ifdef PARTICLE_PRIOR_ONE
            _wT(j) = 1.0 / _uT(i);
#else
            _wT(j) = 1.0 / _nT;
#endif
        }

        _t = t;

        _uT.resize(_nT);
    }
    else if (pt == PAR_D)
    {
        shuffle(pt);

        uvec rank = iSort(pt);

        d(_topD, rank(0));

        for (int i = 0; i < _nD; i++)
            _wD(i) *= _uD(i);

        _wD /= _wD.sum();

        dvec cdf = d_cumsum(_wD);

        cdf /= cdf(_nD - 1);

        _nD = n;
        _wD.resize(_nD);

        dvec d(_nD);

        double u0 = gsl_ran_flat(engine, 0, 1.0 / _nD);  

        int i = 0;
        for (int j = 0; j < _nD; j++)
        {
            double uj = u0 + j * 1.0 / _nD;

            while (uj > cdf[i])
                i++;

            d(j) = _d(i);

#ifdef PARTICLE_PRIOR_ONE
            _wD(j) = 1.0 / _uD(i);
#else
            _wD(j) = 1.0 / _nD;
#endif
        }

        _d = d;

        _uD.resize(_nD);
    }

    normW();
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

    dvec cdf = cumsum(_w);

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
    dmat4 r(n, 4);
    dmat2 t(n, 2);
    dvec d(n);
    
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
            sampleVMS(r, dvec4(1, 0, 0, 0), 0, nG);
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
    dmat4 r(n, 4);
    dmat2 t(n, 2);
    dvec d(n);
    dvec w(n);

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
        dvec wC(n);
        dvec uC(n);

        for (int i = 0; i < n; i++)
        {
            c(i) = _c(order(i));
            wC(i) = _wC(order(i));
            uC(i) = _uC(order(i));
        }

        _nC = n;
        
        _c = c;
        _wC = wC;
        _uC = uC;
    }
    else if (pt == PAR_R)
    {
        if (n > _nR)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        dmat4 r(n, 4);
        dvec wR(n);
        dvec uR(n);

        for (int i = 0; i < n; i++)
        {
            r.row(i) = _r.row(order(i));
            wR(i) = _wR(order(i));
            uR(i) = _uR(order(i));
        }

        _nR = n;
        
        _r = r;
        _wR = wR;
        _uR = uR;
    }
    else if (pt == PAR_T)
    {
        if (n > _nT)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        dmat2 t(n, 2);
        dvec wT(n);
        dvec uT(n);

        for (int i = 0; i < n; i++)
        {
            t.row(i) = _t.row(order(i));
            wT(i) = _wT(order(i));
            uT(i) = _uT(order(i));
        }

        _nT = n;
        
        _t = t;
        _wT = wT;
        _uT = uT;
    }
    else if (pt == PAR_D)
    {
        if (n > _nD)
            REPORT_ERROR("CANNOT SELECT TOP K FROM N WHEN K > N");

        dvec d(n);
        dvec wD(n);
        dvec uD(n);

        for (int i = 0; i < n; i++)
        {
            d(i) = _d(order(i));
            wD(i) = _wD(order(i));
            uD(i) = _uD(order(i));
        }

        _nD = n;
        
        _d = d;
        _wD = wD;
        _uD = uD;
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
        return d_index_sort_descend(_uC);
    else if (pt == PAR_R)
        return d_index_sort_descend(_uR);
    else if (pt == PAR_T)
        return d_index_sort_descend(_uT);
    else if (pt == PAR_D)
        return d_index_sort_descend(_uD);
    else abort();
}

void Particle::setPeakFactor(const ParticleType pt)
{
    uvec order = iSort(pt);

    /***
    if (pt == PAR_C)
        _peakFactorC = GSL_MIN_DBL(0.5, _uC.mean() / _uC(order(0)));
    else if (pt == PAR_R)
        _peakFactorR = GSL_MIN_DBL(0.5, _uR.mean() / _uR(order(0)));
    else if (pt == PAR_T)
        _peakFactorT = GSL_MIN_DBL(0.5, _uT.mean() / _uT(order(0)));
    else if (pt == PAR_D)
        _peakFactorD = GSL_MIN_DBL(0.5, _uD.mean() / _uD(order(0)));
    ***/

    if (pt == PAR_C)
    {
#ifdef PARTICLE_PEAK_FACTOR_C
        _peakFactorC = PEAK_FACTOR_C;
#else
        _peakFactorC = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uC(order(_nC / 2)) / _uC(order(0))));
#endif
    }
    else if (pt == PAR_R)
    {
        _peakFactorR = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uR(order(_nR / 2)) / _uR(order(0))));
        /***
        if (_mode == MODE_2D)
            _peakFactorR = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uR(order(_nR / 2)) / _uR(order(0))));
        else if (_mode == MODE_3D)
            _peakFactorR = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uR(order(_nR / 8)) / _uR(order(0))));
        else
        {
            REPORT_ERROR("INEXISTENT MODE");
            abort();
        }
        ***/
    }
    else if (pt == PAR_T)
    {
#ifdef PARTICLE_TRANS_INIT_GAUSSIAN
        //int t = FLOOR(_nT * (gsl_cdf_gaussian_P(1, 2) - gsl_cdf_gaussian_P(-1, 2)));
        int t = FLOOR(_nT * gsl_cdf_chisq_P(1, 2));
        _peakFactorT = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uT(order(t)) / _uT(order(0))));
#endif

#ifdef PARTICLE_TRANS_INIT_FLAT
        _peakFactorT = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uT(order(_nT / 4)) / _uT(order(0))));
#endif
    }
    else if (pt == PAR_D)
    {
#ifdef PARTICLE_DEFOCUS_INIT_GAUSSIAN
        //int t = FLOOR(_nD * (gsl_cdf_gaussian_P(1, 1) - gsl_cdf_gaussian_P(-1, 1)));
        int t = FLOOR(_nT * gsl_cdf_chisq_P(1, 1));
        _peakFactorD = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uD(order(t)) / _uD(order(0))));
#endif

#ifdef PARTICLE_DEFOCUS_INIT_FLAT
        _peakFactorD = GSL_MAX_DBL(PEAK_FACTOR_MIN, GSL_MIN_DBL(PEAK_FACTOR_MAX, _uD(order(_nD / 2)) / _uD(order(0))));
#endif
    }
}

void Particle::resetPeakFactor()
{
    _peakFactorC = PEAK_FACTOR_MIN;
    _peakFactorR = PEAK_FACTOR_MIN;
    _peakFactorT = PEAK_FACTOR_MIN;
    _peakFactorD = PEAK_FACTOR_MIN;
}

void Particle::keepHalfHeightPeak(const ParticleType pt)
{
    uvec order = iSort(pt);

    if (pt == PAR_C)
    {
        double hh = _uC(order(0)) * _peakFactorC;

        for (int i = 0; i < _nC; i++)
            //if (_uC(i) < hh) _uC(i) = 0;
            if (_uC(i) < hh) _uC(i) = 0; else _uC(i) -= hh;

    }
    else if (pt == PAR_R)
    {
        double hh = _uR(order(0)) * _peakFactorR;

        for (int i = 0; i < _nR; i++)
            //if (_uR(i) < hh) _uR(i) = 0;
            if (_uR(i) < hh) _uR(i) = 0; else _uR(i) -= hh;

    }
    else if (pt == PAR_T)
    {
        double hh = _uT(order(0)) * _peakFactorT;

        for (int i = 0; i < _nT; i++)
            //if (_uR(i) < hh) _uR(i) = 0;
            if (_uT(i) < hh) _uT(i) = 0; else _uT(i) -= hh;
    }
    else if (pt == PAR_D)
    {
        double hh = _uD(order(0)) * _peakFactorD;

        for (int i = 0; i < _nD; i++)
            //if (_uD(i) < hh) _uD(i) = 0;
            if (_uD(i) < hh) _uD(i) = 0; else _uD(i) -= hh;
    }
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

void Particle::rank1st(size_t& cls) const
{
    cls = _topC;
}

void Particle::rank1st(dvec4& quat) const
{
    quat = _topR;
}

void Particle::rank1st(dmat22& rot) const
{
    dvec4 quat;
    rank1st(quat);

    rotate2D(rot, dvec2(quat(0), quat(1)));
}

void Particle::rank1st(dmat33& rot) const
{
    dvec4 quat;
    rank1st(quat);

    rotate3D(rot, quat);
}

void Particle::rank1st(dvec2& tran) const
{
    tran = _topT;
}

void Particle::rank1st(double& df) const
{
    df = _topD;
}

void Particle::rank1st(size_t& cls,
                       dvec4& quat,
                       dvec2& tran,
                       double& df) const
{
    cls = _topC;
    quat = _topR;
    tran = _topT;
    df = _topD;
}

void Particle::rank1st(size_t& cls,
                       dmat22& rot,
                       dvec2& tran,
                       double& df) const
{
    dvec4 quat;
    rank1st(cls, quat, tran, df);

    rotate2D(rot, dvec2(quat(0), quat(1)));
}

void Particle::rank1st(size_t& cls,
                       dmat33& rot,
                       dvec2& tran,
                       double& df) const
{
    dvec4 quat;
    rank1st(cls, quat, tran, df);

    rotate3D(rot, quat);
}

void Particle::rand(size_t& cls) const
{
    gsl_rng* engine = get_random_engine();

    if (_nC == 0) { REPORT_ERROR("_nC SHOULD NOT BE ZERO"); abort(); }

    size_t u = gsl_rng_uniform_int(engine, _nC);

    c(cls, u);
}

void Particle::rand(dvec4& quat) const
{
    gsl_rng* engine = get_random_engine();

    if (_nR == 0) { REPORT_ERROR("_nR SHOULD NOT BE ZERO"); abort(); }

    size_t u = gsl_rng_uniform_int(engine, _nR);

    quaternion(quat, u);
}

void Particle::rand(dmat22& rot) const
{
    dvec4 quat;
    rand(quat);

    rotate2D(rot, dvec2(quat(0), quat(1)));
}

void Particle::rand(dmat33& rot) const
{
    dvec4 quat;
    rand(quat);

    rotate3D(rot, quat);
}

void Particle::rand(dvec2& tran) const
{
    gsl_rng* engine = get_random_engine();

    if (_nT == 0) { REPORT_ERROR("_nT SHOULD NOT BE ZERO"); abort(); }

    size_t u = gsl_rng_uniform_int(engine, _nT);

    t(tran, u);
}

void Particle::rand(double& df) const
{
    gsl_rng* engine = get_random_engine();

    if (_nD == 0) { REPORT_ERROR("_nD SHOULD NOT BE ZERO"); abort(); }

    size_t u = gsl_rng_uniform_int(engine, _nD);

    d(df, u);
}

void Particle::rand(size_t& cls,
                    dvec4& quat,
                    dvec2& tran,
                    double& df) const
{
    rand(cls);
    rand(quat);
    rand(tran);
    rand(df);
}

void Particle::rand(size_t& cls,
                    dmat22& rot,
                    dvec2& tran,
                    double& df) const
{
    dvec4 quat;
    rand(cls, quat, tran, df);

    rotate2D(rot, dvec2(quat(0), quat(1)));
}

void Particle::rand(size_t& cls,
                    dmat33& rot,
                    dvec2& tran,
                    double& df) const
{
    dvec4 quat;
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

        gsl_ran_shuffle(engine, s.data(), _nC, sizeof(size_t));

        uvec c(_nC);
        dvec wC(_nC);
        dvec uC(_nC);

        for (int i = 0; i < _nC; i++)
        {
            c(s(i)) = _c(i);
            wC(s(i)) = _wC(i);
            uC(s(i)) = _uC(i);
        }

        _c = c;
        _wC = wC;
        _uC = uC;
    }
    else if (pt == PAR_R)
    {
        uvec s = uvec(_nR);

        for (int i = 0; i < _nR; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nR, sizeof(size_t));

        dmat4 r(_nR, 4);
        dvec wR(_nR);
        dvec uR(_nR);

        for (int i = 0; i < _nR; i++)
        {
            r.row(s(i)) = _r.row(i);
            wR(s(i)) = _wR(i);
            uR(s(i)) = _uR(i);
        }

        _r = r;
        _wR = wR;
        _uR = uR;
    }
    else if (pt == PAR_T)
    {
        uvec s = uvec(_nT);

        for (int i = 0; i < _nT; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nT, sizeof(size_t));

        dmat2 t(_nT, 2);
        dvec wT(_nT);
        dvec uT(_nT);

        for (int i = 0; i < _nT; i++)
        {
            t.row(s(i)) = _t.row(i);
            wT(s(i)) = _wT(i);
            uT(s(i)) = _uT(i);
        }

        _t = t;
        _wT = wT;
        _uT = uT;
    }
    else if (pt == PAR_D)
    {
        uvec s = uvec(_nD);

        for (int i = 0; i < _nD; i++) s(i) = i;

        gsl_ran_shuffle(engine, s.data(), _nD, sizeof(size_t));

        dvec d(_nD);
        dvec wD(_nD);
        dvec uD(_nD);

        for (int i = 0; i < _nD; i++)
        {
            d(s(i)) = _d(i);
            wD(s(i)) = _wD(i);
            uD(s(i)) = _uD(i);
        }

        _d = d;
        _wD = wD;
        _uD = uD;
    }
}

void Particle::shuffle()
{
    shuffle(PAR_R);
    shuffle(PAR_T);
    shuffle(PAR_D);
}

void Particle::balanceWeight(const ParticleType pt)
{
    if (pt == PAR_C)
    {
        CLOG(FATAL, "LOGGER_SYS") << "PAR_C WEIGHT SHOULD NOT BE BALANCED";
    }
    else if (pt == PAR_R)
    {
        if (_mode == MODE_2D)
        {
            dvec2 mu;
            double k;

            inferVMS(mu, k, _r.leftCols<2>());

            for (int i = 0; i < _nR; i++)
            {
                _wR(i) = 1.0 / pdfVMS(dvec2(_r(i, 0), _r(i, 1)), mu, k);
            }
        }
        else if (_mode == MODE_3D)
        {
            dmat44 A;

            inferACG(A, _r);

            for (int i = 0; i < _nR; i++)
            {
                _wR(i) = 1.0 / pdfACG(_r.row(i).transpose(), A);
            }
        }
    }
    else if (pt == PAR_T)
    {
        double m0, m1, s0, s1, rho;

#ifdef PARTICLE_CAL_VARI_TRANS_ZERO_MEAN
        m0 = 0;
        m1 = 0;
#else
        m0 = gsl_stats_mean(_t.col(0).data(), 1, _t.rows());
        m1 = gsl_stats_mean(_t.col(1).data(), 1, _t.rows());
#endif

        s0 = gsl_stats_sd_m(_t.col(0).data(), 1, _t.rows(), m0);
        s1 = gsl_stats_sd_m(_t.col(1).data(), 1, _t.rows(), m1);

        rho = gsl_stats_covariance(_t.col(0).data(), 1, _t.col(1).data(), 1, _t.rows());
        
        //rho = 0;

        /***
#ifdef PARTICLE_RHO
        rho = gsl_stats_covariance(_t.col(0).data(), 1, _t.col(1).data(), 1, _t.rows());
#else
        rho = 0;
#endif
        ***/

        for (int i = 0; i < _nT; i++)
        {
            _wT(i) = 1.0 / gsl_ran_bivariate_gaussian_pdf(_t(i, 0) - m0,
                                                          _t(i, 1) - m1,
                                                          s0,
                                                          s1,
                                                          rho / s0 / s1);
        }
    }
    else if (pt == PAR_D)
    {
        double m, s;

#ifdef PARTICLE_CAL_VARI_TRANS_ZERO_MEAN
        m = 0;
#else
        m = gsl_stats_mean(_d.data(), 1, _d.size());
#endif

        s = gsl_stats_sd_m(_d.data(), 1, _d.size(), m);

        for (int i = 0; i < _nD; i++)
        {
            _wD(i) = 1.0 / gsl_ran_gaussian_pdf(_d(i) - m, s);
        }
    }

    normW();
}

void Particle::copy(Particle& that) const
{
    that.setMode(_mode);
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
    that.setUC(_uC);
    that.setUR(_uR);
    that.setUT(_uT);
    that.setUD(_uD);
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
    dvec4 mean;

    inferACG(mean, _r);
    ***/

    dvec4 quat;

    for (int i = 0; i < _nR; i++)
    {
        dvec4 quat = _r.row(i).transpose();

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
    size_t c;
    dvec4 q;
    dvec2 t;
    double d;

    FOR_EACH_PAR(par)
    {
        par.c(c, iC);
        par.quaternion(q, iR);
        par.t(t, iT);
        par.d(d, iD);
        printf("%03lu %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
               c,
               q(0), q(1), q(2), q(3),
               t(0), t(1),
               d,
               par.wC(iC) * par.wR(iR) * par.wT(iT) * par.wD(iD));
    }
}

void save(const char filename[],
          const Particle& par,
          const bool saveU)
{
    FILE* file = fopen(filename, "w");

    size_t c;
    dvec4 q;
    dvec2 t;
    double d;

    FOR_EACH_PAR(par)
    {
        par.c(c, iC);
        par.quaternion(q, iR);
        par.t(t, iT);
        par.d(d, iD);
        fprintf(file,
                "%03lu %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
                c,
                q(0), q(1), q(2), q(3),
                t(0), t(1),
                d,
                saveU
              ? par.uC(iC) * par.uR(iR) * par.uT(iT) * par.uD(iD)
              : par.wC(iC) * par.wR(iR) * par.wT(iT) * par.wD(iD));
    }

    fclose(file);
}

void save(const char filename[],
          const Particle& par,
          const ParticleType pt,
          const bool saveU)
{
    FILE* file = fopen(filename, "w");

    if (pt == PAR_C)
    {
        size_t c;

        FOR_EACH_C(par)
        {
            par.c(c, iC);

            fprintf(file,
                    "%03lu %.24lf\n",
                    c,
                    saveU ? par.uC(iC) : par.wC(iC));
        }
    }
    else if (pt == PAR_R)
    {
        dvec4 q;
 
        FOR_EACH_R(par)
        {
            par.quaternion(q, iR);

            fprintf(file,
                    "%15.9lf %15.9lf %15.9lf %15.9lf %.24lf\n",
                    q(0), q(1), q(2), q(3),
                    saveU ? par.uR(iR) : par.wR(iR));
        }
    }
    else if (pt == PAR_T)
    {
        dvec2 t;

        FOR_EACH_T(par)
        {
            par.t(t, iT);

            fprintf(file,
                    "%15.9lf %15.9lf %.24lf\n",
                    t(0), t(1),
                    saveU ? par.uT(iT) : par.wT(iT));
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
                   saveU ? par.uD(iD) : par.wD(iD));
        }
    }

    fclose(file);
}

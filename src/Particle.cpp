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
                   const RFLOAT transS,
                   const RFLOAT transQ,
                   const Symmetry* sym)
{
    init(mode, nC, nR, nT, nD, transS, transQ, sym);
}

Particle::~Particle()
{
    clear();
}

void Particle::init(const int mode,
                    const RFLOAT transS,
                    const RFLOAT transQ,
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
                    const RFLOAT transS,
                    const RFLOAT transQ,
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

            sampleVMS(_r, vec4(1, 0, 0, 0), 1, _nR);

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
        TSGSL_ran_bivariate_gaussian(engine, _transS, _transS, 0, &_t(i, 0), &_t(i, 1));
#endif

#ifdef PARTICLE_TRANS_INIT_FLAT
    // sample for 2D Flat Distribution in a Square
    for (int i = 0; i < _nT; i++)
    {
        _t(i, 0) = TSGSL_ran_flat(engine,
                                -TSGSL_cdf_chisq_Qinv(0.5, 2) * _transS,
                                TSGSL_cdf_chisq_Qinv(0.5, 2) * _transS);
        _t(i, 1) = TSGSL_ran_flat(engine,
                                -TSGSL_cdf_chisq_Qinv(0.5, 2) * _transS,
                                TSGSL_cdf_chisq_Qinv(0.5, 2) * _transS);
    }
#endif

    // initialise defocus distribution

    _d = vec::Constant(_nD, 1);

    // initialise weight

    _wC = vec::Constant(_nC, 1.0 / _nC);
    _wR = vec::Constant(_nR, 1.0 / _nR);
    _wT = vec::Constant(_nT, 1.0 / _nT);
    _wD = vec::Constant(_nD, 1.0 / _nD);

    _uC = vec::Constant(_nC, 1.0 / _nC);
    _uR = vec::Constant(_nR, 1.0 / _nR);
    _uT = vec::Constant(_nT, 1.0 / _nT);
    _uD = vec::Constant(_nD, 1.0 / _nD);

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
        RFLOAT r = TSGSL_ran_flat(engine, 0, _transS);
        RFLOAT t = TSGSL_ran_flat(engine, 0, 2 * M_PI);

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
                     const RFLOAT sD)
{
    gsl_rng* engine = get_random_engine();

    _nD = nD;

    _d.resize(nD);

    for (int i = 0; i < _nD; i++)
        _d(i) = 1 + TSGSL_ran_gaussian(engine, sD);

    _wD = vec::Constant(_nD, 1.0 / _nD);
    
    _uD = vec::Constant(_nD, 1.0 / _nD);
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

RFLOAT Particle::transS() const { return _transS; }

void Particle::setTransS(const RFLOAT transS) { _transS = transS; }

RFLOAT Particle::transQ() const { return _transQ; }

void Particle::setTransQ(const RFLOAT transQ) { _transQ = transQ; }

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

vec Particle::uC() const { return _uC; }

void Particle::setUC(const vec& uC) { _uC = uC; }

vec Particle::uR() const { return _uR; }

void Particle::setUR(const vec& uR) { _uR = uR; }

vec Particle::uT() const { return _uT; }

void Particle::setUT(const vec& uT) { _uT = uT; }

vec Particle::uD() const { return _uD; }

void Particle::setUD(const vec& uD) { _uD = uD; }

const Symmetry* Particle::symmetry() const { return _sym; }

void Particle::setSymmetry(const Symmetry* sym) { _sym = sym; }

void Particle::load(const int nR,
                    const int nT,
                    const int nD,
                    const vec4& q,
                    const RFLOAT k1,
                    const RFLOAT k2,
                    const RFLOAT k3,
                    const vec2& t,
                    const RFLOAT s0,
                    const RFLOAT s1,
                    const RFLOAT d,
                    const RFLOAT s)
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
    
    // _k1 = TSGSL_pow_2(stdR);
    
    _topRPrev = q;
    _topR = q;

    // mat4 p(_nR, 4);

    // sampleACG(_r, _k0, _k1, _nR);

    sampleACG(_r, _k1, _k2, _k3, _nR);

    //sampleACG(p, 1, TSGSL_pow_2(stdR), _nR);

    if (_mode == MODE_3D) symmetrise();
    
    for (int i = 0; i < _nR; i++)
    {
        vec4 pert = _r.row(i).transpose();

        vec4 part;

        /***
        if (TSGSL_ran_flat(engine, -1, 1) >= 0)
            quaternion_mul(part, quat, pert);
        else
            quaternion_mul(part, -quat, pert);
        ***/

        if (TSGSL_ran_flat(engine, -1, 1) >= 0)
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
       TSGSL_ran_bivariate_gaussian(engine,
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
        _d(i) = d + TSGSL_ran_gaussian(engine, _s);

        _wD(i) = 1.0 / _nD;
        _uD(i) = 1.0 / _nD;
    }

}

void Particle::vari(RFLOAT& k1,
                    RFLOAT& k2,
                    RFLOAT& k3,
                    RFLOAT& s0,
                    RFLOAT& s1,
                    RFLOAT& s) const
{
    k1 = _k1;
    k2 = _k2;
    k3 = _k3;
    s0 = _s0;
    s1 = _s1;
    s = _s;
}

void Particle::vari(RFLOAT& rVari,
                    RFLOAT& s0,
                    RFLOAT& s1,
                    RFLOAT& s) const
{
    switch (_mode)
    {
        case MODE_2D:

            rVari = _k1;

            break;

        case MODE_3D:
            /***
            if (_k0 == 0) CLOG(FATAL, "LOGGER_SYS") << "k0 = 0";
            if (TSGSL_isnan(_k0)) CLOG(FATAL, "LOGGER_SYS") << "k0 NAN";
            if (TSGSL_isnan(_k1)) CLOG(FATAL, "LOGGER_SYS") << "k1 NAN";
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

RFLOAT Particle::compress() const
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
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    // return pow(_k1 * _k2 * _k3, -1.0 / 3);

    // return _k0 / _k1;

    // return pow(_k0 / _k1, 1.5) * TSGSL_pow_2(_transS) / _s0 / _s1;

    //return TSGSL_pow_2(_transS) / _s0 / _s1;
}

RFLOAT Particle::wC(const int i) const
{
    return _wC(i);
}

void Particle::setWC(const RFLOAT wC,
                     const int i)
{
    _wC(i) = wC;
}

void Particle::mulWC(const RFLOAT wC,
                     const int i)
{
    _wC(i) *= wC;
}

RFLOAT Particle::wR(const int i) const
{
    return _wR(i);
}

void Particle::setWR(const RFLOAT wR,
                     const int i)
{
    _wR(i) = wR;
}

void Particle::mulWR(const RFLOAT wR,
                     const int i)
{
    _wR(i) *= wR;
}

RFLOAT Particle::wT(const int i) const
{
    return _wT(i);
}

void Particle::setWT(const RFLOAT wT,
                     const int i)
{
    _wT(i) = wT;
}

void Particle::mulWT(const RFLOAT wT,
                     const int i)
{
    _wT(i) *= wT;
}

RFLOAT Particle::wD(const int i) const
{
    return _wD(i);
}

void Particle::setWD(const RFLOAT wD,
                     const int i)
{
    _wD(i) = wD;
}

void Particle::mulWD(const RFLOAT wD,
                     const int i)
{
    _wD(i) *= wD;
}

void Particle::setUC(const RFLOAT uC,
                     const int i)
{
    _uC(i) = uC;
}

void Particle::setUR(const RFLOAT uR,
                     const int i)
{
    _uR(i) = uR;
}

void Particle::setUT(const RFLOAT uT,
                     const int i)
{
    _uT(i) = uT;
}

void Particle::setUD(const RFLOAT uD,
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

void Particle::d(RFLOAT& d,
                 const int i) const
{
    d = _d(i);
}

void Particle::setD(const RFLOAT d,
                    const int i)
{
    _d(i) = d;
}

/***
RFLOAT Particle::k0() const
{
    return _k0;
}

void Particle::setK0(const RFLOAT k0)
{
    _k0 = k0;
}
***/

RFLOAT Particle::k1() const
{
    return _k1;
}

void Particle::setK1(const RFLOAT k1)
{
    _k1 = k1;
}

RFLOAT Particle::k2() const
{
    return _k2;
}

void Particle::setK2(const RFLOAT k2)
{
    _k2 = k2;
}

RFLOAT Particle::k3() const
{
    return _k3;
}

void Particle::setK3(const RFLOAT k3)
{
    _k3 = k3;
}

RFLOAT Particle::s0() const
{
    return _s0;
}

void Particle::setS0(const RFLOAT s0)
{
    _s0 = s0;
}

RFLOAT Particle::s1() const
{
    return _s1;
}

void Particle::setS1(const RFLOAT s1)
{
    _s1 = s1;
}

RFLOAT Particle::s() const
{
    return _s;
}

void Particle::setS(const RFLOAT s)
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

            // _k1 = 1.0 / (1 + _k1); // converting range, extreme sparse, _k1 = 1, extreme dense, _k1 = 0
        }
        else if (_mode == MODE_3D)
        {
            vec4 mean;

            inferACG(mean, _r);

            vec4 quat;

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, quaternion_conj(mean), quat);

                _r.row(i) = quat.transpose();
            }

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
        _s0 = TSGSL_stats_sd_m(_t.col(0).data(), 1, _t.rows(), 0);
        _s1 = TSGSL_stats_sd_m(_t.col(1).data(), 1, _t.rows(), 0);
#else
        _s0 = TSGSL_stats_sd(_t.col(0).data(), 1, _t.rows());
        _s1 = TSGSL_stats_sd(_t.col(1).data(), 1, _t.rows());
#endif

        _rho = 0;
    }
    else if (pt == PAR_D)
    {
        _s = TSGSL_stats_sd(_d.data(), 1, _d.size());
    }
}

void Particle::perturb(const RFLOAT pf,
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
        {
            sampleVMS(d, vec4(1, 0, 0, 0), TSGSL_MIN_RFLOAT(1, _k1 * pf), _nR);

            vec4 quat;

            for (int i = 0; i < _nR; i++)
            {
                quat = _r.row(i).transpose();

                quaternion_mul(quat, quat, d.row(i).transpose());

                _r.row(i) = quat.transpose();
            }
        }
        else if (_mode == MODE_3D)
        {
            sampleACG(d,
                      TSGSL_pow_2(pf) * TSGSL_MIN_RFLOAT(1, _k1),
                      TSGSL_pow_2(pf) * TSGSL_MIN_RFLOAT(1, _k2),
                      TSGSL_pow_2(pf) * TSGSL_MIN_RFLOAT(1, _k3),
                      _nR);

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
    }
    else if (pt == PAR_T)
    {
        gsl_rng* engine = get_random_engine();

        for (int i = 0; i < _nT; i++)
        {
            RFLOAT x, y;

            TSGSL_ran_bivariate_gaussian(engine, _s0, _s1, _rho, &x, &y);

            _t(i, 0) += x * pf;
            _t(i, 1) += y * pf;
        }

#ifdef PARTICLE_RECENTRE

        reCentre();

#endif
    }
    else if (pt == PAR_D)
    {
        gsl_rng* engine = get_random_engine();

        for (int i = 0; i < _nD; i++)
            _d(i) += TSGSL_ran_gaussian(engine, _s) * pf;
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

#ifndef NAN_NO_CHECK
        if ((_wC.sum() == 0) || (TSGSL_isnan(_wC.sum())))
        {
            REPORT_ERROR("NAN DETECTED");

            abort();
        }
#endif

        _wC /= _wC.sum();

        vec cdf = cumsum(_wC);
        
        _nC = n;
        _wC.resize(_nC);

        uvec c(_nC);

        RFLOAT u0 = TSGSL_ran_flat(engine, 0, 1.0 / _nC);  

        int i = 0;
        for (int j = 0; j < _nC; j++)
        {
            RFLOAT uj = u0 + j * 1.0 / _nC;

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

#ifndef NAN_NO_CHECK
        if ((_wR.sum() == 0) || (TSGSL_isnan(_wR.sum())))
        {
            REPORT_ERROR("NAN DETECTED");

            abort();
        }
#endif

        _wR /= _wR.sum();

        /***
        if (_wR.sum() != 1)
        {
            CLOG(WARNING, "LOGGER_SYS") << _wR.sum();

            REPORT_ERROR("WRONG!");

            abort();
        }
        ***/

        vec cdf = cumsum(_wR);

        int nOld = _nR;

        _nR = n;
        _wR.resize(_nR);

        mat4 r(_nR, 4);

        RFLOAT u0 = TSGSL_ran_flat(engine, 0, 1.0 / _nR);  

        int i = 0;
        for (int j = 0; j < _nR; j++)
        {
            RFLOAT uj = u0 + j * 1.0 / _nR;

            while (uj > cdf[i])
            {
                i++;

                if (i >= nOld)
                {
                    CLOG(WARNING, "LOGGER_SYS") << "cdf = " << cdf[i - 1];
                    CLOG(WARNING, "LOGGER_SYS") << "uj = " << uj;

                    abort();
                }
            }
        
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

#ifndef NAN_NO_CHECK
        if ((_wT.sum() == 0) || (TSGSL_isnan(_wT.sum())))
        {
            REPORT_ERROR("NAN DETECTED");

            abort();
        }
#endif
        vec cdf = cumsum(_wT);

        _nT = n;
        _wT.resize(_nT);

        mat2 t(_nT, 2);

        RFLOAT u0 = TSGSL_ran_flat(engine, 0, 1.0 / _nT);  

        int i = 0;
        for (int j = 0; j < _nT; j++)
        {
            RFLOAT uj = u0 + j * 1.0 / _nT;

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

#ifndef NAN_NO_CHECK
        if ((_wD.sum() == 0) || (TSGSL_isnan(_wD.sum())))
        {
            REPORT_ERROR("NAN DETECTED");

            abort();
        }
#endif

        _wD /= _wD.sum();

        vec cdf = cumsum(_wD);

        _nD = n;
        _wD.resize(_nD);

        vec d(_nD);

        RFLOAT u0 = TSGSL_ran_flat(engine, 0, 1.0 / _nD);  

        int i = 0;
        for (int j = 0; j < _nD; j++)
        {
            RFLOAT uj = u0 + j * 1.0 / _nD;

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
void Particle::resample(const RFLOAT alpha)
{
    resample(_n, alpha);
}

void Particle::resample(const int n,
                        const RFLOAT alpha)
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
        c(i) = TSGSL_rng_uniform_int(engine, _m);

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

    RFLOAT u0 = TSGSL_ran_flat(engine, 0, 1.0 / nL);  

    int i = 0;
    for (int j = 0; j < nL; j++)
    {
        RFLOAT uj = u0 + j * 1.0 / nL;

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
RFLOAT Particle::neff() const
{
    return 1.0 / _w.squaredNorm();
}

void Particle::segment(const RFLOAT thres)
{
    uvec order = iSort();

    RFLOAT s = 0;
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

void Particle::flatten(const RFLOAT thres)
{
    uvec order = iSort();

    RFLOAT s = 0;
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
        vec uC(n);

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

        mat4 r(n, 4);
        vec wR(n);
        vec uR(n);

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

        mat2 t(n, 2);
        vec wT(n);
        vec uT(n);

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

        vec d(n);
        vec wD(n);
        vec uD(n);

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
        return index_sort_descend(_uC);
    else if (pt == PAR_R)
        return index_sort_descend(_uR);
    else if (pt == PAR_T)
        return index_sort_descend(_uT);
    else if (pt == PAR_D)
        return index_sort_descend(_uD);
    else abort();
}

bool Particle::diffTopC()
{
    bool diff = (_topCPrev == _topC);

    _topCPrev = _topC;

    return diff;
}

RFLOAT Particle::diffTopR()
{
    RFLOAT diff = 1 - fabs(_topRPrev.dot(_topR));

    _topRPrev = _topR;

    return diff;
}

RFLOAT Particle::diffTopT()
{
    RFLOAT diff = (_topTPrev - _topT).norm();

    _topTPrev = _topT;

    return diff;
}

RFLOAT Particle::diffTopD()
{
    RFLOAT diff = fabs(_topDPrev - _topD);

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

void Particle::rank1st(RFLOAT& df) const
{
    df = _topD;
}

void Particle::rank1st(unsigned int& cls,
                       vec4& quat,
                       vec2& tran,
                       RFLOAT& df) const
{
    cls = _topC;
    quat = _topR;
    tran = _topT;
    df = _topD;
}

void Particle::rank1st(unsigned int& cls,
                       mat22& rot,
                       vec2& tran,
                       RFLOAT& df) const
{
    vec4 quat;
    rank1st(cls, quat, tran, df);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rank1st(unsigned int& cls,
                       mat33& rot,
                       vec2& tran,
                       RFLOAT& df) const
{
    vec4 quat;
    rank1st(cls, quat, tran, df);

    rotate3D(rot, quat);
}

void Particle::rand(unsigned int& cls) const
{
    gsl_rng* engine = get_random_engine();

    if (_nC == 0) { REPORT_ERROR("_nC SHOULD NOT BE ZERO"); abort(); }

    size_t u = TSGSL_rng_uniform_int(engine, _nC);

    c(cls, u);
}

void Particle::rand(vec4& quat) const
{
    gsl_rng* engine = get_random_engine();

    if (_nR == 0) { REPORT_ERROR("_nR SHOULD NOT BE ZERO"); abort(); }

    size_t u = TSGSL_rng_uniform_int(engine, _nR);

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

    if (_nT == 0) { REPORT_ERROR("_nT SHOULD NOT BE ZERO"); abort(); }

    size_t u = TSGSL_rng_uniform_int(engine, _nT);

    t(tran, u);
}

void Particle::rand(RFLOAT& df) const
{
    gsl_rng* engine = get_random_engine();

    if (_nD == 0) { REPORT_ERROR("_nD SHOULD NOT BE ZERO"); abort(); }

    size_t u = TSGSL_rng_uniform_int(engine, _nD);

    d(df, u);
}

void Particle::rand(unsigned int& cls,
                    vec4& quat,
                    vec2& tran,
                    RFLOAT& df) const
{
    rand(cls);
    rand(quat);
    rand(tran);
    rand(df);
}

void Particle::rand(unsigned int& cls,
                    mat22& rot,
                    vec2& tran,
                    RFLOAT& df) const
{
    vec4 quat;
    rand(cls, quat, tran, df);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rand(unsigned int& cls,
                    mat33& rot,
                    vec2& tran,
                    RFLOAT& df) const
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

        TSGSL_ran_shuffle(engine, s.data(), _nC, sizeof(unsigned int));

        uvec c(_nC);
        vec wC(_nC);
        vec uC(_nC);

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

        TSGSL_ran_shuffle(engine, s.data(), _nR, sizeof(unsigned int));

        mat4 r(_nR, 4);
        vec wR(_nR);
        vec uR(_nR);

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

        TSGSL_ran_shuffle(engine, s.data(), _nT, sizeof(unsigned int));

        mat2 t(_nT, 2);
        vec wT(_nT);
        vec uT(_nT);

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

        TSGSL_ran_shuffle(engine, s.data(), _nD, sizeof(unsigned int));

        vec d(_nD);
        vec wD(_nD);
        vec uD(_nD);

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
    }
    else if (pt == PAR_R)
    {
    }
    else if (pt == PAR_T)
    {
    }
    else if (pt == PAR_D)
    {
    }
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
    RFLOAT transM = _transS * TSGSL_cdf_chisq_Qinv(_transQ, 2);

    gsl_rng* engine = get_random_engine();

    for (int i = 0; i < _nT; i++)
        if (NORM(_t(i, 0), _t(i, 1)) > transM)
        {
            // _t.row(i) *= transM / NORM(_t(i, 0), _t(i, 1));

            TSGSL_ran_bivariate_gaussian(engine,
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
    RFLOAT d;

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
    RFLOAT d;

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
        RFLOAT d;

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
    RFLOAT d;
    RFLOAT w;

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

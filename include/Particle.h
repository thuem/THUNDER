/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PARTICLE_H
#define PARTICLE_H

#include <iostream>
#include <numeric>
#include <cmath>

#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Logging.h"
#include "Precision.h"

#include "Coordinate5D.h"
#include "Random.h"
#include "Euler.h"
#include "Functions.h"
#include "Symmetry.h"
#include "DirectionalStat.h"

#define FOR_EACH_C(par) for (int iC = 0; iC < par.nC(); iC++)
#define FOR_EACH_R(par) for (int iR = 0; iR < par.nR(); iR++)
#define FOR_EACH_T(par) for (int iT = 0; iT < par.nT(); iT++)
#define FOR_EACH_D(par) for (int iD = 0; iD < par.nD(); iD++)

#define FOR_EACH_PAR(par) \
    FOR_EACH_C(par) \
        FOR_EACH_R(par) \
            FOR_EACH_T(par) \
                FOR_EACH_D(par)

#define PEAK_FACTOR_MAX 0.5
#define PEAK_FACTOR_MIN 1e-3

#define PEAK_FACTOR_C (1 - 1e-2)

#define PEAK_FACTOR_BASE 2

#define INIT_OUTSIDE_CONFIDENCE_AREA 0.5

#define RHO_MAX (1 - 1e-1)
#define RHO_MIN (-1 + 1e-1)

#define PERTURB_K_MAX 1

enum ParticleType
{
    PAR_C,
    PAR_R,
    PAR_T,
    PAR_D
};

class Particle
{
    private:

        /**
         * @brief the working mode of this particle filter
         * 
         * MODE_2D: the reference is a 2D image, and the perturbation in rotation is in 2D; MODE_3D: the reference is a 3D volume, and the perturbation in rotation is in 2D
         */
        int _mode;

        /**
         * @brief number of support points of the class subspace in this particle filter
         */
        int _nC;

        /**
         * @brief number of support points of the rotation subspace in this particle filter
         */
        int _nR;

        /**
         * @brief number of support points of the translation subspace in this particle filter
         */
        int _nT;

        /**
         * @brief number of support points of the defocus subspace in this particle filter
         */
        int _nD;

        /**
         * @brief the standard deviation of the initial phase of the particle filter in translation, assuming the translation follows a 2D Gaussian distribution
         */
        double _transS;

        /**
         * @brief the re-center threshold of translation
         * 
         * For example, assuming _transQ = 0.01, if a translation lies beyond the confidence area of 99%, this translation will be re-centre to the original point.
         */
        double _transQ;

        /**
         * @brief the ratio of the greatest and the smallest probability of the spport points of the class subspace
         * 
         * It's used to set the probability of the support points with small probability to zero. Set the lower limit as the greatest probabilities multiplies this factor, those greater than lower limit will be substracted by lower limit, and those smaller will be set to 0.
         */
        double _peakFactorC;

        /**
         * @brief the ratio of the greatest and the smallest probability of the spport points of the rotation subspace
         * 
         * It's used to set the probability of the support points with small probability to zero. Set the lower limit as the greatest probabilities multiplies this factor, those greater than lower limit will be substracted by lower limit, and those smaller will be set to 0.
         */
        double _peakFactorR;

        /**
         * @brief the ratio of the greatest and the smallest probability of the spport points of the translation subspace
         * 
         * It's used to set the probability of the support points with small probability to zero. Set the lower limit as the greatest probabilities multiplies this factor, those greater than lower limit will be substracted by lower limit, and those smaller will be set to 0.
         */
        double _peakFactorT;
        
        /**
         * @brief the ratio of the greatest and the smallest probability of the spport points of the defocus subspace
         * 
         * It's used to set the probability of the support points with small probability to zero. Set the lower limit as the greatest probabilities multiplies this factor, those greater than lower limit will be substracted by lower limit, and those smaller will be set to 0.
         */
        double _peakFactorD;

        /**
         * @brief a dvector storing the class of each support point
         */
        uvec _c;

        /**
         * @brief a table storing the rotation information 
         * 
         * MODE_2D: a table storing the rotation information as the first and second elements stand for a unit dvector in circle and the other two elements are zero; MODE_3D: a table storing the rotation information with each row storing a quaternion
         */
        dmat4 _r;

        /**
         * @brief a table storing the translation information with each row storing a 2-dvector with x and y respectively
         */
        dmat2 _t;

        /**
         * @brief a dvector storing the defocus factor of each support point
         */
        dvec _d;

        /**
         * @brief a dvector storing the weight of each support point of the class subspace
         */
        dvec _wC;

        /**
         * @brief a dvector storing the weight of each support point of the rotation subspace
         */
        dvec _wR;

        /**
         * @brief a dvector storing the weight of each support point of the trasnlation subspace
         */
        dvec _wT;

        /**
         * @brief a dvector storing the weight of each support point of the defocus subspace
         */
        dvec _wD;

        /**
         * @brief a dvector storing the likelihood of each support point of the class subspace
         */
        dvec _uC;

        /**
         * @brief a dvector storing the likelihood of each support point of the rotation subspace
         */
        dvec _uR;

        /**
         * @brief a dvector storing the likelihood of each support point of the translation subspace
         */
        dvec _uT;

        /**
         * @brief a dvector storing the likelihood of each support point of the defocus subspace
         */
        dvec _uD;
        
        /**
         * @brief a pointer points to a Symmetry object which indicates the symmetry
         */
        const Symmetry* _sym;

        /**
         * @brief concentration parameter of Angular Central Gaussian distribution of rotation
         * 
         * MODE_2D: Suppose the support points of rotation subspace obeyed the VMS distribution.
         * MODE_3D: Suppose the support points of rotation subspace obeyed the ACG distribution.
         */
        /**
         * @brief concnetration paramter of von Mises distribution of rotation (kappa)
         */
        double _k1;

        double _k2;

        double _k3;

        /**
         * @brief sigma0 of 2D Gaussian distribution of translation
         */
        double _s0;

        /**
         * @brief sigma1 of 2D Gaussian distribution of translation
         */
        double _s1;

        /**
         * @brief rho of 2D Gaussian distribution of translation
         */
        double _rho;

        /**
         * @brief sigma of 1D Gaussian distribution of defocus factor
         */
        double _s;

        /**
         * @brief the score of these support points, the particle filter with higher score will have greater weight in reconstruction process.
         */
        double _score;

        /**
         * @brief the previous most likely class
         */
        size_t _topCPrev;

        /**
         * @brief the most likely class
         */
        size_t _topC;

        /**
         * @brief the previous most likely rotation
         * 
         * MODE_2D: the first element stands for the previous most likely rotation; MODE_3D: quaternion of the previous most likely rotation
         */
        dvec4 _topRPrev;

        /**
         * @brief the most likely rotation
         * 
         * MODE_2D: the first element stands for the most likely rotation; MODE_3D: quaternion of the most likely rotation
         */
        dvec4 _topR;

        /**
         * @brief the previous most likely translation
         */
        dvec2 _topTPrev;

        /**
         * @brief the most likely translation it will be refreshed by resampling
         */
        dvec2 _topT;

        /**
         * @brief the previous most likely defocus factor
         */
        double _topDPrev;

        /**
         * @brief the most likely defocus factor
         */
        double _topD;

        /**
         * @brief default initialiser
         */
        void defaultInit()
        {
            _mode = MODE_3D;

            _nC = 1;

            _sym = NULL;

            _k1 = 1;
            _k2 = 1;
            _k3 = 1;

            _s0 = DBL_MAX;
            _s1 = DBL_MAX;

            _rho = 0;

            _s = 0;

            _topCPrev = 0;
            _topC = 0;

            _topRPrev = dvec4(1, 0, 0, 0);
            _topR = dvec4(1, 0, 0, 0);

            _topTPrev = dvec2(0, 0);
            _topT = dvec2(0, 0);

            _topDPrev = 1;
            _topD = 1;
        }

    public:

        /**
         * @brief default constructor of Particle
         */
        Particle();

        /**
         * @brief constructor of Particle
         */
        Particle(const int mode,             /**< [in] mode of this particle filter, use MODE_2D or MODE_3D to work on 2D mode and 3d mode respectively. */
                 const int nC,               /**< [in] number of support points of the class subspace */
                 const int nR,               /**< [in] number of support points of the rotation subspace */
                 const int nT,               /**< [in] number of support points of the translation subspace */
                 const int nD,               /**< [in] number of support points of the defocus subspace */
                 const double transS,        /**< [in] standard deviation of the initial phase in translation subspace */
                 const double transQ = 0.01, /**< [in] the re-center threshold of translation */
                 const Symmetry* sym = NULL  /**< [in] symmetry of resampling space */
                );

        /**
         * @brief deconstructor of Particle
         */
        ~Particle();

        /**
         * @brief This function initialise Particle.
         */
        void init(const int mode,             /**< [in] mode of this particle filter, use MODE_2D or MODE_3D to work on 2D mode and 3d mode respectively. */
                  const double transS,        /**< [in] standard deviation of the initial phase in translation subspace */
                  const double transQ = 0.01, /**< [in] the re-center threshold of translation */
                  const Symmetry* sym = NULL  /**< [in] symmetry of resampling space */
                );

        /**
         * @brief This function initialises Particle.
         */
        void init(const int mode,             /**< [in] mode of this particle filter, use MODE_2D or MODE_3D to work on 2D mode and 3d mode respectively. */
                  const int nC,               /**< [in] number of support points of the class subspace */
                  const int nR,               /**< [in] number of support points of the rotation subspace */
                  const int nT,               /**< [in] number of support points of the translation subspace */
                  const int nD,               /**< [in] number of support points of the defocus subspace */
                  const double transS,        /**< [in] standard deviation of the initial phase in translation subspace */
                  const double transQ = 0.01, /**< [in] the re-center threshold of translation */
                  const Symmetry* sym = NULL  /**< [in] symmetry of resampling space */
                );

        /**
         * @brief This function resets the support points in this particle filter to a default distribution.
         * 
         * Reset the support points of class subspace and the support points of defocus space to uniform distribution. Reset the support points of translation space to Gaussian distribution. Reset the support points of rotation to a VMS distribution and ACG distribution in MODE_2D and MODE_3D, respectively.
         */
        void reset();

        /**
         * @brief This function resets the support points in this particle filter to a uniform distribution in rotation and 2D Gaussian distribution in translation with a given number of sampling points.
         *
         * @param n number of support points in this particle filter
         */
        /***
        void reset(const int m,
                   const int n
                  );
        ***/

        /**
         * @brief This function resets the support points in this particle filter to a default distribution with given numbers of particls.
         * 
         * Reset the support points of class subspace and the support points of defocus space to uniform distribution with nC, nT and nD support points, respectively.Reset the support points of translation space to Gaussian distribution with nT support points. Reset the support points of rotation to a VMS distribution and ACG distribution with nR support points in MODE_2D and MODE_3D, respectively.
         */
        void reset(const int nC, /**< [in] number of support points of the class subspacess */
                   const int nR, /**< [in] number of support points of the rotation subspace */
                   const int nT, /**< [in] number of support points of the translation subspace */
                   const int nD  /**< [in] number of support points of the defocus subspace */
                  );

        /**
         * @brief initialise defocus factor
         */
        void initD(const int nD,          /**< [in] number of support points of the defocus subspace */
                   const double sD = 0.05 /**< [in] the standard deviation of defocus factor */
                  );

        
        /**
         * @brief This functions returns the mode of this particle fitler.
         * 
         * @return the mode of this particle filter.
         */
        int mode() const;

        /**
         * @brief This functions sets the mode of this particle fitler.
         */
        void setMode(const int mode /**< [in] mode of this particle filter */
                    );

        /**
         * @brief This functions returns the number of support points of the class subspaces in this particle fitler.
         * 
         * @return the number of support points of the class subspaces
         */
        int nC() const;

        /**
         * @brief This functions sets the number of support points of the class subspaces in this particle fitler.
         */
        void setNC(const int nC /**< [in] number of support points of the class subspaces */
                  );

        /**
         * @brief This functions returns the number of support points of the rotation subspace in this particle fitler.
         * 
         * @return the number of support points of the rotation subspace
         */
        int nR() const;

        /**
         * @brief This functions sets the number of support points of the rotation subspace in this particle fitler.
         */
        void setNR(const int nR /**< [in] number of support points of the rotation subspace */
                  );

        /**
         * @brief This functions returns the number of support points about traslation parameters in this particle fitler.
         * 
         * @return the number of support points of the translation subspace
         */
        int nT() const;

        /**
         * @brief This functions sets the number of support points of the translation subspace in this particle fitler.
         */
        void setNT(const int nT /**< [in] number of support points of the translation subspace */
                  );

        /**
         * @brief This functions returns the number of support points of the defocus subspace in this particle fitler.
         * 
         * @return the number of support points of the defocus subspace
         */
        int nD() const;

        /**
         * @brief This functions sets the number of support points of the defocus subspace in this particle fitler.
         */
        void setND(const int nD /**< [in] number of support points of the defocus subspace */
                  );

        /**
         * @brief This function returns the standard deviation of translation, assuming the translation follows a 2D Gaussian distribution.
         * 
         * @return the standard deviation of translation
         */
        double transS() const;

        /**
         * @brief This function sets the standard deviation of translation, assuming the translation follows a 2D Gaussian distribution.
         */
        void setTransS(const double transS /**< [in] transS the standard deviation of translation */
                      );

        /**
         * @brief This function returns the re-center threshold of translation.
         * 
         * @return the re-center threshold of translation
         */
        double transQ() const;

        /**
         * @brief This function sets the re-center theshold of translation.
         */
        void setTransQ(const double transQ /**< [in] the re-center threshold of translation */
                      );

        /**
         * @brief This function returns the array sotring the class information with each element storing an index.
         * 
         * @return the array storing the class information with each element storing an index
         */
        uvec c() const;

        /**
         * @brief This function sets the array sotring the class information with each element storing an index.
         */
        void setC(const uvec& c /**< [in] the array storing the class information with each element storing an index */
                 );

        /**
         * @brief This function returns the table storing the rotation information with each row storing a quaternion.
         * 
         * @return the table storing the rotation information with each row storing a quaternion
         */
        dmat4 r() const;

        /**
         * @brief This function sets the table storing the rotation information with each row storing a quaternion.
         */
        void setR(const dmat4& r /**< [in] the table storing the rotation information with each row storing a quaternion */
                 );

        /**
         * @brief This function returns the table storing the translation information with each row storing a 2-dvector with x and y respectively.
         * 
         * @return the table storing the translation information with each row storing a 2-dvector with x and y respectively
         */
        dmat2 t() const;

        /**
         * @brief This function sets the table storing the translation information with each row storing a 2-dvector with x and y respectively.
         */
        void setT(const dmat2& t /**< [in] the table storing the translation information with each row storing a 2-dvector with x and y respectively */
                 );

        /**
         * @brief This function returns the array storing the translation information with each element storing defocus parameter.
         * 
         * @return the array storing the translation information with each element storing defocus parameter
         */
        dvec d() const;

        /**
         * @brief This function returns the array storing the translation information with each element storing defocus parameter.
         */
        void setD(const dvec& d /**< [in] the array storing the translation information with each element storing defocus parameter */
                 );

        /**
         * @brief This function returns the array of weight of number of support points of the class subspace.
         * 
         * @return the array of weight of number of support points of the class subspace
         */
        dvec wC() const;

        /**
         * @brief This function sets the array of weight of number of support points of the class subspace.
         */
        void setWC(const dvec& wC /**< [in] the array of weight of number of support points of the class subspace */
                  );

        /**
         * @brief This function returns the array of weight of number of support points of the rotation subspace.
         * 
         * @return the array of weight of number of support points of the rotation subspace
         */
        dvec wR() const;

        /**
         * @brief This function sets the array of weight of number of support points of the rotation subspace.
         */
        void setWR(const dvec& wR /**< [in] the array of weight of number of support points of the rotation subspace */
                  );

        /**
         * @brief This function returns the array of weight of number of support points of the translation subspace.
         * 
         * @return the array of weight of number of support points of the translation subspace
         */
        dvec wT() const;

        /**
         * @brief This function sets the array of weight of number of support points of the translation subspace.
         */
        void setWT(const dvec& wT /**< [in] the array of weight of number of support points of the translation subspace */
                  );

        /**
         * @brief This function returns the array of weight of number of support points of the defocus subspace.
         * 
         * @return the array of weight of number of support points of the defocus subspace
         */
        dvec wD() const;

        /**
         * @brief This function sets the array of weight of number of support points of the defocus subspace.
         */
        void setWD(const dvec& wD /**< [in] the array of weight of number of support points of the defocus subspace */
                  );

        /**
         * @brief This function returns the array storing the likelihood of each support point.
         * 
         * @return the array storing the likelihood of each support point
         */
        dvec uC() const;

        /**
         * @brief This function sets the array storing the likelihood of each support point.
         */
        void setUC(const dvec& uC /**< [in] the array storing the likelihood of each support point */
                  );

        /**
         * @brief This function returns the array storing the likelihood of each support point.
         * 
         * @return the array storing the likelihood of each support point
         */
        dvec uR() const;

        /**
         * @brief This function sets the array storing the likelihood of each support point.
         */
        void setUR(const dvec& uR /**< [in] the array storing the likelihood of each support point */
                  );

        /**
         * @brief This function returns the array storing the likelihood of each support point.
         * 
         * @return the array storing the likelihood of each support point
         */
        dvec uT() const;

        /**
         * @brief This function sets the array storing the likelihood of each support point.
         */
        void setUT(const dvec& uT /**< [in] the array storing the likelihood of each support point */
                  );

        /**
         * @brief This function returns the array storing the likelihood of each support point.
         * 
         * @return the array storing the likelihood of each support point
         */
        dvec uD() const;

        /**
         * @brief This function sets the array storing the likelihood of each support point.
         */
        void setUD(const dvec& uD /**< [in] the array storing the likelihood of each support point */
                  );

        /**
         * @brief This function returns the translation parameter with greatest weight.
         * 
         * @return the translation parameter with greatest weight
         */
        dvec2 topT() const;

        /**
         * @brief This function sets the translation parameter with greatest weight.
         */
        void setTopT(const dvec2& topT /**< [in] the translation parameter with greatest weight */
                    );

        /**
         * @brief This function returns the translation parameter with greatest weight in the previous iteration.
         * 
         * @return the translation parameter with greatest weight in the previous iteration
         */
        dvec2 topTPrev() const;

        /**
         * @brief This function sets the translation parameter with greatest weight.
         */
        void setTopTPrev(const dvec2& topTPrev /**< [in] the translation parameter with greatest weight in the previous iteration */
                        );

        /**
         * @brief This function returns the symmetry.
         * 
         * @return a pointer points to the Symmetry object
         */
        const Symmetry* symmetry() const;

        /**
         * @brief This function sets the symmetry.
         */
        void setSymmetry(const Symmetry* sym /**< [in] a pointer points to the Symmetry object */
                        );

        /**
         * @brief This function generates the support points by loading relevant parameters.
         * 
         * This function generates the support points by loading the number of support points of rotation, translation and defocus subspace, the quaternion of translation, standard devaation of rotation, translation dvector, two sigmas of 2D Gaussian distribution of the translation, defocus factor and standard deviation of defocus factor.
         */
        void load(const int nR,      /**< [in] the number of support points of the rotation subspace */
                  const int nT,      /**< [in] the number of support points of the translation subspace */
                  const int nD,      /**< [in] the number of support points of the defocus subspace */
                  const dvec4& q,    /**< [in] the quaternion of rotation */
                  const double k1,   /**< [in] concnetration paramter of von Mises distribution of rotation */
                  const double k2,   /**< [in] concnetration paramter of von Mises distribution of rotation */
                  const double k3,   /**< [in] concnetration paramter of von Mises distribution of rotation */
                  const dvec2& t,    /**< [in] the translation vector */
                  const double s0,   /**< [in] sigma0 of 2D Gaussian distribution of the translation */
                  const double s1,   /**< [in] sigma1 of 2D Gaussian distribution of the translation */
                  const double d,    /**< [in] the defocus factor */
                  const double s,    /**< [in] the standard deviation of defocus factor */
                  const double score /**< [in] the score of these support points */
                 );

        /**
         * @brief This function returns the concentration parameters, including rotation, translation and defocus.
         */
        void vari(double& k1, /**< [out] concnetration paramter of von Mises distribution of rotation */
                  double& k2, /**< [out] concnetration paramter of von Mises distribution of rotation */
                  double& k3, /**< [out] concnetration paramter of von Mises distribution of rotation */
                  double& s0, /**< [out] sigma0 of 2D Gaussian distribution of the translation */
                  double& s1, /**< [out] sigma1 of 2D Gaussian distribution of the translation */
                  double& s   /**< [out] the standard deviation of defocus factor */
                 ) const;

        /**
         * @brief This function returns the concentration parameters, including rotation and translation.
         */
        void vari(double& rVari, /**< [out] the concentration parameter of the rotation */
                  double& s0,    /**< [out] sigma0 of 2D Gaussian distribution of the translation */
                  double& s1,    /**< [out] sigma1 of 2D Gaussian distribution of the translation */
                  double& s      /**< [out] the standard deviation of defocus factor */
                 ) const;

        /**
         * @brief This function returns the concentration parameter of the rotation
         * 
         * @return the concentration parameter of the rotation
         */
        double variR() const;

        /**
         * @brief This function returns the concentration parameter of the translation
         * 
         * @return the concentration parameter of the translation
         */
        double variT() const;

        /**
         * @brief This function returns the concentration parameter of the defocus
         * 
         * @return the concentration parameter of the defocus
         */
        double variD() const;

        /**
         * @brief This function returns the compression grade of the number of support points of the rotation subspace, The higher results usually means the higher deviation.
         * 
         * @return the compression grade of the number of support points of the rotation subspace
         */
        double compressR() const;

        /**
         * @brief This function returns the compression grade of the number of support points of the translation subspace, The higher results usually means the higher deviation.
         * 
         * @return the compression grade of the number of support points of the translation subspace
         */
        double compressT() const;

        /**
         * @brief This function returns the compression grade of the number of support points of the defocus subspace, The higher results usually means the higher deviation.
         * 
         * @return the compression grade of the number of support points of the defocus subspace
         */
        double compressD() const;

        /**
         * @brief This function returns the score of these support points, the particle filter with higher score will have greater weight in reconstruction process.
         * 
         * @return the score of these support points
         */
        double score() const;

        /**
         * @brief This function returns the weight of i-th support points of the class subspace.
         * 
         * @return the weight of i-th support points of the class subspace
         */
        double wC(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the weight of i-th support points of the class subspace.
         */
        void setWC(const double wC, /**< [in] the weight of the i-th support points of the class subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function multiplies a factor to the weight of i-th support points of the class subspace.
         */
        void mulWC(const double wC, /**< [in] weight to be multipied */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the weight of i-th support points of the rotation subspace.
         * 
         * @return the weight of i-th support points of the rotation subspace
         */
        double wR(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the weight of i-th support points of the rotation subspace.
         */
        void setWR(const double wR, /**< [in] the weight of the i-th support points of the rotation subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function multiplies a factor to the weight of i-th support points of the rotation subspace.
         */
        void mulWR(const double wR, /**< [in] weight to be multipied */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the weight of i-th support points of the translation subspace.
         * 
         * @return the weight of i-th support points of the translation subspace
         */
        double wT(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the weight of i-th support points of the translation subspace.
         */
        void setWT(const double wT, /**< [in] the weight of the i-th support points of the translation subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function multiplies a factor to the weight of i-th support points of the translation subspace.
         */
        void mulWT(const double wT, /**< [in] weight to be multipied */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the weight of i-th support points of the defocus subspace.
         * 
         * @return the weight of i-th support points of the defocus subspace
         */
        double wD(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the weight of i-th support points of the defocus subspace.
         */
        void setWD(const double wD, /**< [in] the weight of the i-th support points of the defocus subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function multiplies a factor to the weight of i-th support points of the defocus subspace.
         */
        void mulWD(const double wD, /**< [in] weight to be multipied */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the likelihood of i-th support points of the class subspace.
         * 
         * @return the likelihood of i-th support points of the class subspace
         */
        double uC(const int i /**< [in] index of support point */
                 ) const;
        /**
         * @brief This function sets the likelihood of i-th support points of the class subspace
         */
        void setUC(const double uC, /**< [in] the weight of the i-th support points of the class subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the likelihood of i-th support points of the rotation subspace.
         * 
         * @return the likelihood of i-th support points of the rotation subspace
         */
        double uR(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the likelihood of i-th support points of the rotation subspace.
         */
        void setUR(const double uR, /**< [in] the weight of the i-th support points of the rotation subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the likelihood of i-th support points of the translation subspace.
         * 
         * @return the likelihood of i-th support points of the translation subspace
         */
        double uT(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the likelihood of i-th support points of the translation subspace.
         */
        void setUT(const double uT, /**< [in] the weight of the i-th support points of the translation subspace */
                   const int i      /**< [in] index of support point */
                  );

        /**
         * @brief This function returns the likelihood of i-th support points of the defocus subspace.
         * 
         * @return the likelihood of i-th support points of the defocus subspace
         */
        double uD(const int i /**< [in] index of support point */
                 ) const;

        /**
         * @brief This function sets the likelihood of i-th support points of the defocus subspace.
         */
        void setUD(const double uD, /**< [in] the weight of the i-th support points of the defocus subspace */
                   const int i      /**< [in] index of support point */
                  );
        /**
         * @brief This function normalizes the dvector of the weights.
         */
        void normW();

        /**
         * @brief This function returns the class of the i-th support point.
         */
        void c(size_t& dst, /**< [out] the class */
               const int i  /**< [in]  the index of support point */
              ) const;

        /**
         * @brief This function sets the class of the i-th support point.
         */
        void setC(const size_t src, /**< [in] the class */
                  const int i       /**< [in] the index of support point */
                 );

        /**
         * @brief This function returns the 2D rotation matrix of the i-th support point.
         */
        void rot(dmat22& dst, /**< [out] the 2D rotation matrix */
                 const int i  /**< [in]  the index of support point */
                ) const;

        /**
         * @brief This function returns the 3D rotation matrix of the i-th support point.
         */
        void rot(dmat33& dst, /**< [out] the 3D rotation matrix */
                 const int i  /**< [in]  the index of support point */
                ) const;

        /**
         * @brief This function returns the translation dvector of the i-th support point.
         */
        void t(dvec2& dst, /**< [out] the translation dvector */
               const int i /**< [in]  the index of support point */
              ) const;

        /**
         * @brief This function sets the translation dvector of the i-th support point.
         */
        void setT(const dvec2& src, /**< [in] the translation dvector */
                  const int i       /**< [in] the index of support point */
                 );

        /**
         * @brief This function returns the quaternion of the i-th support point.
         */
        void quaternion(dvec4& dst, /**< [out] the quaternion */
                        const int i /**< [in]  the index of support point */
                       ) const;

        /**
         * @brief This function sets the quaternion of the i-th support point.
         */
        void setQuaternion(const dvec4& src, /**< [in] the quaternion */
                           const int i       /**< [in] the index of support point */
                          );

        /**
         * @brief This function returns the defocus factor of the i-th support point.
         */
        void d(double& d,  /**< [out] the defocus factor */
               const int i /**< [in]  the index of support point */
              ) const;

        /**
         * @brief This function sets the defocus factor of the i-th support point.
         */
        void setD(const double d, /**< [in] the defocus factor */
                  const int i     /**< [in] the index of support point */
                 );

        /**
         * @brief This function returns the first concnetration paramter of von Mises distribution of rotation.
         * 
         * @return the first concnetration paramter of von Mises distribution of rotation
         */
        double k1() const;

        /**
         * @brief This function sets the the first concnetration paramter of von Mises distribution of rotation.
         */
        void setK1(const double k1 /**< [in] the first concnetration paramter of von Mises distribution of rotation */
                  );

        /**
         * @brief This function returns the second concnetration paramter of von Mises distribution of rotation.
         * 
         * @return the second concnetration paramter of von Mises distribution of rotation
         */
        double k2() const;

        /**
         * @brief This function sets the the second concnetration paramter of von Mises distribution of rotation.
         */
        void setK2(const double k2 /**< [in] the second concnetration paramter of von Mises distribution of rotation */
                  );

        /**
         * @brief This function returns the third concnetration paramter of von Mises distribution of rotation.
         * 
         * @return the third concnetration paramter of von Mises distribution of rotation
         */
        double k3() const;

        /**
         * @brief This function sets the the third concnetration paramter of von Mises distribution of rotation.
         */
        void setK3(const double k3 /**< [in] the third concnetration paramter of von Mises distribution of rotation */
                  );

        /**
         * @brief This function returns the sigma0 of 2D Gaussian distribution of the translation.
         * 
         * @return the sigma0 of 2D Gaussian distribution of the translation
         */
        double s0() const;

        /**
         * @brief This function sets the sigma0 of 2D Gaussian distribution of the translation.
         */
        void setS0(const double s0 /**< [in] the sigma0 of 2D Gaussian distribution of the translation */
                  );

        /**
         * @brief This function returns the sigma1 of 2D Gaussian distribution of the translation.
         * 
         * @return the sigma1 of 2D Gaussian distribution of the translation
         */
        double s1() const;

        /**
         * @brief This function sets the sigma1 of 2D Gaussian distribution of the translation.
         */
        void setS1(const double s1 /**< [in] the sigma1 of 2D Gaussian distribution of the translation */
                  );

        /**
         * @brief This function returns the rho of 2D Gaussian distribution of translation.
         * 
         * @return the rho of 2D Gaussian distribution of translation
         */
        double rho() const;

        /**
         * @brief This function sets the rho of 2D Gaussian distribution of translation.
         */
        void setRho(const double rho /**< [in] the rho of 2D Gaussian distribution of translation */
                   );

        /**
         * @brief This function returns the sigma of 1D Gaussian distribution of defocus factor.
         * 
         * @return the sigma of 1D Gaussian distribution of defocus factor
         */
        double s() const;

        /**
         * @brief This function sets the sigma of 1D Gaussian distribution of defocus factor.
         */
        void setS(const double s /**< [in] the sigma of 1D Gaussian distribution of defocus factor */
                 );
        
        /**
         * @brief This function calculates the parameters with greatest weight.
         */
        void calRank1st(const ParticleType pt /**< [in] the support point type of this particle filter */
                       );

        /**
         * @brief This function calculates the concentration paramters, including
         *        rotation and translation.
         */
        void calVari(const ParticleType pt /**< [in] the support point type of this particle filter */
                    );

        /**
         * @brief This function calculates the score of this particle filter.
         */
        void calScore();

        /**
         * @brief This function performs a perturbation on the support points in this particle filter.
         */
        void perturb(const double pf,      /**< [in] the perturbation factor which stands for the portion of 
                                                     confidence area of perturbation of the confidence area
                                                     of the sampling points */
                     const ParticleType pt /**< [in] the support point type of this particle filter */
                    );

        /**
         * @brief This function resamples the support points in this particle filter.
         */
        void resample(const int n,          /**< [in] the number of new support points */
                      const ParticleType pt /**< [in] the support point type of this particle filter */
                     );

        /***
        void resample(const int nR,
                      const int nT,
                      const int nD);
        ***/

        // void resample();

        /**
         * This function resamples the support points in this particle filter with
         * adding a portion of global sampling points.
         *
         * @param alpha the portion of global sampling points in the resampled
         *              support points
         */
        /***
        void resample(const double alpha = 0);
        ***/

        /**
         * This function resamples the support points in this particle filter to a
         * given number of support points with adding a portion of global sampling
         * points.
         *
         * @param n     the number of sampling points of the resampled particle
         *              filter
         * @param alpha the portion of global sampling points in the resampled
         *              support points
         */
        /***
        void resample(const int n,
                      const double alpha = 0);
        ***/

        /**
         * This function returns the neff value of this particle filter, which
         * indicates the degengency of it.
         */
        /***
        double neff() const;

        void segment(const double thres);

        void flatten(const double thres);

        void sort();
        ***/

        /**
         * This function sorts all support points by their weight in a descending
         * order. It only keeps top N support points.
         *
         * @param n the number of support points to keep
         */
        //void sort(const int n);

        /**
         * @brief This function sorts all support points in a subspace by their weight in a descending order. It only keeps top N support points.
         */
        void sort(const int n,          /**< [in] the number of support points to keep */
                  const ParticleType pt /**< [in] the support point type of this particle filter */
                 );

        /**
         * @brief This function sorts all support points by their weight in a descending order. It only keeps top nC, nR, nT, nD support points for class parameter, rotation parameters, translation parameters and defocus parameter, respectively.
         */
        void sort(const int nC, /**< [in] the number of support points of the class subspace to keep */
                  const int nR, /**< [in] the number of support points of the rotation subspace to keep */
                  const int nT, /**< [in] the number of support points of the translation subspace to keep */
                  const int nD  /**< [in] the number of support points of the defocus subspace to keep */
                 );

        /**
         * @brief This function sorts all support points by their weight in a descending order.
         */
        void sort();

        /**
         * @brief This function returns the index of sorting of the support points' weight in a descending order.
         * 
         * @return the index of sorting of the support points' weight in a descending order.
         */
        uvec iSort(const ParticleType pt /**< [in] the support point type of this particle filter */
                  ) const;

        /**
         * @brief This function calculates the peak factor of this particle filter.
         */
        void setPeakFactor(const ParticleType pt /**< [in] the support point type of this particle filter */
                          );
        
        /**
         * @brief This function set the peak factor of this particle filter as default.
         */
        void resetPeakFactor();

        /**
         * @brief This function sets the small likelihood to zero.
         */
        void keepHalfHeightPeak(const ParticleType pt /**< [in] the support point type of this particle filter */
                               );

        /**
         * @brief This function update the most likely class, and returns whether the most likely class changed.
         * 
         * @return whether the most likely class changed.
         */
        bool diffTopC();

        /**
         * @brief This function returns the difference between the most likely rotations between two iterations. This function also resets the most likely rotatation.
         * 
         * @return the difference between the most likely rotations between two iterations
         */
        double diffTopR();

        /**
         * @brief This function returns the difference between the most likely translations between two iterations. This function also resets the most likely translation.
         * 
         * @return the difference between the most likely translations between two iterations
         */
        double diffTopT();

        /**
         * @brief This function returns the difference between the most likely defocus factor between two iterations. This function also resets the most likely defocus factor.
         * 
         * @return the most likely defocus factor between two iterations
         */
        double diffTopD();

        /**
         * @brief This function gives the most likely class.
         */
        void rank1st(size_t& cls /**< [out] the most likely class */
                    ) const;
        
        /**
         * @brief This function gives the most likely quaternion of rotation.
         */
        void rank1st(dvec4& quat /**< [out] the most likely class */
                    ) const;

        /**
         * @brief This function gives the most likely rotation matrix in 2D.
         */
        void rank1st(dmat22& rot /**< [out] the most likely rotation matrix in 2D */
                    ) const;

        /**
         * @brief This function gives the most likely rotation matrix in 3D.
         */
        void rank1st(dmat33& rot /**< [out] the most likely rotation matrix in 3D */
                    ) const;

        /**
         * @brief This function gives the most likely translation vector.
         */
        void rank1st(dvec2& tran /**< [out] the most likely translation vector */
                    ) const;

        /**
         * @brief This function gives the most likely defocus factor.
         */
        void rank1st(double& d /**< [out] the most likely defocus factor */
                    ) const;

        /**
         * @brief This function reports the most likely class, quaternion of rotation, translation vector and defocus factor.
         */
        void rank1st(size_t& cls, /**< [out] the most likely class */
                     dvec4& quat, /**< [out] the most likely quaternion of rotation */
                     dvec2& tran, /**< [out] the most likely translation vector */
                     double& d    /**< [out] the most likely defocus factor */
                    ) const;

        /**
         * @brief This function reports the most likely class, rotation matrix in 2D, translation vector and defocus factor.
         */
        void rank1st(size_t& cls, /**< [out] the most likely class */
                     dmat22& rot, /**< [out] the most likely rotation matrix in 2D */
                     dvec2& tran, /**< [out] the most likely translation vector */
                     double& d    /**< [out] the most likely defocus factor */
                    ) const;
        /**
         * @brief This function reports the most likely class, rotation matrix in 3D, translation vector and defocus factor.
         */
        void rank1st(size_t& cls, /**< [out] the most likely class */
                     dmat33& rot, /**< [out] the most likely rotation matrix in 3D */
                     dvec2& tran, /**< [out] the most likely translation vector */
                     double& d    /**< [out] the most likely defocus factor */
                    ) const;

        /**
         * @brief This function gives the class of a random support point.
         */
        void rand(size_t& cls /**< [out] the class of a random support point */
                 ) const;

        /**
         * @brief This function gives the quaternion of rotation of a random support point.
         */
        void rand(dvec4& quat /**< [out] the quaternion of rotation of a random support point */
                 ) const;

        /**
         * @brief This function gives the rotation matrix in 2D of a random support point.
         */
        void rand(dmat22& rot /**< [out] the rotation matrix in 2D of a random support point */
                 ) const;

        /**
         * @brief This function gives the rotation matrix in 3D of a random support point.
         */
        void rand(dmat33& rot /**< [out] the rotation matrix in 3D of a random support point */
                 ) const;

        /**
         * @brief This function gives the translation vector of a random support point.
         */
        void rand(dvec2& tran /**< [out] the translation vector of a random support point */
                 ) const;

        /**
         * @brief This function gives the defocus factor of a random support point.
         */
        void rand(double& d  /**< [out] the defocus factor */
                 ) const;

        /**
         * @brief This function gives the class, quaternion of rotation, translation vector and defocus factor of a random support point.
         */
        void rand(size_t& cls, /**< [out] the class of a random support point */
                  dvec4& quat, /**< [out] the quaternion of rotation of a random support point */
                  dvec2& tran, /**< [out] the translation vector of a random support point */
                  double& d    /**< [out] the defocus factor of a random support point */
                 ) const;

        /**
         * @brief This function gives the class, rotation matrix in 2D, translation dvector and defocus factor of a random support point.
         */
        void rand(size_t& cls, /**< [out] the class of a random support point of a random support point */
                  dmat22& rot, /**< [out] the rotation matrix in 2D of a random support point */
                  dvec2& tran, /**< [out] the translation vector of a random support point */
                  double& d    /**< [out] the defocus factor of a random support point */
                 ) const;

        /**
         * @brief This function gives the class, rotation matrix in 3D, translation vector and defocus factor of a random support point.
         */
        void rand(size_t& cls, /**< [out] the class of a random support point of a random support point */
                  dmat33& rot, /**< [out] the rotation matrix in 3D of a random support point */
                  dvec2& tran, /**< [out] the translation vector of a random support point */
                  double& d    /**< [out] the defocus factor of a random support point */
                 ) const;

        /**
         * @brief This function shuffles the sampling points.
         */
        void shuffle(const ParticleType pt /**< [in] the support point type of this particle filter */
                    );

        /**
         * @brief This function shuffles the sampling points.
         */
        void shuffle();

        /**
         * @brief This function calculates the prior weight of support points in this filter, which should be mutiplied by likelihood to get the true final weight.
         */
        void balanceWeight(const ParticleType pt /**< [in] the support point type of this particle filter */
                          );

        /**
         * @brief This function will copy the content to another Particle object.
         */
        void copy(Particle& that /**< [out] the destination object */
                 ) const;

        /**
         * @brief This function will copy the content to another Particle object.
         */
        Particle copy() const;
    
    private:

        /**
         * @brief This function symmetrises the support points in this particle filter according to the symmetry information. This operation will be only performed in 3D mode.
         */
        void symmetrise(const dvec4* anchor = NULL /**< [in] the anchor point of symmetrise */
                       );

        /**
         * @brief This function re-centres in the translation of the support points in this particle filter.
         */
        void reCentre();

        /**
         * @brief This function clears up the content in this particle filter.
         */
        void clear();
};

/**
 * @brief This function displays the information in this particle filter.
 */
void display(const Particle& par /**< [in] the particle filter */
            );

/**
 * @brief This function save this particle filter to a file.
 */
void save(const char filename[],   /**< [in] the file name for saving */
          const Particle& par,     /**< [in] the particle filter to be saved */
          const bool saveU = false /**< [in] determin save w or save u */
         );

void save(const char filename[],   /**< [in] the file name for saving */
          const Particle& par,     /**< [in] the particle filter to be saved */
          const ParticleType pt,   /**< [in] the support point type of this particle filter */
          const bool saveU = false /**< [in] determin save w or save u */
         );

/**
 * @brief This function load a particle filter from a file.
 *
 * @param particle the particle filter to be loaded
 * @param filename the file name for loading
 */
/***
void load(Particle& particle,
          const char filename[]);
          ***/

#endif  //PARTICLE_H

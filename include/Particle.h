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

#include "Typedef.h"
#include "Macro.h"
#include "Logging.h"

#include "Coordinate5D.h"
#include "Random.h"
#include "Euler.h"
#include "Functions.h"
#include "Symmetry.h"
#include "DirectionalStat.h"

using namespace std;

class Particle
{
    private:

        /**
         * numer of particles in this particle filter
         */
        int _n;

        /**
         * the standard deviation of translation, assuming the translation
         * follows a 2D Gaussian distribution
         */
        double _transS;

        /**
         * the re-center threshold of translation
         */
        double _transQ;

        /**
         * a table storing the rotation information with each row storing a
         * quaternion
         */
        mat4 _r;

        /**
         * a table storing the translation information with each row storing a
         * 2-vector with x and y respectively
         */
        mat2 _t;

        /**
         * a vector storing the weight of each particle
         */
        vec _w;
        
        /**
         * a pointer points to a Symmetry object which indicating the symmetry
         */
        const Symmetry* _sym = NULL;

        /**
         * concentration parameter of Angular Central Gaussian distribution of
         * rotation
         */
        double _k0 = 0;

        /**
         * concentration parameter of Angular Central Gaussian distribution of
         * rotation
         */
        double _k1 = 0;

        /**
         * sigma0 of 2D Gaussian distribution of translation
         */
        double _s0 = 0;

        /**
         * sigma1 of 2D Gaussian distribution of translation
         */
        double _s1 = 0;

        /**
         * rho of 2D Gaussian distribution of translation
         */
        double _rho = 0;

        /**
         * quaternion of the most likely rotation
         */
        vec4 _topR = vec4({1, 0, 0, 0});

    public:

        /**
         * default constructor of Particle
         */
        Particle();

        /**
         * constructor of Particle
         *
         * @param n number of particles in this particle filter
         * @param transS standard deviation of translation
         * @param transQ the re-center threshold of translation
         * @param sym symmetry of resampling space
         */
        Particle(const int n,
                 const double transS,
                 const double transQ = 0.01,
                 const Symmetry* sym = NULL);

        /**
         * deconstructor of Particle
         */
        ~Particle();

        /**
         * This function initialise Particle.
         *
         * @param transS stndard deviation of translation
         * @param transM the maximum translation
         * @param sym symmetry of resampling space
         */
        void init(const double transS,
                  const double transQ = 0.01,
                  const Symmetry* sym = NULL);

        /**
         * This function initialises Particle.
         *
         * @param n number of particles in this particle filter
         * @param transS stndard deviation of translation
         * @param transM the maximum translation
         * @param sym symmetry of resampling space
         */
        void init(const int n,
                  const double transS,
                  const double transQ = 0.01,
                  const Symmetry* sym = NULL);

        /**
         * This function resets the particles in this particle to a uniform
         * distribution.
         */
        void reset();

        /**
         * This function resets the particles in this particle filter to a uniform
         * distribution in rotation and 2D Gaussian distribution in translation
         * with a given number of sampling points.
         *
         * @param n number of particles in this particle filter
         */
        void reset(const int n);

        /**
         * This function resets the particles in this particle filter to a
         * uiform distribution in rotation with nR sampling points, and 2D
         * Gaussian distribution in translation with nT sampling points. The
         * total number of particles in this particle will be nR x nT. The
         * sampling points for the iR-th rotation and the iT-th translation will
         * be at (iR * nT + iT) index of the particles in this particle filter.
         *
         * @param nR the number of rotation in this particle filter
         * @param nT the number of translation in this particle filter
         */
        void reset(const int nR,
                   const int nT);

        /**
         * This function returns the number of particles in this particle
         * filter.
         */
        int n() const;

        /**
         * This function returns the concentration parameters, including
         * rotation and translation.
         *
         * @param k0 the concentration parameter of the rotation
         * @param k1 the concentration parameter of the rotation
         * @param s0 sigma0 of 2D Gaussian distribution of the translation
         * @param s1 sigma1 of 2D Gaussian distribution of the translation
         * @param rho rho of 2D Gaussian distribution of the translation
         */
        void vari(double& k0,
                  double& k1,
                  double& s0,
                  double& s1,
                  double& rho) const;

        /**
         * This function returns the concentration parameters, including
         * rotation and translation.
         *
         * @param rVar the concentration parameter of the rotation
         * @param s0 sigma0 of 2D Gaussian distribution of the translation
         * @param s1 sigma1 of 2D Gaussian distribution of the translation
         */
        void vari(double& rVari,
                  double& s0,
                  double& s1) const;

        /**
         * This function returns the weight of the i-th particle in this
         * particle filter.
         *
         * @param i the index of particle
         */
        double w(const int i) const;

        /**
         * This function sets the weight of the i-th particle in this particle
         * filter.
         *
         * @param w the weight of particle
         * @param i the index of particle
         */
        void setW(const double w,
                  const int i);

        /**
         * This function multiply the weight of the i-th particle in this
         * particle with a factor.
         *
         * @param w the factor
         * @param i the index of particle
         */
        void mulW(const double w,
                  const int i);

        /**
         * This function normalizes the vector of the weights.
         */
        void normW();

        /**
         * This function returns the 5D coordinates of the i-th particle.
         *
         * @param dst the 5D coordinate
         * @param i the index of particle
         */
        void coord(Coordinate5D& dst,
                   const int i) const;

        /**
         * This function returns the rotation matrix of the i-th particle.
         *
         * @param dst the rotation matrix
         * @param i the index of particle
         */
        void rot(mat33& dst,
                 const int i) const;

        /**
         * This function returns the translation vector of the i-th particle.
         *
         * @param dst the translation vector
         * @param i the index of particle
         */
        void t(vec2& dst,
               const int i) const;

        /**
         * This function sets the translation vector of the i-th particle.
         *
         * @param src the translation vector
         * @param i the index of particle
         */
        void setT(const vec2& src,
                  const int i);

        /**
         * This function returns the quaternion of the i-th particle.
         *
         * @param dst the quaternion
         * @param i the index of particle
         */
        void quaternion(vec4& dst,
                        const int i) const;

        /**
         * This function sets the symmetry.
         *
         * @param sym a pointer points to the Symmetry object
         */
        void setSymmetry(const Symmetry* sym);

        /**
         * This function calculates the concentration paramters, including
         * rotation and translation.
         */
        void calVari();

        /**
         * This function performs a perturbation on the particles in this
         * particle filter.
         *
         * @param pf perturbation factor, which stands for the portion of
         *           confidence area of perturbation of the confidence area
         *           of the sampling points
         */
        void perturb(const double pf = 0.2);

        /**
         * This function resamples the particles in this particle filter with
         * adding a portion of global sampling points.
         *
         * @param alpha the portion of global sampling points in the resampled
         *              particles
         */
        void resample(const double alpha = 0);

        /**
         * This function resamples the particles in this particle filter to a
         * given number of particles with adding a portion of global sampling
         * points.
         *
         * @param n the number of sampling points of the resampled particle
         *          filter
         * @param alpha the portion of global sampling points in the resampled
         *              particles
         */
        void resample(const int n,
                      const double alpha = 0);

        /**
         * This function returns the neff value of this particle filter, which
         * indicates the degengency of it.
         */
        double neff() const;

        /**
         * This function sorts all particles by their weight in a descending
         * order. It only keeps top N particles.
         *
         * @param n the number of particles to keep
         */
        void sort(const int n);

        /**
         * This function returns the index of sorting of the particles' weight
         * in a descending order.
         */
        uvec iSort() const;

        /**
         * This function returns the difference between the most likely
         * rotations between two iterations. This function also resets the most likely
         * rotatation.
         */
        double diffTop();

        /**
         * This function reports the 1-st rank coordinate by parameters.
         * 
         * @param quat the quaternion of the most likely rotation
         * @param tran the translation of the most likely coordinate
         */
        void rank1st(vec4& quat,
                     vec2& tran) const;

        /**
         * This function reports the 1-st rank coordinates by parameters.
         * 
         * @param rot the rotation matrix of the most likely rotation
         * @param tran the translation of the most likely coordinate
         */
        void rank1st(mat33& rot,
                     vec2& tran) const;

        /**
         * This function randomly reports a coordinate by parameters.
         *
         * @param quat the quaternion of the rotation
         * @param tran the translation
         */
        void rand(vec4& quat,
                  vec2& tran) const;

        /**
         * This function randomly reports a coordinate by parameters.
         *
         * @param quat the rotation matrix of the rotation
         * @param tran the translation
         */
        void rand(mat33& rot,
                  vec2& tran) const;

        void shuffle();
    
    private:

        /**
         * This function symmetrises the particles in this particle filter
         * according to the symmetry information.
         */
        void symmetrise();

        /**
         * This function re-centres in the translation of the particles in this
         * particle filter.
         */
        void reCentre();

        /**
         * This function clears up the content in this particle filter.
         */
        void clear();
};

/**
 * This function displays the information in this particle filter.
 *
 * @param particle the particle filter
 */
void display(const Particle& particle);

/**
 * This function save this particle filter to a file.
 *
 * @param filename the file name for saving
 * @param particle the particle filter to be saved
 */
void save(const char filename[],
          const Particle& particle);

/***
void load(Particle& particle,
          const char filename[]);
***/

#endif  //PARTICLE_H

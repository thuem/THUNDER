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

#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>

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
         * the translation range along X-axis is [_maxX, maxX]
         */
        double _maxX;

        /**
         * the translation range along Y-axis is [_maxY, maxY]
         */
        double _maxY;

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

    public:

        /**
         * default constructor of Particle
         */
        Particle();

        /**
         * constructor of Particle
         * @param n number of particles in this particle filter
         * @param maxX maximum X-axis translation in pixel
         * @param maxY maximum Y-axis translation in pixel
         * @param sym symmetry of resampling space
         */
        Particle(const int n,
                 const double maxX,
                 const double maxY,
                 const Symmetry* sym = NULL);

        /**
         * deconstructor of Particle
         */
        ~Particle();

        /**
         * This function initialises Particle.
         * @param n number of particles in this particle filter
         * @param maxX maximum X-axis translation in pixel
         * @param maxY maximum Y-axis translation in pixel
         * @param sym symmetry of resampling space
         */
        void init(const int n,
                  const double maxX,
                  const double maxY,
                  const Symmetry* sym = NULL);

        /**
         * This function resets the particles in this particle to a uniform
         * distribution.
         */
        void reset();

        /**
         * This function resets the particles in this particle to a uniform
         * distribution with a given number of sampling points.
         */
        void reset(const int n);

        /**
         * This function returns the number of particles in this particle
         * filter.
         */
        int n() const;

        /**
         * This function returns the concentration parameters, including
         * rotation and translation.
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
         * This function returns the weight of the i-th particle in this
         * particle filter.
         * @param i the index of particle
         */
        double w(const int i) const;

        /**
         * This function sets the weight of the i-th particle in this particle
         * filter.
         * @param w the weight of particle
         * @param i the index of particle
         */
        void setW(const double w,
                  const int i);

        /**
         * This function multiply the weight of the i-th particle in this
         * particle with a factor.
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
         * @param dst the 5D coordinate
         * @param i the index of particle
         */
        void coord(Coordinate5D& dst,
                   const int i) const;

        /**
         * This function returns the rotation matrix of the i-th particle.
         * @param dst the rotation matrix
         * @param i the index of particle
         */
        void rot(mat33& dst,
                 const int i) const;

        /**
         * This function returns the translation vector of the i-th particle.
         * @param dst the translation vector
         * @param i the index of particle
         */
        void t(vec2& dst,
               const int i) const;

        /**
         * This function sets the translation vector of the i-th particle.
         * @param src the translation vector
         * @param i the index of particle
         */
        void setT(const vec2& src,
                  const int i);

        /**
         * This function returns the quaternion of the i-th particle.
         * @param dst the quaternion
         * @param i the index of particle
         */
        void quaternion(vec4& dst,
                        const int i) const;

        /**
         * This function sets the symmetry.
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
         */
        void perturb();

        /**
         * This function resamples the particles in this particle filter with
         * adding a portion of global sampling points.
         * @param alpha the portion of global sampling points in the resampled
         * particles
         */
        void resample(const double alpha = 0);

        /**
         * This function resamples the particles in this particle filter to a
         * given number of particles with adding a portion of global sampling
         * points.
         * @param n the number of sampling points of the resampled particle
         * filter
         * @param alpha the portion of global sampling points in the resampled
         * particles
         */
        void resample(const int n,
                      const double alpha = 0);

        /**
         * This function returns the neff value of this particle filter, which
         * indicates the degengency of it.
         */
        double neff() const;

        /**
         * This function returns the index of sorting of the particles' weight
         * in a descending order.
         */
        uvec iSort() const;
    
    private:

        /**
         * This function symmetrises the particles in this particle filter
         * according to the symmetry information.
         */
        void symmetrise();

        /**
         * This function clears up the content in this particle filter.
         */
        void clear();
};

/**
 * This function displays the information in this particle filter.
 * @param particle the particle filter
 */
void display(const Particle& particle);

/**
 * This function save this particle filter to a file.
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

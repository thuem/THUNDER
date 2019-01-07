/** @file
 *  @author Mingxu Hu
 *  @author Bing Li
 *  @version 1.4.11.081221
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2015/03/23 | 0.0.1.050323  | new file
 *  Mingxu Hu   | 2018/12/21 | 1.4.11.081221 | add documentation
 *  Huabin Ruan | 2018/12/25 | 1.4.11.081225 | add support for adding description info in the thu file
 *
 * ****************************************************************************/

#ifndef DATABASE_H
#define DATABASE_H

/**
 * @brief index of the key standing for acceleration voltage of the microscopy
 */
#define THU_VOLTAGE 0
#define THU_VOLTAGE_FORMAT %18.9f

/**
 * @brief index of the key standing for defocus U
 */
#define THU_DEFOCUS_U 1
#define THU_DEFOCUS_U_FORMAT %18.9f

/**
 * @brief index of the key standing for defocus V
 */
#define THU_DEFOCUS_V 2
#define THU_DEFOCUS_V_FORMAT %18.9f

/**
 * @brief index of the key standing for defocus angle
 */
#define THU_DEFOCUS_THETA 3
#define THU_DEFOCUS_THETA_FORMAT %18.9f

/**
 * @brief index of the key standing for TODO
 */
#define THU_CS 4
#define THU_CS_FORMAT %18.9f

/**
 * @brief index of the key standing for amplitude contrast
 */
#define THU_AMPLITUTDE_CONTRAST 5
#define THU_AMPLITUTDE_CONTRAST_FORMAT %18.9f

/**
 * @brief index of the key standing for phase shift
 */
#define THU_PHASE_SHIFT 6
#define THU_PHASE_SHIFT_FORMAT %18.9f

/**
 * @brief index of the key standing for the directory path of each particle image
 */
#define THU_PARTICLE_PATH 7

/**
 * @brief format of THU_PARTICLE_PATH key in .thu file
 */
#define THU_PARTICLE_PATH_FORMAT %s

/**
 * @brief index of the key standing for the directory path of micrograph which this particle image belongs to
 */
#define THU_MICROGRAPH_PATH 8

/**
 * @brief format of THU_MICROGRAPH_PATH key in .thu file
 */
#define THU_MICROGRAPH_PATH_FORMAT %s

/**
 * @brief index of the key standing for X coordinate of this particle image in the micrograph
 */
#define THU_COORDINATE_X 9
#define THU_COORDINATE_X_FORMAT %18.9f

/**
 * @brief index of the key standing for Y coordinate of this particle image in the micrograph
 */
#define THU_COORDINATE_Y 10
#define THU_COORDINATE_Y_FORMAT %18.9f

/**
 * @brief index of the key standing for group ID
 */
#define THU_GROUP_ID 11

/**
 * @brief format of THU_GROUP_ID key in .thu file
 */
#define THU_GROUP_ID_FORMAT %6d

/**
 * @brief index of the key standing for class ID
 */
#define THU_CLASS_ID 12

/**
 * @brief format of THU_CLASS_ID key in .thu file
 */
#define THU_CLASS_ID_FORMAT %6d

/**
 * @brief index of the key standing for the 1st element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_0 13
#define THU_QUATERNION_0_FORMAT %18.9f

/**
 * @brief index of the key standing for the 2nd element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_1 14
#define THU_QUATERNION_1_FORMAT %18.9f

/**
 * @brief index of the key standing for the 3rd element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_2 15
#define THU_QUATERNION_2_FORMAT %18.9f

/**
 * @brief index of the key standing for the 4th element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_3 16
#define THU_QUATERNION_3_FORMAT %18.9f

/**
 * @brief index of the key standing for the 1st concentration parameter in the parameter matrix of ACG distribution
 */
#define THU_K1 17
#define THU_K1_FORMAT %18.9f

/**
 * @brief index of the key standing for the 2nd concentration parameter in the parameter matrix of ACG distribution
 */
#define THU_K2 18
#define THU_K2_FORMAT %18.9f

/**
 * @brief index of the key standing for the 3rd concentration parameter in the parameter matrix of ACG distribution
 */
#define THU_K3 19
#define THU_K3_FORMAT %18.9f

/**
 * @brief index of the key standing for the translation along X-axis
 */
#define THU_TRANSLATION_X 20
#define THU_TRANSLATION_X_FORMAT %18.9f

/**
 * @brief index of the key standing for the translation along Y-axis
 */
#define THU_TRANSLATION_Y 21
#define THU_TRANSLATION_Y_FORMAT %18.9f

/**
 * @brief index of the key standign for the standard deviation of the translation along X-axis
 */
#define THU_STD_TRANSLATION_X 22
#define THU_STD_TRANSLATION_X_FORMAT %18.9f

/**
 * @brief index of the key standing for the standard deviation of the translation along Y-axis
 */
#define THU_STD_TRANSLATION_Y 23
#define THU_STD_TRANSLATION_X_FORMAT %18.9f

// #define THU_RHO_TRANSLATION_XY 24
// #define THU_RHO_TRANSLATION_XY_FORMAT %18.9f

/**
 * @brief index of the key standing for the defocus factor
 */
#define THU_DEFOCUS_FACTOR 24
#define THU_DEFOCUS_FACTOR_FORMAT %18.9f

/**
 * @brief index of the key standing for the standard deviation of defoucs factor
 */
#define THU_STD_DEFOCUS_FACTOR 25
#define THU_STD_DEFOCUS_FACTOR_FORMAT %18.9f

/**
 * @brief index of the key standing for the quality score of each particle image
 */
#define THU_SCORE 26
#define THU_SCORE_FORMAT %18.9f

#include <cstring>
#include <cstdio>
#include <iostream>

#include "Typedef.h"
#include "Macro.h"
#include "Precision.h"

#include "Parallel.h"
#include "Utils.h"
#include "Random.h"

/**
 * @brief CTF attributes
 */
struct CTFAttr
{
    /**
     * @brief acceleration voltage of the micrography
     */
    RFLOAT voltage;

    /**
     * @brief defocus U
     */
    RFLOAT defocusU;

    /**
     * @brief defocus V
     */
    RFLOAT defocusV;

    /**
     * @brief defocus angle
     */
    RFLOAT defocusTheta;

    /**
     * @brief TODO
     */
    RFLOAT Cs;

    /**
     * @brief amplitude contrast
     */
    RFLOAT amplitudeContrast;

    /**
     * @brief phase shift
     */
    RFLOAT phaseShift;
};

/**
 * @brief This class manages .thu file, including reading, writing and shuffling information.
 */
class Database : public Parallel
{
    private:

        /**
         * @brief FILE pointer which points to the .thu file
         */
        FILE* _db;

        /**
         * @brief the ID of the first particle assigned to this process
         */
        int _start;
        
        /**
         * @brief the ID of the last particle assigned to of this process
         */
        int _end;

        /**
         * @brief the offset in File of each line (particle)
         */
        vector<long> _offset;

        /**
         * @brief the register of each particle
         */
        vector<int> _reg;

    public:

        /**
         * @brief default constructor
         */
        Database();

        /**
         * @brief constructor given the filename of the .thu file
         */
        Database(const char database[] /**< [in] the filename of the .thu file */
                );

        /**
         * @brief default deconstructor
         */
        ~Database();

        /**
         * @brief open a .thu file for reading or writing information
         */
        void openDatabase(const char database[] /**< [in] the filename of the .thu file */
                         );


        void openDatabase(const char *database, const char *outputPath,  const int rank);

        /**
         * @brief save information to a .thu file
         */
        void saveDatabase(const char database[] /**< [in] the filename of the .thu file */
                         );

        /**
         * @brief TODO
         */
        int start() const { return _start; };

        /**
         * @brief TODO
         */
        int end() const { return _end; };

        /**
         * @brief total number of particles
         */
        int nParticle() const;

        /**
         * @brief total number of groups
         */
        int nGroup() const;

        /**
         * @brief number of particles assigned to this process
         */
        int nParticleRank();

        /**
         * @brief assign particles to each process
         */
        void assign();

        /**
         * @brief record the shift of each line, prepare for later use
         */
        void index();

        /**
         * @brief shuffle particles
         */
        void shuffle();

        /**
         * @brief TODO
         *
         * @return TODO
         */
        long offset(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns X coordinate of the i-th particle image in the micrograph.
         *
         * @return X coordinate of the i-th particle image in the micrograph
         */
        RFLOAT coordX(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns Y coordinate of the i-th particle image in the micrograph.
         *
         * @return Y coordinate of the i-th particle image in the micrograph
         */
        RFLOAT coordY(const int i /**< [in] index of particle */) const;
        
        /**
         * @brief This function returns group ID of the i-th particle image in the micrograph.
         *
         * @return group ID of the i-th particle image in the micrograph
         */
        int groupID(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns directory path of the i-th particle image in the micrograph.
         *
         * @return directory path of the i-th particle image in the micrograph
         */
        string path(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns directory path of micrograph which the i-th particle image belongs to.
         *
         * @return directory path of micrograph which the i-th particle image belongs to
         */
        string micrographPath(const int i /**< [in] index of particle */) const;
        
        /**
         * @brief TODO
         */
        void ctf(RFLOAT& voltage,            /**< [out] acceleration voltage of microscopy */
                 RFLOAT& defocusU,           /**< [out] TODO */
                 RFLOAT& defocusV,           /**< [out] TODO */
                 RFLOAT& defocusTheta,       /**< [out] TODO */
                 RFLOAT& Cs,                 /**< [out] TODO */
                 RFLOAT& amplitudeConstrast, /**< [out] TODO */
                 RFLOAT& phaseShift,         /**< [out] TODO */
                 const int i                 /**< [in] index of particle */
                ) const;

        /**
         * @brief TODO
         */
        void ctf(CTFAttr& dst, /**< [out] TODO */
                 const int i   /**< [in] index of particle */
                ) const;

        /**
         * @brief This function returns the class ID of the i-th particle.
         *
         * @return the classID of the i-th particle
         */
        int cls(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns the unit quaternion of the i-th particle.
         *
         * @return the unit quaternion of the i-th particle
         */
        dvec4 quat(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns the 1st concentration parameter of the parameter matrix of ACG distribution.
         *
         * @return the-1st concentration parameter of the parameter matrix of ACG distribution
         */
        RFLOAT k1(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns the 2nd concentration parameter of the parameter matrix of ACG distribution.
         *
         * @return the 2nd concentration parameter of the parameter matrix of ACG distribution
         */
        RFLOAT k2(const int i /**< [in] index of particle */) const;

        /**
         * @brief This function returns the 3rd concentration parameter of the parameter matrix of ACG distribution.
         *
         * @return the 3rd concentration parameter of the parameter matrix of ACG distribution
         */
        RFLOAT k3(const int i /**< [in] index of particle */) const;

        /**
         * @brief TODO
         */
        dvec2 tran(const int i /**< [in] index of particle */) const;

        /**
         * @brief TODO
         */
        RFLOAT stdTX(const int i /**< [in] index of particle */) const;

        /**
         * @brief TODO
         */
        RFLOAT stdTY(const int i /**< [in] index of particle */) const;

        /**
         * @brief TODO
         */
        RFLOAT d(const int i) const;

        /**
         * @brief TODO
         */
        RFLOAT stdD(const int i) const;

        /**
         * @brief This function returns the quality score of each particle.
         *
         * @return the quality of each particle
         */
        RFLOAT score(const int i /**< [in] particle index */) const;

    private:

        /**
         * @brief This function calculates the ID of the first and the last particle assigned to this process.
         */
        void split(int& start,        /**< [out] the ID of the first particle assigned to this process */
                   int& end,          /**< [out] the ID of the last particle assigned to this process */
                   const int commRank /**< [in] the rank of this process */
                  );

        /**
         *  @brief This function generates a new thu database withou containing comments lines
         */

        void reGenDatabase(char *newDatabase,    /**< [out] Full name of the new database */
                           const char *outputDir,/**< [in] Directory used to save new database */
                           const char *database, /**< [in] Original database name */ 
                           const int rank        /**< [in] the rank of current process */
                          );
};

#endif // DATABASE_H

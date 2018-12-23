/** @file
 *  @author Mingxu Hu
 *  @author Bing Li
 *  @version 1.4.11.081221
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Mingxu Hu | 2018/12/21 | 1.4.11.081221 | add documentation
 *
 * ****************************************************************************/

#ifndef DATABASE_H
#define DATABASE_H

/**
 * @brief index of the key standing for acceleration voltage of the microscopy
 */
#define THU_VOLTAGE 0

/**
 * @brief format of THU_VOLTAGE key in .thu file
 */
#define THU_VOLTAGE_FORMAT %12.6f

/**
 * @brief index of the key standing for defocus U
 */
#define THU_DEFOCUS_U 1

/**
 * @brief format of THU_DEFOCUS_U key in .thu file
 */
#define THU_DEFOCUS_U_FORMAT %12.6f

/**
 * @brief index of the key standing for defocus V
 */
#define THU_DEFOCUS_V 2

/**
 * @brief format of THU_DEFOCUS_V key in .thu file
 */
#define THU_DEFOCUS_V_FORMAT %12.6f

/**
 * @brief index of the key standing for defocus angle
 */
#define THU_DEFOCUS_THETA 3

/**
 * @brief format of THU_DEFOCUS_THETA key in .thu file
 */
#define THU_DEFOCUS_THETA_FORMAT %12.6f

/**
 * @brief index of the key standing for //TODO
 */
#define THU_CS 4

/**
 * @brief format of THU_CS key in .thu file
 */
#define THU_CS_FORMAT %12.6f

/**
 * @brief index of the key standing for amplitude contrast
 */
#define THU_AMPLITUTDE_CONTRAST 5

/**
 * @brief format of THU_AMPLITUTDE_CONTRAST key in .thu file
 */
#define THU_AMPLITUTDE_CONTRAST_FORMAT %12.6f

/**
 * @brief index of the key standing for phase shift
 */
#define THU_PHASE_SHIFT 6

/**
 * @brief format of THU_PHASE_SHIFT key in .thu file
 */
#define THU_PHASE_SHIFT_FORMAT %12.6f

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

/**
 * @brief format of THU_COORDINATE_X key in .thu file
 */
#define THU_COORDINATE_X_FORMAT %12.6f

/**
 * @brief index of the key standing for Y coordinate of this particle image in the micrograph
 */
#define THU_COORDINATE_Y 10

/**
 * @brief format of THU_COORDINATE_Y key in .thu file
 */
#define THU_COORDINATE_Y_FORMAT %12.6f

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

/**
 * @brief format of THU_QUATERNION_0 key in .thu file
 */
#define THU_QUATERNION_0_FORMAT %12.6f

/**
 * @brief index of the key standing for the 2nd element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_1 14

/**
 * @brief format of THU_QUATERNION_1 key in .thu file
 */
#define THU_QUATERNION_1_FORMAT %12.6f

/**
 * @brief index of the key standing for the 3rd element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_2 15

/**
 * @brief format of THU_QUATERNION_2 key in .thu file
 */
#define THU_QUATERNION_2_FORMAT %12.6f

/**
 * @brief index of the key standing for the 4th element of the unit quaternion representing a rotation
 */
#define THU_QUATERNION_3 16

/**
 * @brief format of THU_QUATERNION_3 key in .thu file
 */
#define THU_QUATERNION_3_FORMAT %12.6f

/**
 * @brief index of the key standing for the 1st concentration parameter in the parameter matrix of ACG distribution
 */
#define THU_K1 17

/**
 * @brief format of THU_K1 key in .thu file
 */
#define THU_K1_FORMAT %12.6f

/**
 * @brief index of the key standing for the 2nd concentration parameter in the parameter matrix of ACG distribution
 */
#define THU_K2 18

/**
 * @brief format of THU_K2 key in .thu file
 */
#define THU_K2_FORMAT %12.6f

/**
 * @brief index of the key standing for the 3rd concentration parameter in the parameter matrix of ACG distribution
 */
#define THU_K3 19

/**
 * @brief format of THU_K3 key in .thu file
 */
#define THU_K3_FORMAT %12.6f

/**
 * @brief index of the key standing for the translation along X-axis
 */
#define THU_TRANSLATION_X 20

/**
 * @brief format of THU_TRANSLATION_X key in .thu file
 */
#define THU_TRANSLATION_X_FORMAT %12.6f

/**
 * @brief index of the key standing for the translation along Y-axis
 */
#define THU_TRANSLATION_Y 21

/**
 * @brief format of THU_TRANSLATION_Y key in .thu file
 */
#define THU_TRANSLATION_Y_FORMAT %12.6f

/**
 * @brief index of the key standign for the standard deviation of the translation along X-axis
 */
#define THU_STD_TRANSLATION_X 22

/**
 * @brief format of THU_STD_TRANSLATION_X key in .thu file
 */
#define THU_STD_TRANSLATION_X_FORMAT %12.6f

/**
 * @brief index of the key standing for the standard deviation of the translation along Y-axis
 */
#define THU_STD_TRANSLATION_Y 23

/**
 * @brief format of THU_STD_TRANSLATION_Y key in .thu file
 */
#define THU_STD_TRANSLATION_Y_FORMAT %12.6f

/**
 * @brief index of the key standing for the defocus factor
 */
#define THU_DEFOCUS_FACTOR 24

/**
 * @brief format of THU_DEFOCUS_FACTOR key in .thu file
 */
#define THU_DEFOCUS_FACTOR_FORMAT %12.6f

/**
 * @brief index of the key standing for the standard deviation of defoucs factor
 */
#define THU_STD_DEFOCUS_FACTOR 25

/**
 * @brief format of THU_STD_DEFOCUS_FACTOR key in .thu file
 */
#define THU_STD_DEFOCUS_FACTOR_FORMAT %12.6f

/**
 * @brief index of the key standing for the quality score of each particle image
 */
#define THU_SCORE 26

/**
 * @brief format of THU_SCORE key in .thu file
 */
#define THU_SCORE_FORMAT %12.6f

#include <cstring>
#include <cstdio>
#include <iostream>

#include "Typedef.h"
#include "Macro.h"
#include "Precision.h"

#include "Parallel.h"
#include "Utils.h"
#include "Random.h"

struct CTFAttr
{
    RFLOAT voltage;

    RFLOAT defocusU;

    RFLOAT defocusV;

    RFLOAT defocusTheta;

    RFLOAT Cs;

    RFLOAT amplitudeContrast;

    RFLOAT phaseShift;
};

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

        /**
         * @brief save information to a .thu file
         */
        void saveDatabase(const char database[] /**< [in] the filename of the .thu file */
                         );

        /**
         * @brief //TODO
         */
        int start() const { return _start; };

        /**
         * @biref //TODO
         */
        int end() const { return _end; };

        /**
         * @brief total number of particles //TODO
         */
        int nParticle() const;

        /**
         * total number of groups
         */
        int nGroup() const;

        /**
         * number of particles assigned to this process
         */
        int nParticleRank();

        /**
         * assign particles to each process
         */
        void assign();

        /**
         * record the shift of each line, prepare for later use
         */
        void index();

        void shuffle();

        long offset(const int i) const;

        RFLOAT coordX(const int i) const;

        RFLOAT coordY(const int i) const;
        
        int groupID(const int i) const;

        string path(const int i) const;

        string micrographPath(const int i) const;
        
        void ctf(RFLOAT& voltage,
                 RFLOAT& defocusU,
                 RFLOAT& defocusV,
                 RFLOAT& defocusTheta,
                 RFLOAT& Cs,
                 RFLOAT& amplitudeConstrast,
                 RFLOAT& phaseShift,
                 const int i) const;

        void ctf(CTFAttr& dst,
                 const int i) const;

        int cls(const int i) const;

        dvec4 quat(const int i) const;

        RFLOAT k1(const int i) const;

        RFLOAT k2(const int i) const;

        RFLOAT k3(const int i) const;

        dvec2 tran(const int i) const;

        RFLOAT stdTX(const int i) const;

        RFLOAT stdTY(const int i) const;

        RFLOAT d(const int i) const;

        RFLOAT stdD(const int i) const;

        RFLOAT score(const int i) const;

    private:

        void split(int& start,
                   int& end,
                   int commRank);
};

#endif // DATABASE_H

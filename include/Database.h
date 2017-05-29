/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef DATABASE_H
#define DATABASE_H

#define THU_VOLTAGE 0
#define THU_VOLTAGE_FORMAT %12.6f

#define THU_DEFOCUS_U 1
#define THU_DEFOCUS_U_FORMAT %12.6f

#define THU_DEFOCUS_V 2
#define THU_DEFOCUS_V_FORMAT %12.6f

#define THU_DEFOCUS_THETA 3
#define THU_DEFOCUS_THETA_FORMAT %12.6f

#define THU_CS 4
#define THU_CS_FORMAT %12.6f

#define THU_AMPLITUTDE_CONTRAST 5
#define THU_AMPLITUTDE_CONTRAST_FORMAT %12.6f

#define THU_PHASE_SHIFT 6
#define THU_PHASE_SHIFT_FORMAT %12.6f

#define THU_PARTICLE_PATH 7
#define THU_PARTICLE_PATH_FORMAT %s

#define THU_MICROGRAPH_PATH 8
#define THU_MICROGRAPH_PATH_FORMAT %s

#define THU_COORDINATE_X 9
#define THU_COORDINATE_X_FORMAT %12.6f

#define THU_COORDINATE_Y 10
#define THU_COORDINATE_Y_FORMAT %12.6f

#define THU_GROUP_ID 11
#define THU_GROUP_ID_FORMAT %6d

#define THU_CLASS_ID 12
#define THU_CLASS_ID_FORMAT %6d

#define THU_QUATERNION_0 13
#define THU_QUATERNION_0_FORMAT %12.6f

#define THU_QUATERNION_1 14
#define THU_QUATERNION_1_FORMAT %12.6f

#define THU_QUATERNION_2 15
#define THU_QUATERNION_2_FORMAT %12.6f

#define THU_QUATERNION_3 16
#define THU_QUATERNION_3_FORMAT %12.6f

#define THU_STD_ROTATION 17
#define THU_STD_ROTATION_FORMAT %12.6f

#define THU_TRANSLATION_X 18
#define THU_TRANSLATION_X_FORMAT %12.6f

#define THU_TRANSLATION_Y 19
#define THU_TRANSLATION_Y_FORMAT %12.6f

#define THU_STD_TRANSLATION_X 20
#define THU_STD_TRANSLATION_X_FORMAT %12.6f

#define THU_STD_TRANSLATION_Y 21
#define THU_STD_TRANSLATION_X_FORMAT %12.6f

#define THU_DEFOCUS_FACTOR 22
#define THU_DEFOCUS_FACTOR_FORMAT %12.6f

#define THU_STD_DEFOCUS_FACTOR 23
#define THU_STD_DEFOCUS_FACTOR_FORMAT %12.6f

#define THU_SCORE 24
#define THU_SCORE_FORMAT %12.6f

#include <cstring>
#include <cstdio>
#include <iostream>

#include "Typedef.h"
#include "Macro.h"

#include "Parallel.h"
#include "Utils.h"
#include "Random.h"

struct CTFAttr
{
    double voltage;

    double defocusU;

    double defocusV;

    double defocusTheta;

    double Cs;

    double amplitudeContrast;

    double phaseShift;
};

class Database : public Parallel
{
    private:

        FILE* _db;

        /**
         * the ID of the first particle of this process
         */
        int _start;
        
        /**
         * the ID of the last particle of this process
         */
        int _end;

        /**
         * the offset in File of each line (particle)
         */
        vector<long> _offset;

        /**
         * the register of each particle
         */
        vector<int> _reg;

    public:

        Database();

        Database(const char database[]);

        ~Database();

        /**
         * open a .thu file
         */
        void openDatabase(const char database[]);

        /**
         * save a .thu file
         */
        void saveDatabase(const char database[]);

        int start() const { return _start; };

        int end() const { return _end; };

        /**
         * total number of particles
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

        double coordX(const int i) const;

        double coordY(const int i) const;
        
        int groupID(const int i) const;

        string path(const int i) const;

        string micrographPath(const int i) const;
        
        void ctf(double& voltage,
                 double& defocusU,
                 double& defocusV,
                 double& defocusTheta,
                 double& Cs,
                 double& amplitudeConstrast,
                 double& phaseShift,
                 const int i) const;

        void ctf(CTFAttr& dst,
                 const int i) const;

        int cls(const int i) const;

        vec4 quat(const int i) const;

        double stdR(const int i) const;

        vec2 tran(const int i) const;

        double stdTX(const int i) const;

        double stdTY(const int i) const;

        double d(const int i) const;

        double stdD(const int i) const;

    private:

        void split(int& start,
                   int& end,
                   int commRank);
};

#endif // DATABASE_H

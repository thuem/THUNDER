/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef DATABASE_H
#define DATABASE_H

#include "Parallel.h"
#include "Utils.h"

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
        int nParticle();

        /**
         * total number of groups
         */
        int nGroup();

        /**
         * number of particles assigned to this process
         */
        int nParticleRank();

        /**
         * assign particles to each process
         */
        void assign();

        void split(int& start,
                   int& end,
                   int commRank);
};

#endif // DATABASE_H

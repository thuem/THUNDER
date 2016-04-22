/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <vector>

#include "Database.h"

class Experiment : public Database
{
    private:
        void add_column(const char* table, const char* column, const char* attr);
        void get_id(std::vector<int>& dst, const char* table, const char* column, int value);

    public:

        Experiment();

        Experiment(const char database[]);

        void addColumnXOff();

        void addColumnYOff();

        void particleIDsMicrograph(vector<int>& dst,
                                   const int micrographID);
        /* return IDs of particles belonging to a certain micrograph */

        void particleIDsGroup(vector<int>& dst,
                              const int groupID);
        /* return IDs of particles belonging to a certain group*/
};

#endif // EXPERIMENT_H

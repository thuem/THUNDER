/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef Experiment_H
#define Experiment_H

#include "Database.h"

class Experiment : public Database
{
    public:

        Experiment();

        Experiment(const char database[]);

        void addColumnXOff();

        void addColumnYOff();
};

#endif // Experiment_H

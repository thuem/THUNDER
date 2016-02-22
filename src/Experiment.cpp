/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Experiment.h"

Experiment::Experiment() : Database() {}

Experiment::Experiment(const char database[]) : Database(database) {}

void Experiment::addColumnXOff()
{
    ADD_COLUMN(particles, XOff, integer);
}

void Experiment::addColumnYOff()
{
    ADD_COLUMN(particles, YOff, integer);
}

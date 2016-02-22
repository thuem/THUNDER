/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Experiment.h"

using namespace std;

int main(int argc, const char* argv[])
{
    Experiment exp("test.db");

    exp.createTableParticles();
    exp.addColumnXOff();
    // exp.addColumnYOff();
}

/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Experiment.h"

#define N 1000

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
    loggerInit();

    Experiment exp("test.db");

    exp.createTableParticles();
    exp.addColumnXOff();
    exp.addColumnYOff();

    exp.createTableMicrographs();

    for (int i = 0; i < N; i++)
        exp.appendParticle("", i / 10, i / 10);

    vector<int> partIDs;
    exp.particleIDsMicrograph(partIDs, 0);
    for (size_t i = 0; i < partIDs.size(); i++)
        cout << partIDs[i] << endl;
    exp.particleIDsGroup(partIDs, 0);
    for (size_t i = 0; i < partIDs.size(); i++)
        cout << partIDs[i] << endl;
}

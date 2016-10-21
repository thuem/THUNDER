/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "MLOptimiser.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    // double a[10] = {0.8, 0.3, 0.5, 1, 0.3, 0.2, 0.1, 0.5, 0, 0.1};
    double a[10] = {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2};

    unsigned int r[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    unsigned int t[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    double b = atof(argv[1]);

    cout << searchPlace(a, b, 0, 10) << endl;

    recordTopK(a, r, t, b, 100, 100, 10);

    for (int i = 0; i < 10; i++)
        cout << i << ": " << a[i] << ", " << r[i] << ", " << t[i] << endl;

    return 0;
}

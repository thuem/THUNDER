/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Functions.h"

using namespace std;

int main(int argc, const char* argv[])
{
    /***
    for (double i = 0; i <= 1.5; i += 0.01)
        cout << i << " " << MKB_FT(i, 1, 0.5) << endl;
    ***/

    for (double i = 0; i <= 1.5; i += 0.01)
        cout << i << " " << MKB_RL(i, 1, 0.5) << endl;
}

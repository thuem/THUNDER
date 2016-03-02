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
    for (double i = 0; i <= 2.5; i += 0.01)
        cout << i << " " << MKB_FT(i, 2, 15) << endl;

    /***
    for (double i = 0; i <= 1.5; i += 0.01)
        cout << i << " " << MKB_RL(i, 2, 0.5) << endl;
    ***/
}

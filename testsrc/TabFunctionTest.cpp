/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "TabFunction.h"

using namespace std;

using namespace placeholders;

int main(int argc, const char* argv[])
{
    TabFunction tab(bind(MKB_FT, _1, 2, atoi(argv[1])), 0, 2.5, 100000);

    // cout << atoi(argv[1]) << endl;
    for (double i = 0; i <= 2.5; i += 0.01)
        cout << i << " " << tab(i) << endl;

    /***
    for (double i = 0; i <= 1.5; i += 0.01)
        cout << i << " " << MKB_RL(i, 2, 0.5) << endl;
    ***/
}

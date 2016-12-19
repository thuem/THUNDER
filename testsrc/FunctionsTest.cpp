/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Functions.h"

#define BLOB_A 1.9
#define BLOB_ALPHA 15

#define PF 2

using namespace std;

int main(int argc, const char* argv[])
{
    // cout << atoi(argv[1]) << endl;
    /***
    for (double i = 0; i <= 2.5; i += 0.01)
        cout << i << " " << MKB_FT(i, 2, atoi(argv[1])) << endl;
        ***/

    /***
    for (double i = 0; i <= 1.5; i += 0.01)
        cout << i << " " << MKB_RL(i, 2, 0.5) << endl;
    ***/
    
    /***
    for (double i = 0; i <= 0.5; i += 0.01)
        cout << i << " " << MKB_RL(i, PF * BLOB_A, BLOB_ALPHA) << endl;
    ***/

    for (double i = 0; i <= 0.5; i += 0.01)
        cout << i << " " << TIK_RL(i) << endl;

    /***
    for (double i = 0; i <= BLOB_A * PF; i += 0.01)
        cout << i << " " << MKB_FT(i, BLOB_A * PF, BLOB_ALPHA) << endl;
        ***/

    //cout << MKB_BLOB_VOL(BLOB_A * PF, BLOB_ALPHA) << endl;
}

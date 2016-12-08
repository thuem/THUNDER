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
#define BLOB_ALPHA 10

#define PF 2



int main(int argc, const char* argv[])
{
    // std::cout << atoi(argv[1]) << std::endl;
    /***
    for (double i = 0; i <= 2.5; i += 0.01)
        std::cout << i << " " << MKB_FT(i, 2, atoi(argv[1])) << std::endl;
    ***/

    /***
    for (double i = 0; i <= 1.5; i += 0.01)
        std::cout << i << " " << MKB_RL(i, 2, 0.5) << std::endl;
    ***/
    
    /***
    for (double i = 0; i <= BLOB_A * PF; i += 0.01)
        std::cout << i << " " << MKB_RL(i, PF * BLOB_A, BLOB_ALPHA) << std::endl;
    ***/

    for (double i = 0; i <= 1.; i += 0.01)
        std::cout << i << " " << TIK_RL(i) << std::endl;

    /***
    for (double i = 0; i <= BLOB_A * PF; i += 0.01)
        std::cout << i << " " << MKB_FT(i, BLOB_A * PF, BLOB_ALPHA) << std::endl;
        ***/
}

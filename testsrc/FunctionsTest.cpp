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

//#define TEST_MKB_FT
//#define TEST_MKB_RL
//#define TEST_TIL_RL
//#define TEST_MKB_BLOB_VOL

int main(int argc, const char* argv[])
{
#ifdef TEST_MKB_FT
    for (double i = 0; i <= BLOB_A * PF; i += 0.01)
        std::cout << i << " " << MKB_FT(i, BLOB_A * PF, BLOB_ALPHA) << std::endl;
#endif

#ifdef TEST_MKB_RL
    for (double i = 0; i <= 0.5; i += 0.01)
        std::cout << i << " " << MKB_RL(i, BLOB_A * PF, BLOB_ALPHA) << std::endl;
#endif

#ifdef TEST_TIL_RL
    for (double i = 0; i <= 0.5; i += 0.01)
        std::cout << i << " " << TIK_RL(i) << std::endl;
#endif

#ifdef TEST_MKB_BLOB_VOL
    std::cout << MKB_BLOB_VOL(BLOB_A * PF, BLOB_ALPHA) << std::endl;
#endif
}

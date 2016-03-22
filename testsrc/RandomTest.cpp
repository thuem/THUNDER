/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Random.h"

using namespace std;

int main(int argc, const char* argv[])
{
    char r[10];
    rand(r, 10);
    for (int i = 0; i < 10; i++)
        cout << r[i] << endl;
}

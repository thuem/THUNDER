/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "TabFunction.h"

#include <boost/bind.hpp>

int main(int argc, const char* argv[])
{
    TabFunction tab(boost::bind(MKB_FT, boost::placeholders::_1, 2, atoi(argv[1])), 0, 2.5, 100000);

    // std::cout << atoi(argv[1]) << std::endl;
    for (double i = 0; i <= 2.5; i += 0.01)
        std::cout << i << " " << tab(i) << std::endl;

    /***
    for (double i = 0; i <= 1.5; i += 0.01)
        std::cout << i << " " << MKB_RL(i, 2, 0.5) << std::endl;
    ***/
}

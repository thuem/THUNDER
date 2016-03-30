/*******************************************************************************
 * Author: Kunpeng Wang
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Timer.h"

void timing()
{
    static struct timeval last_tv;
    static bool flag = false;

    if (!flag)
    {
        gettimeofday(&last_tv, NULL);
        std::cout << "----------------------------" << std::endl
                  << "*Timing start!" << std::endl
                  << "\tsecond: " 
                  << last_tv.tv_sec << std::endl
                  << "\tmicosecond: "
                  << last_tv.tv_usec << std::endl
                  << "----------------------------" << std::endl;
        
        flag = true;
    }else {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        std::cout << "----------------------------" << std::endl
                  << "*Time consumed" << std::endl;
        if (tv.tv_usec - last_tv.tv_usec < 0)
        {
            std::cout << "\tsecond: "
                      << tv.tv_sec  - last_tv.tv_sec - 1 
                      << std::endl
                      << "\tmicosecond: "
                      << tv.tv_usec + 1000000 - last_tv.tv_usec
                      << std::endl;
        }else {
            std::cout << "\tsecond: "
                      << tv.tv_sec  - last_tv.tv_sec
                      << std::endl
                      << "\tmicosecond: "
                      << tv.tv_usec - last_tv.tv_usec
                      << std::endl;
        }
        std::cout << "----------------------------" << std::endl;
        
        last_tv.tv_sec  = tv.tv_sec;
        last_tv.tv_usec = tv.tv_usec;
    }
}
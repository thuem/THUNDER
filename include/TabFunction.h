/*******************************************************************************
 * Author: Kunpeng Wang, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef TAB_FUNCTION_H
#define TAB_FUNCTION_H

#include <cmath>
#include <functional>

#include "Macro.h"
#include "Functions.h"

using namespace std;

class TabFunction
{
	private:

        double* _tab = NULL;

        double _a = 0;
        double _b = 0;

        double _s = 0;

        int _n = 0;

	public:

        TabFunction();
        
        ~TabFunction();
        
        TabFunction(function<double(const double)> func,
                    const double a,
                    const double b,
                    const int n);

        void init(function<double(const double)> func,
                  const double a,
                  const double b,
                  const int n);

        double operator()(const double x) const;
};

#endif // TAB_FUNCTION_H

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

#include <boost/move/make_unique.hpp>
#include <boost/function.hpp>

#include "Macro.h"
#include "Functions.h"

using boost::function;

class TabFunction
{
	private:

        boost::movelib::unique_ptr<double[]> _tab;

        double _a;

        double _b;

        double _s;

        int _n;

	public:

        /**
         * default constructor
         */
        TabFunction();
        
        /*
         * default deconstructor
         */
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

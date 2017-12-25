//This header file is add by huabin
#include "huabin.h"
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

        boost::movelib::unique_ptr<RFLOAT[]> _tab;

        RFLOAT _a;

        RFLOAT _b;

        RFLOAT _s;

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
        
        TabFunction(function<RFLOAT(const RFLOAT)> func,
                    const RFLOAT a,
                    const RFLOAT b,
                    const int n);

        void init(function<RFLOAT(const RFLOAT)> func,
                  const RFLOAT a,
                  const RFLOAT b,
                  const int n);

        RFLOAT operator()(const RFLOAT x) const;
};

#endif // TAB_FUNCTION_H

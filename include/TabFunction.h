/*******************************************************************************
 * Author: Kunpeng Wang, Mingxu Hu, Siyuan Ren
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
#include "Precision.h"

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
        
        inline RFLOAT* getData() const{ return _tab.get(); } 

        inline RFLOAT getStep() const{ return _s; }
};

#endif // TAB_FUNCTION_H

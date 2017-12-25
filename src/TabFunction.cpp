//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Kunpeng Wang, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "TabFunction.h"

TabFunction::TabFunction() : _a(0), _b(0), _s(0), _n(0) {}

TabFunction::~TabFunction()
{
}

TabFunction::TabFunction(function<RFLOAT(const RFLOAT)> func,
                         const RFLOAT a,
                         const RFLOAT b,
                         const int n)
{
    init(func, a, b, n);
}

void TabFunction::init(function<RFLOAT(const RFLOAT)> func,
			           const RFLOAT a,
                       const RFLOAT b,
                       const int n)
{
    _a = a;
    _b = b;
    _n = n;

	_s = (_b - _a) / _n;

	_tab.reset(new RFLOAT[_n + 1]);

	for (int i = 0; i <= _n; i++)
        _tab[i] = func(_a + i * _s);
}

RFLOAT TabFunction::operator()(const RFLOAT x) const
{
    return _tab[AROUND((x - _a) / _s)];
}


#ifndef TABFUNCTION_H
#define TABFUNCTION_H

#include <math.h>

#include <functional>

class TabFunction
{
	private:
		double *_table;
		double _begin, _end, _step;


	public:
		TabFunction() {}
		~TabFunction() {}

		TabFunction(double(*pFunc)(const double x),
			        const double begin,
			        const double end,
			        const int N);

		double operator()(double x);
};

#endif
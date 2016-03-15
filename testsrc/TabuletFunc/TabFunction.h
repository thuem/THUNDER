
#ifndef TABFUNCTION_H
#define TABFUNCTION_H

#include <math.h>

#include <functional>

using namespace std;

using namespace placeholders;

class TabFunction
{
	private:
		double *_table;
		double _begin, _end, _step;


	public:
		TabFunction() {}
		~TabFunction() {}

		TabFunction(function<double(const double)> foo,
			        const double begin,
			        const double end,
			        const int N);

		double operator()(double x);
};

#endif

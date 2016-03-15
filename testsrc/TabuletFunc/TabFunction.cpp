#include "TabFunction.h"

// TabFunction::TabFunction(double(*pFunc)(const double x),
// 			        	 const double begin,
// 			        	 const double end,
// 			        	 const int N)
// {
// 	_table = new double[N];
// 	_begin = begin;
// 	_end = end;
// 	_step = (end - begin) / N;

// 	double var = begin;
// 	for (int i = 0; i < N; i++, var += _step)
// 		_table[i] = pFunc(var);
// }

TabFunction::TabFunction(std::function< double(double) >func,
			        	 const double begin,
			        	 const double end,
			        	 const int N)
{
	_table = new double[N];
	_begin = begin;
	_end = end;
	_step = (end - begin) / N;

	double var = begin;
	for (int i = 0; i < N; i++, var += _step)
		_table[i] = func(var);
}

double TabFunction::operator()(double x)
{
	if (x < _begin || x > _end)
		return 0.0; // exception handler

	double n = (x - _begin) / _step;
	int a = floor(n);
	int b = ceil(n);

	int N = (n - a < b - n ? a : b);

	return _table[N];
}
/***********************************************************************
 * FileName: TabFunction.cu
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "TabFunction.cuh"

namespace cuthunder {

HD_CALLABLE TabFunction::TabFunction(double a,
                                     double b,
 			                    	 double s,
 			                    	 double *tab,
 			                    	 int size)
{
	init(a, b, s, tab, size);
}

HD_CALLABLE void TabFunction::init(double a,
		                           double b,
		                    	   double s,
		                    	   double *tab,
		                    	   int size)
{
	_begin = a;
	_end = b;
	_step = s;

	_table = tab;
	_size = size;
}

HD_CALLABLE double TabFunction::operator()(const double x) const
{
	int index = (int)rint((x - _begin) / _step);

	if (index < 0)
		return _table[0];
	else if(index > _size)
		return _table[_size];
	else
		return _table[index];
}

} // end namespace cuthunder

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

HD_CALLABLE TabFunction::TabFunction(RFLOAT a,
                                     RFLOAT b,
 			                    	 RFLOAT s,
 			                    	 RFLOAT *tab,
 			                    	 int size)
{
	init(a, b, s, tab, size);
}

HD_CALLABLE void TabFunction::init(RFLOAT a,
		                           RFLOAT b,
		                    	   RFLOAT s,
		                    	   RFLOAT *tab,
		                    	   int size)
{
	_begin = a;
	_end = b;
	_step = s;

	_table = tab;
	_size = size;
}

HD_CALLABLE RFLOAT TabFunction::operator()(const RFLOAT x) const
{
#ifdef SINGLE_PRECISION
	int index = (int)rintf((x - _begin) / _step);
#else
	int index = (int)rint((x - _begin) / _step);
#endif    
	if (index < 0)
		return _table[0];
	else if(index > _size)
		return _table[_size];
	else
		return _table[index];
}

} // end namespace cuthunder

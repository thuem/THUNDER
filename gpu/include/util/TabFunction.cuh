/***********************************************************************
 * FileName: TabFunction.cuh
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef TABFUNCTION_CUH
#define TABFUNCTION_CUH

#include "Device.cuh"

namespace cuthunder {

/***
 * Todo: 
 *   Store table on constant memory on the device if the size
 * of memory permitting.
 ***/

class TabFunction
{
	public:

		HD_CALLABLE TabFunction() {}

		HD_CALLABLE ~TabFunction() {}

		HD_CALLABLE TabFunction(double a,
			                    double b,
			                    double s,
			                    double *tab,
			                    int size);

		HD_CALLABLE void init(double a,
		                      double b,
		                      double s,
		                      double *tab,
		                      int size);

        HD_CALLABLE int size() const { return _size; }

        HD_CALLABLE void devPtr(double *tab) { _table = tab; }

        HD_CALLABLE double* devPtr() const { return _table; }

		HD_CALLABLE double operator()(const double x) const;

	private:

		double* _table = NULL;
		int _size = 0;

        double _begin = 0;
        double _end = 0;
        double _step = 0;

};

} // end namespace cuthunder

#endif

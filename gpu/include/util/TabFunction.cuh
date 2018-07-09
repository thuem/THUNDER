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

#include "Config.h"
#include "Precision.h"

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

		HD_CALLABLE TabFunction(RFLOAT a,
			                    RFLOAT b,
			                    RFLOAT s,
			                    RFLOAT *tab,
			                    int size);

		HD_CALLABLE void init(RFLOAT a,
		                      RFLOAT b,
		                      RFLOAT s,
		                      RFLOAT *tab,
		                      int size);

        HD_CALLABLE int size() const { return _size; }

        HD_CALLABLE void devPtr(RFLOAT *tab) { _table = tab; }

        HD_CALLABLE RFLOAT* devPtr() const { return _table; }

		HD_CALLABLE RFLOAT operator()(const RFLOAT x) const;

	private:
		
        RFLOAT* _table = NULL;
		int _size = 0;

        RFLOAT _begin = 0;
        RFLOAT _end = 0;
        RFLOAT _step = 0;

};

} // end namespace cuthunder

#endif

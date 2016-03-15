#include <iostream>
#include <functional>

#include <gsl/gsl_sf_bessel.h>

#include "TabFunction.h"

using namespace std::placeholders;

// int func(int a, std::function< int(int, int) > b)
// {
//     return b(a, 1);
// }

// int c(int x, int y, int z)
// {
//     return x+y+z;
// }

int main(int argc, char const *argv[])
{
	// TabFunction bessel_j0(gsl_sf_bessel_j0,
	// 	                  0.0,
	// 	                  1,
	// 	                  100000);

	// auto gsl_bessel_I3_5 = std::bind(gsl_sf_bessel_Inu, 3.5, _2);
	TabFunction bessel_I3_5(std::bind(gsl_sf_bessel_Inu, 3.5, _2),
	                  		0.0,
	                  		1.0,
	                  		100000);

	for (double x = 0.00011; x < 0.5; x += 0.00254)
		std::cout << "x = " << x << " : " << bessel_I3_5(x) << std::endl;

	// int i = func( 10, std::bind( &c, _1, _2, some-value ) );

	return 0;
}


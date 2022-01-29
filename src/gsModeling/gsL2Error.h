/** @file gsL2Error.h

    @brief This is a helper file with a few functions that I should
    integrate elsewhere.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsAssembler/gsQuadRule.h>

namespace gismo
{

    template <class T>
    T evalExp(T u, T v)
    {
	return (2.0 / 3) * (math::exp(-1 * math::sqrt((10 * u - 3) * (10 * u - 3)
						      +
						      (10 * v - 3) * (10 * v - 3)))
			    +
			    math::exp(-1 * math::sqrt((10 * u + 3) * (10 * u + 3)
						      +
						      (10 * v + 3) * (10 * v + 3)))
			    );
    }

    template <class T>
    T L2distFromExp(const gsTensorBSpline<2, T>& spline, bool verbose = false)
    {
	gsOptionList legendreOpts;
	legendreOpts.addInt   ("quRule","Quadrature rule used (1) Gauss-Legendre; (2) Gauss-Lobatto; (3) Patch-Rule",gsQuadrature::GaussLegendre);
	legendreOpts.addReal("quA", "Number of quadrature points: quA*deg + quB", 2.0  );
	legendreOpts.addInt ("quB", "Number of quadrature points: quA*deg + quB", 3    );
	legendreOpts.addSwitch("overInt","Apply over-integration or not?",false);
	gsQuadRule<real_t>::uPtr legendre = gsQuadrature::getPtr(spline.basis(), legendreOpts);

	gsMatrix<T> points;
	gsVector<T> weights;
	gsMatrix<T> values;
	T result = 0;

	for (auto domIt = spline.basis().makeDomainIterator(); domIt->good(); domIt->next() )
	{
	    if(verbose)
	    {
		gsInfo<<"---------------------------------------------------------------------------\n";
		gsInfo  <<"Element with corners (lower) "
			<<domIt->lowerCorner().transpose()<<" and (higher) "
			<<domIt->upperCorner().transpose()<<" :\n";
	    }

	    // Gauss-Legendre rule (w/o over-integration)
	    legendre->mapTo(domIt->lowerCorner(), domIt->upperCorner(),
			    points, weights);

	    if(verbose)
		gsInfo << "The rule uses " << points.cols() << " points." << std::endl;

	    spline.eval_into(points, values);
	    for(index_t j=0; j<values.cols(); j++)
		result += weights(j) * math::pow(values(0, j) - evalExp(points(0, j), points(1, j)), 2);
	}
	return math::sqrt(result);
    }
    
} // namespace gismo
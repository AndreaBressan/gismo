/** @file gsNormL2.h

    @brief Computes the Semi H1 norm, needs for the parallel computing.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): A. Farahat
*/


#pragma once

namespace gismo
{


template <class T>
class gsG1ASVisitorResidualSeminormH1
{
public:

    gsG1ASVisitorResidualSeminormH1()
    {
        f2param = false;

    }

    void initialize(const std::vector<gsMultiBasis<>> * basis,
                    gsQuadRule<T> & rule,
                    unsigned      & evFlags) // replace with geoEval ?
    {
        // Setup Quadrature
        const unsigned d = basis->at(0).dim();
        gsVector<index_t> numQuadNodes( d );
        for (unsigned i = 0; i < d; ++i)
            numQuadNodes[i] = basis->at(0).degree(i) + 1;

        // Setup Quadrature
        rule = gsGaussRule<T>(numQuadNodes);// harmless slicing occurs here

        // Set Geometry evaluation flags
        evFlags = NEED_MEASURE|NEED_VALUE|NEED_GRAD_TRANSFORM;
    }

    // Evaluate on element.
    void evaluate(gsGeometryEvaluator<T> & geoEval,
                  const std::vector<gsSparseMatrix<>> * sol_sparse,
                  const std::vector<gsMultiBasis<>> * basis_vec,
                  std::vector<gsG1System<real_t>> & sys_vec,
                  gsMatrix<T>            & quNodes)
    {
        gsMatrix<unsigned> actives;
        gsMatrix<T> bGrads;

        basis_vec->at(0).basis(geoEval.id()).active_into(quNodes.col(0), actives);

        // Evaluate basis functions on element
        basis_vec->at(0).basis(geoEval.id()).deriv_into(quNodes,bGrads);

        f1ders.setZero(2,actives.rows());
        for (index_t i = 0; i < sol_sparse->at(0).rows(); i++)
            for (index_t j = 0; j < actives.rows(); j++)
                f1ders += sol_sparse->at(0).at(i,sys_vec.at(0).get_numBasisFunctions()[geoEval.id()] + actives.at(j)) * bGrads.block(2*j,0,2,f1ders.dim().second);

        // Evaluate second function
        geoEval.evaluateAt(quNodes);

        basis_vec->at(1).basis(geoEval.id()).active_into(quNodes.col(0), actives);

        // Evaluate basis functions on element
        basis_vec->at(1).basis(geoEval.id()).deriv_into(quNodes,bGrads);

        f2ders.setZero(2,actives.rows());
        for (index_t i = 0; i < sol_sparse->at(1).rows(); i++)
            for (index_t j = 0; j < actives.rows(); j++)
                f2ders += sol_sparse->at(1).at(i,sys_vec.at(1).get_numBasisFunctions()[geoEval.id()] + actives.at(j)) * bGrads.block(2*j,0,2,f2ders.dim().second);

        // Evaluate second function
        geoEval.evaluateAt(quNodes);

//        _func2.deriv_into( f2param ? quNodes : geoEval.values() , f2ders); // Not working and useless

    }


    // assemble on element
    inline T compute(gsDomainIterator<T>    & geo,
                     gsGeometryEvaluator<T> & geoEval,
                     gsVector<T> const      & quWeights,
                     T & accumulated)
    {
        T sum(0.0);
        for (index_t k = 0; k < quWeights.rows(); ++k) // loop over quadrature nodes
        {

            gsMatrix<T> Jk = geoEval.jacobian(k);
            gsMatrix<T> G = Jk.transpose() * Jk;
            gsMatrix<T> G_inv = G.cramerInverse();

            f1pders = f1ders.col(k); // Computed gradient

            f2pders =  f2ders.col(k); // Exact gradient

            const T weight = quWeights[k] * sqrt(G.determinant());

            sum += weight * (f1pders - f2pders).squaredNorm();

        }
        accumulated += sum;
        return sum;
    }


protected:

    gsMatrix<T> f1ders, f2ders;
    gsMatrix<T> f1pders, f2pders; // f2pders only needed if f2param = true

    bool f2param;

};






}







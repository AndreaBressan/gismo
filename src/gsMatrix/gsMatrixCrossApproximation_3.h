/** @file gsMatrixCrossApproximation_3.h

    @brief Greedy algorithm for approximating matrices with tensor-products.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsMatrix/gsVector.h>
#include <gsMatrix/gsMatrix.h>

#include <random>

namespace gismo
{

/** This is an algorithm similar to that described in Section 2.3 of
I. Georgieva, C. Hofreither: An algorithm for low-rank approximation
of bivariate functions using splines, JCAM 310, pp. 80 -- 91, 2017.

However, in addition to row pivoting there is also a possibility to
search for a maximum in the entire matrix. That is more precise but
probably much slower.
*/

template <class T>
class gsMatrixCrossApproximation_3
{
public:
    gsMatrixCrossApproximation_3(const gsMatrix<T>& matrix, index_t stopcrit=1)
	: m_Z(matrix),
	  m_mat(matrix),
	  m_ik(0), m_uNum(m_mat.cols()),
	  m_stopcrit(stopcrit)
    {
	index_t maxRank = std::min(m_mat.rows(), m_mat.cols());
	m_U.resize(m_mat.rows(), maxRank);
	m_V.resize(m_mat.cols(), maxRank);
	m_T.resize(maxRank, maxRank);
	
	m_U.setZero(); // for computing ptsApprox
	m_V.setZero(); // for computing ptsApprox
	m_T.setZero();
    }

    gsMatrixCrossApproximation_3(const gsMatrix<T>& matrix, index_t stopcrit, index_t maxIter)
	: m_Z(matrix),
	  m_mat(matrix),
	  m_U(m_mat.rows(), maxIter),
	  m_V(m_mat.cols(), maxIter),
	  m_T(maxIter, maxIter),
	  m_ik(0),
	  m_stopcrit(stopcrit)
    {
	m_U.setZero(); // for computing ptsApprox
	m_V.setZero(); // for computing ptsApprox
	m_T.setZero();
    }


    bool nextIteration(T& sigma, gsVector<T>& uVec, gsVector<T>& vVec, bool pivot, T zero=1e-32)
    {
	if(pivot)
	    return nextPivotingIteration(sigma, uVec, vVec, 0, zero);
	else
	    return nextFullIteration(sigma, uVec, vVec, zero);
    }

    void updateMatrix(const gsMatrix<T>& matrix)
    {
	if(m_mat.rows() != matrix.rows() || m_mat.cols() != matrix.cols())
	    gsWarn << "Warning: replacing a ("
		   << m_mat.rows()  << " x " << m_mat.cols() << ") matrix with a ("
		   << matrix.rows() << " x " << matrix.cols() << ") one." << std::endl;

	m_mat = matrix;
    }

    void getUsedI(std::set<index_t>& usedI) const
    {
	usedI = m_usedI;
    }

    void getUsedJ(std::set<index_t>& usedJ) const
    {
	usedJ = m_usedJ;
    }

    void compute(bool pivot, index_t maxIter, T zero = 1e-32);

    void getU(gsMatrix<T>& U) const
    {
	U = m_U;
    }

    void getV(gsMatrix<T>& V) const
    {
	V = m_V;
    }

    void getT(gsMatrix<T>& TT) const
    {
	TT = m_T;
    }

    void getRest(gsMatrix<T>& result) const
    {
	result = m_mat;
    }

protected:

    bool nextFullIteration(T& sigma, gsVector<T>& uVec, gsVector<T>& vVec, T zero=1e-15)
    {
	index_t i=0, j=0;
	if( !findAbsMax(i, j, zero))
	    return false;

	// Note to future self: the order in the tensor-product is
	// somewhat surprising, u is a column and v a row.
	sigma = 1.0 / m_mat(i, j);
	uVec  = m_mat.col(j);
	vVec  = m_mat.row(i);

	matrixUtils::addTensorProduct(m_mat, -1 * sigma, uVec, vVec);
	return true;
    }

    bool nextPivotingIteration(T& sigma, gsVector<T>& uVec, gsVector<T>& vVec, index_t shifts = 0, T zero=1e-32)
    {
	if(shifts > m_mat.rows())
	{
	    // Prevent a possible infinite loop.
	    //gsInfo << "shifting too much" << std::endl;
	    return false;
	}

	index_t jk = findAbsMaxRow(m_ik);
	if(math::abs(m_mat(m_ik, jk)) < zero)
	{
	    m_ik = (m_ik + 1) % m_mat.rows();
	    return nextPivotingIteration(sigma, uVec, vVec, shifts+1, zero);
	}
	else
	{
	    m_usedI.insert(m_ik);
	    m_usedJ.insert(  jk);

	    sigma = 1.0 / m_mat(m_ik, jk);
	    uVec  = m_mat.col(jk);
	    vVec  = m_mat.row(m_ik);

	    m_ik = findAbsMaxCol(jk);
	    matrixUtils::addTensorProduct(m_mat, -1 * sigma, uVec, vVec);
	    return true;//stopcrit();
	}
    }

    /*index_t findAbsMaxRow(index_t i) const
    {
	// Using a row instead of random-access to the entire matrix seems a little bit faster.
	index_t jk = 0;
	T max = 0;
	//gsVector<T> row = m_mat.row(i);
	for(index_t j=0; j<m_mat.cols(); j++)
	    //for(index_t j=0; j<m_uNum; j++)
	{
	    //T curr = math::abs(row(i));
	    T curr = math::abs(m_mat(i, j));
	    if(curr > max)
	    {
		max = curr;
		jk = j;
	    }
	}
	return jk;
	}*/

    index_t findAbsMaxRow(index_t i) const
    {
	index_t minIndex, maxIndex;
	gsVector<T> row = m_mat.row(i);
	row.minCoeff(&minIndex);
	row.maxCoeff(&maxIndex);
	if(math::abs(row(minIndex)) > math::abs(row(maxIndex)))
	    return minIndex;
	else
	    return maxIndex;
    }	

    index_t findAbsMaxCol(index_t j) const
    {
	index_t ik = 0;
	T max = 0;
	for(index_t i=0; i<m_mat.rows(); i++)
	{
	    T curr = math::abs(m_mat(i, j));
	    if(curr > max)
	    {
		max = curr;
		ik = i;
	    }
	}
	return ik;
    }

    /// Finds the element with the highest absolute value
    /// and returns false iff the max is equal to 0.
    bool findAbsMax(index_t& i_res, index_t& j_res, T zero=1e-15) const
    {
	i_res = 0;
	j_res = 0;
	T max = 0;
	for(index_t i=0; i<m_mat.rows(); i++)
	{
	    for(index_t j=0; j<m_mat.cols(); j++)
	    {
		T curr = math::abs(m_mat(i,j));
		if(curr > max)
		{
		    max = curr;
		    i_res = i;
		    j_res = j;
		}
	    }
	}

	return (math::abs(max) > zero);
    }

    T getMaxEntry() const
    {
	T maxEntry = 0; // AKA M_max
	for(auto it=m_usedI.begin(); it!=m_usedI.end(); ++it)
	    for(auto jt=m_usedJ.begin(); jt!=m_usedJ.end(); ++jt)
		maxEntry = std::max(maxEntry, math::abs(m_Z(*it, *jt)));

	return maxEntry;
    }

    /// Stopping criterium from K. Frederix, M. Van Barel: Solving a
    /// large dense linear system by adaptive cross approximation,
    /// JCAM, 2010.
    bool stopcrit_frederix(index_t numChecks, T tol) const
    {
	bool errTooBig = false;
	T maxEntry = getMaxEntry();
	gsMatrix<T> ptsApprox = m_U * m_T * m_V.transpose();
	for(index_t ell = 1; ell < numChecks; ell++)
	{
	    std::random_device dev;
	    std::mt19937 rng(dev());
	    std::uniform_int_distribution<std::mt19937::result_type> dist(0, m_mat.rows() - 1);

	    index_t i_ell = dist(rng);
	    while(m_usedI.find(i_ell) != m_usedI.end())
		i_ell = dist(rng);

	    index_t j_ell = dist(rng);
	    while(m_usedJ.find(j_ell) != m_usedJ.end())
		j_ell = dist(rng);

	    if((math::abs(m_mat(i_ell, j_ell) - ptsApprox(i_ell, j_ell)) / maxEntry) > tol)
		errTooBig = true;
	}
	return !errTooBig;
    }

    bool stopcrit() const
    {
	switch(m_stopcrit)
	{
	case 0:
	    return false;
	case 1:
	    return stopcrit_frederix(10, 1e-6);
	default:
	    gsWarn << "Unknown stopping criterium " << m_stopcrit << "." << std::endl;
	    return true;
	}	
    }

protected: // elements

    gsMatrix<T> m_Z, m_mat, m_U, m_V, m_T;

    std::set<index_t> m_usedI, m_usedJ;

    index_t m_ik, m_uNum;

    index_t m_stopcrit;    
};

template <class T>
void gsMatrixCrossApproximation_3<T>::compute(bool pivot, index_t maxIter, T zero)
{
    T sigma;
    gsVector<T> uVec, vVec;
    index_t maxRank = std::min(m_mat.rows(), m_mat.cols());
    for(index_t i=0; i<maxRank; i++)
    {
	// Test for the stopping criterium.
	// When finishing, shrink the matrices.
	if(!nextIteration(sigma, uVec, vVec, pivot, zero) || i >= maxIter || stopcrit())
	{
	    gsInfo << "Finishing at rank " << i << "." << std::endl;
	    m_U.conservativeResize(gsEigen::NoChange, i);
	    m_V.conservativeResize(gsEigen::NoChange, i);
	    m_T.conservativeResize(i, i);
	    return;
	}
	else
        {
	    // Compute the next iteration.
	    m_U.col(i) = uVec;
	    m_V.col(i) = vVec;
	    m_T(i, i)  = sigma;
	}
    }
    gsInfo << "Finishing at the full rank (" << maxRank << ")." << std::endl;
}

} // namespace gismo

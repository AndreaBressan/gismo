/** @file gsMatrixUtils.h

    @brief A few util functions for matrix manipulations.

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#pragma once

#include <gsMatrix/gsVector.h>
#include <gsMatrix/gsMatrix.h>

namespace gismo
{
    namespace matrixUtils
    {
	template <class T>
	gsMatrix<T> tensorProduct(const gsVector<T>& u, const gsVector<T>& v)
	{
	    gsMatrix<T> result(u.size(), v.size());
	    for(index_t i=0; i<u.size(); i++)
		for(index_t j=0; j<v.size(); j++)
		    result(i, j) = u(i) * v(j);

	    return result;
	}

	template <class T>
	void addTensorProduct(gsMatrix<T>& result, T sigma, const gsVector<T>& u, const gsVector<T>& v)
	{
	    GISMO_ASSERT(u.size() == result.rows(), "row mismatch in addTensorProduct");
	    GISMO_ASSERT(v.size() == result.cols(), "col mismatch in addTensorProduct");

	    // for(index_t i=0; i<result.rows(); i++)
	    // 	for(index_t j=0; j<result.cols(); j++)
	    // 	    result(i, j) += sigma * u(i) * v(j);

	    // The following seems to be quite a bit faster than my for-loops.
	    result += (sigma * u) * v.transpose();
	    //result = result + sigma * u * v.transpose();

	    // for(index_t i=0; i<result.rows(); i++)
	    // 	result.row(i) += sigma * u(i) * v;
	}

	template <class T>
	T det2x2(T c00, T c0n, T cm0, T cmn)
	{
	    // gsMatrix<T> mat(2, 2);
	    // mat << c00, c0n,
	    // 	cm0, cmn;
	    // return mat.det();
	    return c00 * cmn - c0n * cm0;
	}

	template <class T>
	gsMatrix<T> diag(const gsVector<T>& vec)
	{
	    index_t size = vec.size();
	    gsMatrix<T> result(size, size);
	    result.setZero();
	    for(index_t i=0; i<size; i++)
		result(i, i) = vec(i);
	    return result;
	}

	template <class T>
	gsSparseMatrix<T> sparseDiag(const gsVector<T>& vec)
	{
	    index_t size = vec.size();
	    gsSparseMatrix<T> result(size, size);
	    for(index_t i=0; i<size; i++)
		result(i, i) = vec(i);
	    return result;   
	}

	/// Vector of size \a size with each element equal to 1.
	template <class T>
	gsVector<T> oneVector(index_t size)
	{
	    gsVector<T> result(size);
	    for(index_t i=0; i<size; i++)
		result(i) = T(1);
	    return result;
	}

	template <class T>
	gsMatrix<T> identity(index_t size)
	{
	    gsMatrix<T> result(size, size);
	    result.setZero();
	    for(index_t i=0; i<size; i++)
		result(i, i) = T(1);
	    return result;
	}
	
	template <class T>
	T maxNorm(const gsMatrix<T>& mat)
	{
	    T res = 0;

	    for(index_t i=0; i<mat.rows(); i++)
		for(index_t j=0; j<mat.cols(); j++)
		    res = std::max(res, math::abs(mat(i, j)));

	    return res;
	}

	/// Appends \a vec as a new col to \a mat.
	template <class T>
	void appendCol(gsMatrix<T>& mat, const gsVector<T> vec)
	{
	    if(mat.rows() != vec.size())
		gsWarn << "wrong sizes " << mat.rows() << " != " << vec.size() << std::endl;

	    index_t oldCols = mat.cols();
	    mat.conservativeResize(gsEigen::NoChange, oldCols + 1);
	    mat.col(oldCols) = vec;
	}

	template <class T>
	void appendDiag(gsMatrix<T>& mat, T num)
	{
	    index_t oldRows = mat.rows();
	    index_t oldCols = mat.cols();
	    index_t newDiag = std::min(oldRows, oldCols);

	    mat.conservativeResize(oldRows + 1, oldCols + 1);
	    for(index_t i=0; i<mat.rows(); i++)
		mat(i, oldCols) = 0;

	    for(index_t j=0; j<mat.cols(); j++)
		mat(oldRows, j) = 0;

	    mat(newDiag, newDiag) = num;
	}

	template <class T>
	void printSize(const std::string& name, const gsMatrix<T>& mat)
	{
	    gsInfo << name << ": (" << mat.rows() << "x" << mat.cols() << ")" << std::endl;
	}
	template <class T>
	void print(const std::string& name, const gsMatrix<T>& mat)
	{
	    printSize(name, mat);
	    gsInfo << mat << std::endl;
	}

    } // namespace matrixUtils

} // namespace gismo

/** @file gsLowRankFitting.hpp

    @brief Bodies of functions from gsLowRankFitting.h

    This file is part of the G+Smo library.
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): D. Mokris
*/

#include <gsModeling/gsLowRankFitting.h>

namespace gismo
{

template <class T>
index_t gsLowRankFitting<T>::partitionParam(gsMatrix<T>& uPar, gsMatrix<T>& vPar) const
{
    index_t uNum = 0;
    T prevMax = -1 * std::numeric_limits<T>::max();
    for(index_t i=0; this->m_param_values(0, i) > prevMax; i++)
    {
	uNum++;
	prevMax = this->m_param_values(0, i);
    }
    index_t vNum = this->m_param_values.cols() / uNum;
    // gsInfo << "uNum: " << uNum << std::endl;
    // gsInfo << "vNum: " << vNum << std::endl;

    // Separate the u- and v-parameters.
    uPar.resize(1, uNum);
    vPar.resize(1, vNum);
    for(index_t i=0; i<uNum; i++)
    	uPar(0, i) = this->m_param_values(0, i);

    for(index_t j=0; j<vNum; j++)
	vPar(0, j) = this->m_param_values(1, j * uNum);

    // gsInfo << "uPar:\n" << uPar << std::endl;
    // gsInfo << "vPar:\n" << vPar << std::endl;
    return uNum;
}

template <class T>
gsMatrix<T> gsLowRankFitting<T>::convertToMN(index_t rows) const
{
    gsMatrix<T> result(rows, this->m_points.rows() / rows);

    for(index_t i=0; i<this->m_points.rows(); i++)
	result(i%rows, i/rows) = this->m_points(i, 0);

    return result;
}

template <class T>
gsSparseMatrix<T> gsLowRankFitting<T>::convertToSparseMN(index_t rows) const
{
    gsSparseMatrix<T> result(rows, this->m_points.rows() / rows);

    for(index_t i=0; i<this->m_points.rows(); i++)
	result(i%rows, i/rows) = this->m_points(i, 0);

    return result;
}

template <class T>
gsMatrix<T> gsLowRankFitting<T>::convertBack(const gsMatrix<T>& points) const
{
    gsMatrix<T> result(1, points.rows() * points.cols());

    for(index_t i=0; i<points.rows(); i++)
	for(index_t j=0; j<points.cols(); j++)
	    result(points.rows() * j + i) = points(i, j);

    return result;	    
}

template <class T>
gsMatrix<T> gsLowRankFitting<T>::getErrorsMN(size_t rows) const
{
    // Challenge: Can we merge this and convertToMN into a single function?

    gsMatrix<T> val_i;
    this->m_result->eval_into(this->m_param_values, val_i);

    gsMatrix<T> result(rows, this->m_pointErrors.size() / rows);
    for(size_t i=0; i<this->m_pointErrors.size(); i++)
	result(i%rows, i/rows) = this->m_points(i, 0) - val_i(0, i); // math::abs?
    // Signed distance, works in 1D only.
	    //this->m_pointErrors[i];

    return result;
}

template <class T>
void gsLowRankFitting<T>::initPQ(const gsMatrix<T>& uWeights, const gsMatrix<T>& vWeights)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsBSplineBasis<T> uBasis = tpBasis->component(0);
    gsBSplineBasis<T> vBasis = tpBasis->component(1);

    gsMatrix<T> uPar, vPar;
    //index_t uNum = partitionParam(uPar, vPar);
    partitionParam(uPar, vPar);

    gsSparseMatrix<T> Xs = uBasis.collocationMatrix(uPar);
    gsSparseMatrix<T> Ys = vBasis.collocationMatrix(vPar);
    m_X = gsMatrix<T>(Xs.transpose());
    m_Y = gsMatrix<T>(Ys.transpose());

    GISMO_ASSERT(m_X.cols() == uWeights.rows(), "Cannot compute X * U due to size mismatch.");
    GISMO_ASSERT(uWeights.cols() == m_X.cols(), "Cannot compute U * X^T due to size mismatch.");

    GISMO_ASSERT(m_Y.cols() == vWeights.rows(), "Cannot compute Y * V due to size mismatch.");
    GISMO_ASSERT(vWeights.cols() == m_Y.cols(), "Cannot compute V * Y^T due to size mismatch.");

    m_P = (m_X * uWeights * m_X.transpose()).inverse() * m_X * uWeights;
    m_Q = (m_Y * vWeights * m_Y.transpose()).inverse() * m_Y * vWeights;
}

template <class T>
void gsLowRankFitting<T>::computeSVD(index_t sample, const std::string& filename)
{
    clearErrors();

    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsBSplineBasis<T> uBasis = tpBasis->component(0);
    gsBSplineBasis<T> vBasis = tpBasis->component(1);

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<T> ptsMN = convertToMN(uNum);

    // 1. Perform an SVD on it.
    gsSvd<T> pointSVD(ptsMN);
    //pointSVD.sanityCheck(ptsMN);
    //gsInfo << "Rank: " << pointSVD.rank() << std::endl;

    // 2. Iterate.
    std::vector<T> LIErr, L2Err, l2Err, dofs;
    for(index_t r=0; r<pointSVD.rank(); r++)
    {
	// 2.0. Fit u with the u-basis.
	gsFitting<T> uFitting(uPar, pointSVD.u(r).transpose(), uBasis);
	uFitting.compute();
	// A note to future self: Forgetting about the transpose lead to preconditioners failing.

	// 2.1. Fit v with the v-basis.
	gsFitting<T> vFitting(vPar, pointSVD.v(r).transpose(), vBasis);
	vFitting.compute();

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<T> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<T> vCoefs = vFitting.result()->coefs().col(0);
	coefs += pointSVD.s(r) * matrixUtils::tensorProduct(uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	T maxErr = this->maxPointError();
	//gsInfo << "err SVD: " << maxErr << std::endl;
	m_maxErr.push_back(maxErr);
	// optional and a bit costly:
	//m_L2Err.push_back(L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample));
	dofs.push_back((r+1) * uCoefs.size() * vCoefs.size());

	m_l2Err.push_back(this->get_l2Error());
	m_decompErr.push_back(pointSVD.l2decompErr(r+1));
    }

    // 3. Make a convergence graph.
    // gsWriteGnuplot(LIErr, filename + "svd_max.dat");
    // gsWriteGnuplot(L2Err, filename + "svd_L2.dat");
    // gsWriteGnuplot(dofs, L2Err, "svdL2.dat");
    // gsWriteGnuplot(l2Err, filename + "-svd_l2.dat");
}

template <class T>
void gsLowRankFitting<T>::computeCross(bool pivot,
				       index_t maxIter,
				       index_t sample)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsMatrix<T> coefs(tpBasis->component(0).size(),
		      tpBasis->component(1).size());
    coefs.setZero();

    index_t uNum = math::sqrt(this->m_param_values.cols());
    gsMatrix<T> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma;
    gsVector<T> uVec, vVec;
    T prevErr = 1e8;
    bool scnm = true; // Stopping Criterium Not Met
    for(index_t i=0; i<maxIter && crossApp.nextIteration(sigma, uVec, vVec, pivot) && scnm; i++)
    {
	gsMatrix<T> uMat(uVec.size(), 1);
	gsMatrix<T> vMat(vVec.size(), 1);
	uMat.col(0) = uVec;
	vMat.col(0) = vVec;

	coefs = coefs + sigma * (m_P * uMat * (m_Q * vMat).transpose());
	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	T maxErr = this->maxPointError();
	T l2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample);
	gsInfo << "max err piv: " << maxErr << ", l2 err piv: " << l2Err << std::endl;
	scnm = (l2Err < prevErr);
	if(!scnm)
	    gsInfo << "Finishing at rank " << i+1 << "." << std::endl;
	prevErr = l2Err;
	m_maxErr.push_back(maxErr);
	m_L2Err.push_back(l2Err);
    }
    //gsWriteParaview(*this->m_result, "result");
}

template <class T>
void gsLowRankFitting<T>::computeCross_3(bool pivot,
					 index_t maxIter,
					 index_t sample)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsMatrix<T> coefs(tpBasis->component(0).size(),
		      tpBasis->component(1).size());
    coefs.setZero();

    index_t uNum = math::sqrt(this->m_param_values.cols());
    gsMatrix<T> ptsMN = convertToMN(uNum);
    gsMatrixCrossApproximation_3<T> crossApp(ptsMN);
    crossApp.compute(pivot, maxIter);
    gsMatrix<T> uMat, vMat, tMat;
    crossApp.getU(uMat);
    crossApp.getV(vMat);
    crossApp.getT(tMat);

    coefs = m_P * uMat * tMat * (m_Q * vMat).transpose();

    delete this->m_result;
    this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
    this->computeErrors();
    T maxErr = this->maxPointError();
    T l2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample);
    gsInfo << "max err piv: " << maxErr << ", l2 err piv: " << l2Err << std::endl;

    for(int i=0; i<4; i++) // Refine; TODO: check tol.
    {
	tpBasis->uniformRefine();
	// Recompute P, Q (TODO: weights).
	initPQ(matrixUtils::identity<T>(uNum),
	       matrixUtils::identity<T>(uNum));
	coefs = m_P * uMat * tMat * (m_Q * vMat).transpose();
	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	T maxErr = this->maxPointError();
	T l2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample);
	gsInfo << "(R) max err piv: " << maxErr << ", l2 err piv: " << l2Err << std::endl;
	m_maxErr.push_back(maxErr);
	m_L2Err.push_back(l2Err);
    }
}

template <class T>
void gsLowRankFitting<T>::computeRes()
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsBSplineBasis<T> uBasis = tpBasis->component(0);
    gsBSplineBasis<T> vBasis = tpBasis->component(1);

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    // 0. Convert points to an m x n matrix.
    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);
    gsMatrix<T> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma = 0;
    gsVector<T> uVec, vVec;
    //index_t iter = 0;
    for (index_t rank=0; rank<ptsMN.rows(); rank++)
    {
	crossApp.nextIteration(sigma, uVec, vVec, false);

	//gsInfo << "iter: " << iter++ << std::endl;
	// 2.0. Fit u with the u-basis.
	// gsInfo << "uPar:\n" << uPar << std::endl;
	// gsInfo << "uPts:\n" << uVec.transpose() << std::endl;
	gsFitting<T> uFitting(uPar, uVec.transpose(), uBasis);
	uFitting.compute();
	
	// 2.1. Fit v with the v-basis.
	// gsInfo << "vPar:\n" << vPar << std::endl;
	// gsInfo << "vPts:\n" << vVec.transpose() << std::endl;
	gsFitting<T> vFitting(vPar, vVec.transpose(), vBasis);
	vFitting.compute();

	// 2.3. Add the coefficients to the running tensor-product B-spline.
	gsVector<T> uCoefs = uFitting.result()->coefs().col(0);
	gsVector<T> vCoefs = vFitting.result()->coefs().col(0);
	matrixUtils::addTensorProduct(coefs, sigma, uCoefs, vCoefs);

	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	//gsInfo << this->maxPointError() << std::endl;

	crossApp.updateMatrix(getErrorsMN(uNum));
    }

    // 3. Make a convergence graph.
}

template <class T>
void gsLowRankFitting<T>::computeFull(const gsVector<T>& uWeights, const gsVector<T>& vWeights)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsBSplineBasis<T> uBasis = tpBasis->component(0);
    gsBSplineBasis<T> vBasis = tpBasis->component(1);

    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);

    // Note that X and Y here are the same way as in the paper, unlike the older implementation.
    gsSparseMatrix<T> Xs = uBasis.collocationMatrix(uPar);
    gsSparseMatrix<T> Ys = vBasis.collocationMatrix(vPar);
    gsSparseMatrix<T> uWs = matrixUtils::sparseDiag(uWeights);
    gsSparseMatrix<T> vWs = matrixUtils::sparseDiag(vWeights);

    gsSparseMatrix<T> Z = convertToSparseMN(uNum);
    gsSparseMatrix<T> lhs1 = Xs.transpose() * uWs * Xs;
    lhs1.makeCompressed();
    gsSparseMatrix<T> lhs2 = Ys.transpose() * vWs * Ys;
    lhs2.makeCompressed();

    // Saving the matrices beforehand leads to a speed up of an order of magnitude.
    gsMatrix<T> rhs1 = Xs.transpose() * uWs * Z;
    typename gsEigen::SparseLU<typename gsSparseMatrix<T>::Base> solver1;
    solver1.analyzePattern(lhs1);
    solver1.factorize(lhs1);

    gsMatrix<T> D = solver1.solve(rhs1);
    typename gsEigen::SparseLU<typename gsSparseMatrix<T>::Base> solver2;
    solver2.analyzePattern(lhs2);
    solver2.factorize(lhs2);

    gsMatrix<T> rhs2 = Ys.transpose() * (D.transpose());
    gsMatrix<T> CT = solver2.solve(rhs2);

    //if(printErr)
    //{
	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(CT.transpose()).transpose())).release();
	this->computeErrors();
	gsInfo << "compute full, max err: " << this->maxPointError() << std::endl;
	gsWriteParaview(*this->m_result, "result");
	//}
}

template <class T>
T gsLowRankFitting<T>::methodB(bool printErr)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsBSplineBasis<T> uBasis = tpBasis->component(0);
    gsBSplineBasis<T> vBasis = tpBasis->component(1);

    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);

    // Note that X and Y here are the same way as in the paper, unlike the older implementation.
    gsSparseMatrix<T> Xs = uBasis.collocationMatrix(uPar);
    gsSparseMatrix<T> Ys = vBasis.collocationMatrix(vPar);

    gsSparseMatrix<T> Z = convertToSparseMN(uNum);

    gsStopwatch timef1, timef2, times1, times2;

    gsSparseMatrix<T> lhs1 = Xs.transpose() * Xs;
    lhs1.makeCompressed();
    gsSparseMatrix<T> lhs2 = Ys.transpose() * Ys;
    lhs2.makeCompressed();
    gsMatrix<T> rhs1 = Xs.transpose() * Z;

    timef1.restart();
    typename gsEigen::SparseLU<typename gsSparseMatrix<T>::Base> solver1;
    solver1.analyzePattern(lhs1);
    solver1.factorize(lhs1);
    timef1.stop();

    times1.restart();
    gsMatrix<T> D = solver1.solve(rhs1);
    times1.stop();

    timef2.restart();
    typename gsEigen::SparseLU<typename gsSparseMatrix<T>::Base> solver2;
    solver2.analyzePattern(lhs2);
    solver2.factorize(lhs2);
    timef2.stop();
    gsInfo << "factorization: " << timef1.elapsed() + timef2.elapsed() << std::endl;

    gsMatrix<T> rhs2 = Ys.transpose() * (D.transpose());
    times2.restart();
    gsMatrix<T> CT = solver2.solve(rhs2);
    times2.stop();

    T solutionTime = timef1.elapsed() + timef2.elapsed() + times1.elapsed() + times2.elapsed();
    gsInfo << "solution: " << solutionTime << std::endl;

    if(printErr)
    {
	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(CT.transpose()).transpose())).release();
	this->computeErrors();
	gsInfo << "method B: " << this->maxPointError() << std::endl;
	gsWriteParaview(*this->m_result, "result");
    }
    return solutionTime;
}

template <class T>
T gsLowRankFitting<T>::methodC(bool printErr, index_t maxIter, bool pivot)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsBSplineBasis<T> uBasis = tpBasis->component(0);
    gsBSplineBasis<T> vBasis = tpBasis->component(1);

    gsMatrix<T> uPar, vPar;
    index_t uNum = partitionParam(uPar, vPar);

    // X and Y are now the same as in the paper.
    gsSparseMatrix<T> Xs = uBasis.collocationMatrix(uPar);
    gsSparseMatrix<T> Ys = vBasis.collocationMatrix(vPar);

    gsMatrix<T> Z = convertToMN(uNum);
    gsMatrixCrossApproximation_3<T> crossApp(Z, 0);
    // Using the older implementation leads to trouble in cost_12.
    //gsMatrixCrossApproximation<T> crossApp(Z);

    gsStopwatch time, timef, times, timec;

    gsSparseMatrix<T> lhs1 = Xs.transpose() * Xs;
    lhs1.makeCompressed();
    gsSparseMatrix<T> lhs2 = Ys.transpose() * Ys;
    lhs2.makeCompressed();
    time.restart();
    timef.restart();
    typename gsEigen::SparseLU<typename gsSparseMatrix<T>::Base> solver1;
    solver1.analyzePattern(lhs1);
    solver1.factorize(lhs1);
    // typename gsSparseSolver<T>::BiCGSTABILUT solver1(lhs1);
    // typename gsSparseSolver<T>::BiCGSTABILUT solver2(lhs2);

    typename gsEigen::SparseLU<typename gsSparseMatrix<T>::Base> solver2;
    solver2.analyzePattern(lhs2);
    solver2.factorize(lhs2);
    timef.stop();
    gsInfo << "factorization: " << timef.elapsed() << std::endl;

    // Using the original version instead of _3 seems a bit faster.
    // crossApp.compute(true, maxIter);
    // gsMatrix<T> uMat, vMat, tMat;
    // crossApp.getU(uMat);
    // crossApp.getV(vMat);
    // crossApp.getT(tMat);
    gsMatrix<T> uMat(uNum, maxIter), vMat(uNum, maxIter), tMat(maxIter, maxIter);
    gsVector<T> uVec, vVec;
    //T sigma;
    tMat.setZero();

    std::vector<gsVector<T>> uVecs(maxIter), vVecs(maxIter);
    std::vector<T> sigmas(maxIter);

    m_decompTime = timef.elapsed();
    T solutionTime = 0;
    gsMatrix<T> dMat(uBasis.size(), 0), eMat(vBasis.size(), 0);
    for(index_t i=0; i<maxIter; i++)
    {
	timec.restart();
	crossApp.nextIteration(sigmas[i], uVecs[i], vVecs[i], pivot);
	timec.stop();
	m_decompTime += timec.elapsed();

	tMat(i, i)  = sigmas[i];

	gsMatrix<T> rhs1 = Xs.transpose() * uVecs[i];
	gsMatrix<T> rhs2 = Ys.transpose() * vVecs[i];

	times.restart();
	gsVector<T> dVec = solver1.solve(rhs1);
	gsVector<T> eVec = solver2.solve(rhs2);
	times.stop();
	solutionTime += times.elapsed();

	matrixUtils::appendCol<T>(dMat, dVec);
	matrixUtils::appendCol<T>(eMat, eVec);
    }
    gsInfo << "cross approximation: " << m_decompTime << std::endl;

    gsInfo << "solution: " << solutionTime << std::endl;
    time.stop();

    if(printErr)
    {
	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(dMat * tMat * eMat.transpose()).transpose())).release();
	this->computeErrors();
	gsInfo << "method C: " << this->maxPointError() << std::endl;
	gsWriteParaview(*this->m_result, "result");
    }
    //return time.elapsed();
    return solutionTime;
}

template <class T>
void gsLowRankFitting<T>::CR2I_old(const gsMatrix<T>& bott,
				   const gsMatrix<T>& left,
				   const gsMatrix<T>& rght,
				   const gsMatrix<T>& topp) const
{
    gsMatrix<T> res(25, 2);

    for(index_t k=0; k<2; k++)
    {
	gsMatrix<T> c(5, 5);

	// Prepare the boundary.
	for(index_t i=0; i<c.rows(); i++)
	{
	    c(i, 0) = left(i, k);
	    c(i, 4) = rght(i, k);
	}
	for(index_t j=0; j<c.cols(); j++)
	{
	    c(0, j) = bott(j, k);
	    c(4, j) = topp(j, k);
	}

	T delta = matrixUtils::det2x2(c(0, 0), c(0, 4),
				      c(4, 0), c(4, 4));
	if(delta == 0)
	    GISMO_ERROR("Infeasible configuration.");

	for(index_t j=1; j<c.cols() - 1; j++)
	{
	    T lambda = (1.0/delta) * matrixUtils::det2x2(c(0, j), c(0, 4),
							 c(4, j), c(4, 4));

	    T rho    = (1.0/delta) * matrixUtils::det2x2(c(0, 0), c(0, j),
							 c(4, 0), c(4, j));
	    
	    for(index_t i=1; i<c.rows() - 1; i++)
		c(i, j) = lambda * c(i, 0) + rho * c(i, 4);
	}

	// Transcribe from c to res(k).
	for(index_t i=0; i<c.rows(); i++)
	    for(index_t j=0; j<c.cols(); j++)
		res(5 * i + j, k) = c(i, j);
    }

    gsKnotVector<T> KV(0.0, 1.0, 0, 5);
    gsTensorBSpline<2, T> param(KV, KV, res);
    gsWriteParaview(param, "param", 1000, false, true);

    // gsBSpline<T> fBott(0.0, 1.0, 0, 4, bott);
    // gsBSpline<T> fLeft(0.0, 1.0, 0, 4, left);
    // gsBSpline<T> fRght(0.0, 1.0, 0, 4, rght);
    // gsBSpline<T> fTopp(0.0, 1.0, 0, 4, topp);
    // gsWriteParaview(fBott, "bott");
    // gsWriteParaview(fLeft, "left");
    // gsWriteParaview(fRght, "rght");
    // gsWriteParaview(fTopp, "topp");
}

// TODO next time:
// - CR2I_old disagrees with the old example: is it CR2I or AR5I there?
// - Implement CR2I_new.
// - Compare the two.
template <class T>
void gsLowRankFitting<T>::CR2I_new(const gsMatrix<T>& bott,
				   const gsMatrix<T>& left,
				   const gsMatrix<T>& rght,
				   const gsMatrix<T>& topp) const
{
    gsKnotVector<T> KV(0.0, 1.0, 0, 5);
    gsBSplineBasis<T> uBasis(KV);
    gsBSplineBasis<T> vBasis(KV);

    gsBSpline<T> bSpline(uBasis, bott), tSpline(uBasis, topp);
    gsBSpline<T> lSpline(vBasis, left), rSpline(vBasis, rght);

    gsMatrix<T> uPar(1, 5), vPar(1, 5);
    uPar << 0, 0.2, 0.4, 0.6, 0.8;
    vPar << 0, 0.2, 0.4, 0.6, 0.8;

    gsMatrix<T> coefs(uBasis.size(), vBasis.size());
    coefs.setZero();

    gsSparseMatrix<T> Xs = uBasis.collocationMatrix(uPar);
    gsSparseMatrix<T> Ys = vBasis.collocationMatrix(vPar);

    gsMatrix<T> X(Xs.transpose());
    gsMatrix<T> Y(Ys.transpose());

    gsMatrix<T> uLeast = (X * X.transpose()).inverse() * X;
    gsMatrix<T> vLeast = (Y * Y.transpose()).inverse() * Y;

    gsMatrix<T> res(25, 2);

    // TODO: The following uses CPs but we need to assemble values in uPar and vPar.
    for(index_t k=0; k<2; k++)
    {
	gsMatrix<T> U(5, 2);
	gsMatrix<T> TT(2, 2);
	gsMatrix<T> V(5, 2);
	TT.setZero();

	U.col(0) = bott.col(k);
	U.col(1) = topp.col(k) - bott.col(k);

	TT(0, 0) = 1.0 / bott(0, k);
	TT(1, 1) = 1.0 / (topp(4, k) - TT(0, 0));

	V.col(0) = left.col(k);
	V.col(1) = rght.col(k);

	coefs = uLeast * U * TT * V.transpose() * vLeast.transpose();

	for(index_t i=0; i<5; i++)
	    for(index_t j=0; j<5; j++)
		res(5 * i + j, k) = coefs(i, j);
    }

    gsTensorBSpline<2, T> result(KV, KV, res);
    gsWriteParaview(result, "result_new", 1000, false, true);
}

template <class T>
void gsLowRankFitting<T>::computeCrossWithRef(bool pivot,
					      index_t maxIter,
					      index_t sample)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);
    gsMatrix<T> coefs(tpBasis->component(0).size(),
		      tpBasis->component(1).size());
    coefs.setZero();

    index_t uNum = math::sqrt(this->m_param_values.cols());
    gsMatrix<T> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation<T> crossApp(ptsMN);
    T sigma;
    gsVector<T> uVec, vVec;
    T prevErr = 1e8;
    bool scnm = true; // Stopping Criterium Not Met

    gsMatrix<T> uMat(uNum, maxIter);
    gsMatrix<T> vMat(uNum, maxIter);
    gsMatrix<T> tMat(maxIter, maxIter);
    tMat.setZero();

    std::vector<real_t> L2errors;
    std::vector<index_t> iter;

    for(index_t i=0; i<maxIter && crossApp.nextIteration(sigma, uVec, vVec, pivot) && scnm; i++)
    {
	uMat.col(i) = uVec;
	vMat.col(i) = vVec;
	tMat(i, i) = sigma;

	coefs = coefs + sigma * (m_P * uMat.col(i) * (m_Q * vMat.col(i)).transpose());
	delete this->m_result;
	this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
	this->computeErrors();
	T maxErr = this->maxPointError();
	T l2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample);
	L2errors.push_back(l2Err);
	iter.push_back(i);
	gsInfo << "(L) max err piv: " << maxErr << ", l2 err piv: " << l2Err << std::endl;
	scnm = (i < 6 || (l2Err <= 0.999 * prevErr));
	gsWriteParaview(*this->result(), "justbefore", 10000, false, true);
	if(!scnm)
	{
	    // Refine.
	    gsInfo << "Refining at rank " << i+1 << std::endl;
	    tpBasis->uniformRefine(); // _withCoefs does not make it any better. (-;

	    // Recompute P, Q (TODO: weights).
	    initPQ(matrixUtils::identity<T>(uNum),
		   matrixUtils::identity<T>(uNum));

	    // Update coefs (matrix multiplication would be more efficient).
	    // coefs.resize(math::sqrt(this->m_basis->size()),
	    // 		 math::sqrt(this->m_basis->size()));
	    // coefs.setZero();
	    coefs = m_P * uMat * tMat * (m_Q * vMat).transpose();
	    // for(index_t j=0; j<=i; j++)
	    // {
	    // 	coefs += tMat(j, j) * (m_P * uMat.col(j) * (m_Q * vMat.col(j)).transpose());

		delete this->m_result;
		this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
		this->computeErrors();
		T maxErr = this->maxPointError();
		T l2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), sample);
		gsInfo << "(R) max err piv: " << maxErr << ", l2 err piv: " << l2Err << std::endl;
		gsWriteParaview(*this->result(), "fitting_r", 10000, false, true);
		L2errors.push_back(l2Err);
		iter.push_back(i);
	    // }

	    prevErr = l2Err;
	    scnm = true;
	}
	else
	{
	    // Continue
	    prevErr = l2Err;
	    m_maxErr.push_back(maxErr);
	    m_L2Err.push_back(l2Err);
	}
    }
    //gsWriteParaview(*this->m_result, "result");
    gsWriteGnuplot(iter, L2errors, "example-7.dat");
}

template <class T>
void gsLowRankFitting<T>::computeCrossWithRefAndStop(T tol, bool pivot)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);

    gsMatrix<T> coefs;

    index_t uNum = math::sqrt(this->m_param_values.cols());
    gsMatrix<T> ptsMN = convertToMN(uNum);

    gsMatrixCrossApproximation_3<T> crossApp(ptsMN, 0);
    //T sigma;
    gsVector<T> uVec, vVec;
    T l2Err(1);

    gsMatrix<T> uMat(uNum, 0);
    gsMatrix<T> vMat(uNum, 0);
    gsMatrix<T> tMat(0, 0);

    std::vector<real_t> L2errors, decompErrors;
    std::vector<index_t> iter;

    gsMatrix<T> residua(uNum, uNum);
    residua.setZero();
    bool scnm = true; // Stopping Criteria Not Met

    crossApp.compute(true, uNum); // 200
    gsMatrix<T> uTmp, vTmp, tTmp;
    crossApp.getU(uTmp);
    crossApp.getT(tTmp);
    crossApp.getV(vTmp);

    for(index_t i=0; i<tTmp.cols() && scnm && l2Err > tol; i++)
    {
	// TODO: We might be required to do more than one refinement at a time!
	// T sigma;
	// gsVector<T> uVec, vVec;
	// if(!crossApp.nextIteration(sigma, uVec, vVec, pivot))
	//     scnm = false;

	// TODO next time: compute gives the correct decomposition,
	// going through nextIteration does not. Why?

	matrixUtils::appendCol<T>(uMat, uTmp.col(i));
	matrixUtils::appendCol<T>(vMat, vTmp.col(i));
	matrixUtils::appendDiag<T>(tMat, tTmp(i, i));

	// matrixUtils::appendCol<T>(uMat, uVec);
	// matrixUtils::appendCol<T>(vMat, vVec);
	// matrixUtils::appendDiag<T>(tMat, sigma);

	gsMatrix<T> zMat = uMat * tMat * vMat.transpose();
	gsMatrix<T> rest = zMat - ptsMN;
	decompErrors.push_back(rest.norm());

	coefs = m_P * zMat * m_Q.transpose();

	// TODO next times: coefs seem to be good enough (they work
	// with convert back) but values are far off. Why? Also, for
	// the interpolation it is vice versa (values good,
	// convertBack bad).

	gsMatrix<T> values = m_X.transpose() * coefs * m_Y;
	gsMatrix<T> pointwiseErr = values - ptsMN;
	l2Err = pointwiseErr.norm();
	gsInfo << "iteration " << i << ", sqrt(DOF) " << coefs.rows() << ", ";
	gsInfo << "l2 err: " << l2Err << std::endl;

	// Save error.
	L2errors.push_back(l2Err);
	iter.push_back(i);

	// Figure out, whether to proceed with ACA or refine.
	//crossApp.getRest(rest);
	// gsMatrix<T> b_i = uMat.col(i) * tMat(i, i) * vMat.col(i).transpose();
	// gsMatrix<T> residuum = m_X.transpose() * m_P * b_i * m_Q.transpose() * m_Y - b_i;
	gsMatrix<T> residua = values - uMat * tMat * vMat.transpose();
	// gsInfo << "residua: " << residua.norm() << std::endl
	//        << "rest:    " << rest.norm() << std::endl;
	//residua += residuum;

	if((residua.norm() - rest.norm() < tol))
	    gsInfo << "proceeding" << std::endl; 
	else if(2 * coefs.rows() - 3 > uNum)
	    gsInfo << "cannot refine any further" << std::endl;
	else
	{
	    gsInfo << "refining at rank " << i << std::endl;
	    tpBasis->uniformRefine();
	    // Recompute P, Q (TODO: weights).
	    initPQ(matrixUtils::identity<T>(uNum),
		   matrixUtils::identity<T>(uNum));
	}
    }
    delete this->m_result;
    this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
    this->computeErrors();
    gsInfo << "old err: " << this->get_l2Error()  << std::endl;

    // This is the L^2-error using a quadrature rule (!).
    //T gsl2Err = L2Error(*static_cast<gsTensorBSpline<2, T>*>(this->result()), 6);
    //gsInfo << "gs l2 err " << gsl2Err << std::endl;

    gsWriteGnuplot(iter, L2errors, "example-8.dat");
    gsWriteGnuplot(iter, decompErrors, "example-8-decomp.dat");
}

template <class T>
int gsLowRankFitting<T>::computeCrossWithStop(T epsAccept, T epsAbort, bool pivot, bool verbose)
{
    auto tpBasis = static_cast<gsTensorBSplineBasis<2, T>*>(this->m_basis);

    gsMatrix<T> coefs;

    index_t uNum = m_uNpts;
    index_t vNum = this->m_param_values.cols() / uNum;
    gsMatrix<T> ptsMN = convertToMN(uNum);
    gsEigen::ColPivHouseholderQR<gsMatrix<T>> qrDecomp(ptsMN);

    if(verbose)
	gsInfo << "data rank: " << qrDecomp.rank() << std::endl;

    gsMatrixCrossApproximation_3<T> crossApp(ptsMN, 0);
    gsVector<T> uVec, vVec;
    T l2Err(1), maxErr(1), currErr(1), L2Err(-1), sigma;

    gsMatrix<T> uMat(uNum, 0);
    gsMatrix<T> vMat(vNum, 0);
    gsMatrix<T> tMat(0, 0);

    gsMatrix<T> residua(uNum, vNum);
    residua.setZero();

    // gsSvd<T> pointSVD(ptsMN);
    // gsInfo << "SVD rank: " << pointSVD.rank() << std::endl;

    gsStopwatch time;
    T sumTimes(0);
    clearErrors();
    index_t result = -1;
    for(index_t i=0; i<uNum; i++)
    //for(index_t i=0; i<pointSVD.rank(); i++)
    {
	time.restart();
	crossApp.nextIteration(sigma, uVec, vVec, pivot, m_zero);
	time.stop();
	sumTimes += time.elapsed();

	m_rank = i+1;
	matrixUtils::appendCol<T>(uMat, uVec);
	matrixUtils::appendCol<T>(vMat, vVec);
	matrixUtils::appendDiag<T>(tMat, sigma);

	// matrixUtils::appendCol<T>(uMat, pointSVD.u(i));
	// matrixUtils::appendCol<T>(vMat, pointSVD.v(i));
	// matrixUtils::appendDiag<T>(tMat, pointSVD.s(i));

	// TODO: This is unfortunate notation. In the paper I call ptsMN Z.
	gsMatrix<T> zMat = uMat * tMat * vMat.transpose();
	T rest = (zMat - ptsMN).norm();
	m_decompErr.push_back(rest);

	coefs = m_P * zMat * m_Q.transpose();

	gsMatrix<T> values = m_X.transpose() * coefs * m_Y;
	gsMatrix<T> pointwiseErr = ptsMN - values;
	l2Err = pointwiseErr.norm();
	maxErr = std::max(math::abs(pointwiseErr.maxCoeff()),
			  math::abs(pointwiseErr.minCoeff()));
	T eps_j = (zMat - values).norm();

	if(verbose)
	{
	    // gsInfo << "iteration " << i << ", sqrt(DOF) " << coefs.rows() << ", ";
	    gsInfo << "l2 err: "  << l2Err
		   << ", rest: "  << rest
		   << ", eps_j: " << eps_j
		   << ", est: "   << eps_j - rest << std::endl;
	}

	// Save the l2- and max-error ...
	m_l2Err.push_back(l2Err);
	m_maxErr.push_back(maxErr);

	// ... and the L2-error as well, if required.
	if(m_sample != -1)
	{
	    this->m_result=tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
	    m_L2Err.push_back(L2Error(*(this->m_result), m_sample, T(1), 0, false));
	}

	switch(m_errType)
	{
	case gsErrType::l2 :
	    currErr = l2Err;
	    break;
	case gsErrType::max :
	    currErr = maxErr;
	    break;
	case gsErrType::L2 :
	    currErr = L2Err;
	    break;
	default:
	    gsWarn << "Undefined error type." << std::endl;
	    break;
	}

	// Figure out what to do.
	if(currErr < epsAccept)
	{
	    gsInfo << "Finished after iteration " << i+1 << std::endl;
	    result = 0; // success
	    break;
	}
	//else if(l2Err - rest > 0.9 * l2Err) // Seems to work decently!
	else if(eps_j - rest > epsAbort)
	{
	    gsInfo << "Aborting after iteration " << i+1 << std::endl;
	    result = 1; // cannot converge
	    break;
	}
	else if(i == uNum - 1)
	//else if(i == pointSVD.rank())
	{
	    gsInfo << "Max iter reached." << std::endl;
	    result = 2; // maxIterReached
	    break;
	}
    }
    delete this->m_result;
    this->m_result = tpBasis->makeGeometry(give(convertBack(coefs).transpose())).release();
    this->computeErrors();
    gsInfo << "l2E: " << l2Err << std::endl;
    gsInfo << "Decomposition time: " << sumTimes << std::endl;

    return result;
}

} // namespace gismo

#include <gismo.h>
#include <gsModeling/gsLowRankFitting.h>
#include <gsModeling/gsL2Error.h>
#include <gsIO/gsWriteGnuplot.h>
#include <gsMatrix/gsMatrixCrossApproximation_3.h>

#include <random>

using namespace gismo;

gsMatrix<real_t> testMatrix(index_t example = 0)
{
    if(example == 0)
    {
	gsMatrix<real_t> coefs(3, 3);
	coefs.setZero();
	coefs(0, 0) = 1;
	coefs(0, 1) = 2;
	coefs(0, 2) = 3;
	coefs(1, 0) = 3;
	coefs(1, 1) = 2;
	coefs(1, 2) = 1;
	coefs(2, 0) = 1;

	return coefs;
    }
    else if(example == 1)
    {
	gsMatrix<real_t> coefs(4, 4);
	coefs.setZero();
	coefs(0, 0) = 2;
	coefs(0, 1) = 3;
	coefs(1, 0) = -1;
	coefs(1, 1) = 3;
	coefs(1, 2) = 4;
	coefs(2, 2) = 1;
	coefs(2, 3) = -1;
	coefs(3, 3) = 1;

	return coefs;
    }
    else if(example == 2)
    {
	gsMatrix<real_t> coefs(5,5);
	coefs.setZero();
	coefs(0, 4) = 1;
	coefs(1, 3) = -1;
	coefs(2, 2) = 1;
	coefs(3, 1) = -1;
	coefs(4, 0) = 1;

	coefs(0, 0) = 2;
	coefs(1, 1) = 3;
	coefs(3, 3) = -2;
	coefs(4, 4) = -3;

	coefs(1, 4) = 5;
	coefs(2, 3) = 7;
	coefs(3, 4) = -1;
	coefs(4, 3) = 9;

	return coefs;
    }
    else if(example == 3)
    {
	gsMatrix<real_t> coefs(4, 4);
	coefs.setZero();

	coefs(0, 1) = 1;
	coefs(0, 2) = 1;
	coefs(0, 3) = 1;

	coefs(1, 2) = 1;
	coefs(1, 3) = 1;

	coefs(2, 3) = 1;

	return coefs;
    }
    else if(example == 4)
    {
	gsMatrix<real_t> coefs(5, 5);
	coefs.setZero();

	coefs(0, 1) = 0.666667;
	coefs(0, 2) = 1;
	coefs(0, 3) = 1;
	coefs(0, 4) = 1;

	coefs(1, 2) = 1;
	coefs(1, 3) = 1;
	coefs(1, 4) = 1;

	coefs(2, 3) = 1;
	coefs(2, 4) = 1;

	coefs(3, 4) = 0.666667;

	return coefs;
    }
    else if(example == 5)
    {
	gsMatrix<real_t> coefs(4, 4);
	coefs.setZero();
	coefs(0, 0) = -2;
	coefs(0, 1) = -3;
	coefs(1, 0) = 1;
	coefs(1, 1) = -3;
	coefs(1, 2) = -4;
	coefs(2, 2) = -1;
	coefs(2, 3) = 1;
	coefs(3, 3) = -1;

	return coefs;
    }
    else
    {
	index_t numU = 5;
	index_t numV = 5;
	gsMatrix<real_t> coefs(numU, numV);
	for(index_t i = 0; i < numU; i++)
	{
	    for(index_t j = 0; j < numV; j++)
	    {
		real_t u = i / (numU - 1.0);
		real_t v = j / (numV - 1.0);
		coefs(i, j) = (math::exp(math::sqrt(u * u + v * v))) / 4.0;
	    }
	}
	return coefs;
    }
}

bool checkSvd()
{
    gsMatrix<real_t> coefs = testMatrix();
    gsSvd<real_t> svd(coefs);
    return svd.sanityCheck(coefs);
}

void checkCrossApp(index_t example, bool pivot)
{
    gsMatrix<real_t> coefs = testMatrix(example);
    gsMatrix<real_t> check(coefs.rows(), coefs.cols());
    check.setZero();

    gsInfo << "Target:\n" << coefs << std::endl;

    gsMatrixCrossApproximation<real_t> crossApp(coefs);

    real_t sigma;
    gsVector<real_t> uVec, vVec;
    // for(index_t i=0; i<coefs.rows(); i++)
    // {
    // 	crossApp.nextIteration(sigma, uVec, vVec, pivot);

    // 	matrixUtils::addTensorProduct(check, sigma, uVec, vVec);
    // 	gsInfo << "Iteration " << i << ":\n" 
    // 	       << "check:\n" << check << std::endl;
    // }

    index_t i=0;
    while(crossApp.nextIteration(sigma, uVec, vVec, pivot))
    {
	i++;
	matrixUtils::addTensorProduct(check, sigma, uVec, vVec);
	gsInfo << "Iteration " << i << ":\n" 
	       << "check:\n" << check << std::endl;
    }
	
}

void checkCrossAppMat(index_t example, bool pivot)
{
    gsMatrix<real_t> coefs = testMatrix(example);
    gsMatrix<real_t> check(coefs.rows(), coefs.cols());
    check.setZero();

    gsMatrix<real_t> U(coefs.rows(), coefs.cols());
    gsMatrix<real_t> V(coefs.rows(), coefs.cols());
    gsMatrix<real_t> T(coefs.cols(), coefs.cols());
    T.setZero();

    gsInfo << "Target:\n" << coefs << std::endl;

    gsMatrixCrossApproximation_3<real_t> crossApp(coefs);

    real_t sigma;
    gsVector<real_t> uVec, vVec;
    for(index_t i=0; i<coefs.rows(); i++)
    {
	crossApp.nextIteration(sigma, uVec, vVec, pivot);
	U.col(i) = uVec;
	V.col(i) = vVec;
	T(i, i) = sigma;
    }

    gsInfo << "U:\n" << U << "\nT:\n" << T << "\nV:\n" << V << std::endl;

    gsInfo << "UTV^T:\n" << U * T * V.transpose() << std::endl;
	
}

template <class T>
T l2Error(const gsGeometry<T>* result,
	  const gsMatrix<T>& params,
	  const gsMatrix<T>& points)
{
    std::vector<T> errors;
    gsMatrix<T> results;
    result->eval_into(params, results);

    T err = 0;
    for(index_t row = 0; row != points.rows(); row++)
    {
	for(index_t col = 0; col != points.cols(); col++)
	{
	    err += math::pow(points(row, col) - results(row, col), 2);
	}
    }

    return sqrt(err);
}

template <class T>
void sampleData(const gsVector<T>& uPar, const gsVector<T>& vPar,
		gsMatrix<T>& params, gsMatrix<T>& points,
		const gsGeometry<T>& geometry)
{
    index_t uNum = uPar.size();
    index_t vNum = vPar.size();
    index_t numSamples = uNum * vNum;

    params.resize(2, numSamples);

    for(index_t i=0; i<uNum; i++)
    {
	for(index_t j=0; j<vNum; j++)
	{
	    index_t glob = j * uNum + i;
	    T u = uPar(i);
	    T v = vPar(j);

	    params(0, glob) = u;
	    params(1, glob) = v;

	}
    }

    // gsInfo << "params:\n" << params << std::endl;
    // gsInfo << "points:\n" << points << std::endl;

    geometry.eval_into(params, points);

}

template <class T>
void sampleData(const gsVector<T>& uPar, const gsVector<T>& vPar,
		gsMatrix<T>& params, gsMatrix<T>& points, index_t sample)
{
    index_t uNum = uPar.size();
    index_t vNum = vPar.size();
    index_t numSamples = uNum * vNum;

    params.resize(2, numSamples);
    points.resize(1, numSamples);

    // gsInfo << "uPar:\n";
    // for(index_t i=0; i<uPar.size(); i++)
    // 	gsInfo << uPar(i) << " ";
    // gsInfo << std::endl;

    for(index_t i=0; i<uNum; i++)
    {
	for(index_t j=0; j<vNum; j++)
	{
	    index_t glob = j * vNum + i;
	    real_t u = uPar(i);
	    real_t v = vPar(j);

	    params(0, glob) = u;
	    params(1, glob) = v;

	    points(0, glob) = evalSample(u, v, sample);
	    //gsInfo << points(0, glob) << std::endl;
	}
    }
    // gsInfo << "params:\n" << params << std::endl;
    // gsInfo << "points:\n" << points << std::endl;
}

template <class T>
gsVector<T> sampleUniform(index_t size, T tMin, T tMax)
{
    gsVector<T> res(size);
    for(index_t i=0; i<size; i++)
    {
	real_t tLoc = i / (size - 1.0);
	real_t tAct = tMin * (1 - tLoc) + tMax * tLoc;
	res(i) = tAct;
    }
    return res;
}

template <class T>
void sampleData(index_t uNum, index_t vNum, gsMatrix<T>& params, gsMatrix<T>& points,
		index_t sample,
		T uMin = 0, T vMin = 0, T uMax = 1, T vMax = 1)
{
    gsVector<T> uPar = sampleUniform(uNum, uMin, uMax);
    gsVector<T> vPar = sampleUniform(vNum, vMin, vMax);

    sampleData(uPar, vPar, params, points, sample);
}

template <class T>
void sampleData(index_t numSide, gsMatrix<T>& params, gsMatrix<T>& points,
		index_t sample, T tMin = 0, T tMax = 1)
{
    sampleData(numSide, numSide, params, points, sample, tMin, tMin, tMax, tMax);
}

template <class T>
void sampleDataGre(const gsKnotVector<T>& knotsU, const gsKnotVector<T>& knotsV,
		   gsMatrix<T>& params, gsMatrix<T>& points, index_t sample)
{
    gsMatrix<T> greU, greV;
    knotsU.greville_into(greU);
    knotsV.greville_into(greV);

    gsVector<T> uPar = greU.row(0);
    gsVector<T> vPar = greV.row(0);

    sampleData(uPar, vPar, params, points, sample);
}

template <class T>
void sampleDataGre(index_t numSide, gsMatrix<T>& params, gsMatrix<T>& points,
		   index_t sample, T minT = 0, T maxT = 1, index_t deg = 3)
{
    gsKnotVector<T> kv(minT, maxT, numSide - deg - 1, deg + 1);
    sampleDataGre(kv, kv, params, points, sample);
}

template <class T>
void sampleDataGre(index_t uNum, index_t vNum,
		   gsMatrix<T>& params, gsMatrix<T>& points,
		   const gsGeometry<T>& geometry,
		   T uMin = 0, T uMax = 1, T vMin = 0, T vMax = 1, index_t deg = 3)
{
    gsKnotVector<T> uKnots(uMin, uMax, uNum - deg - 1, deg + 1);
    gsKnotVector<T> vKnots(vMin, vMax, vNum - deg - 1, deg + 1);
    
    gsMatrix<T> uGre, vGre;
    uKnots.greville_into(uGre);
    vKnots.greville_into(vGre);

    gsVector<T> uPar = uGre.row(0);
    gsVector<T> vPar = vGre.row(0);

    sampleData(uPar, vPar, params, points, geometry);
}

template <class T>
void sampleParamsAndWeights(const gsBSplineBasis<T>& basis,
			    gsVector<T>& params,
			    gsVector<T>& weights,
			    T quA = 1.0,
			    index_t quB = 1,
			    bool verbose = false)
{
    gsOptionList legendreOpts;
    legendreOpts.addInt   ("quRule","Quadrature rule used (1) Gauss-Legendre; (2) Gauss-Lobatto; (3) Patch-Rule",gsQuadrature::GaussLegendre);
    legendreOpts.addReal("quA", "Number of quadrature points: quA*deg + quB", quA);
    legendreOpts.addInt ("quB", "Number of quadrature points: quA*deg + quB", quB);
    legendreOpts.addSwitch("overInt","Apply over-integration or not?",false);
    gsQuadRule<real_t>::uPtr legendre = gsQuadrature::getPtr(basis, legendreOpts);

    params.clear();
    weights.clear();

    gsMatrix<T> locParams, globParams(1, 0);
    gsVector<T> locWeights;
    index_t start;

    for (auto domIt = basis.makeDomainIterator(); domIt->good(); domIt->next() )
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
			locParams, locWeights);

	if (verbose)
	{
	    gsInfo  << "* \t Gauss-Legendre\n"
		    << "- points:\n"  << locParams              <<"\n"
		    << "- weights:\n" << locWeights.transpose() <<"\n";
	}

	// Append locParams to globParams.
	start = globParams.cols();
	globParams.conservativeResize(gsEigen::NoChange, start + locParams.cols());
	globParams.block(0, start, globParams.rows(), locParams.cols()) = locParams;

	// Append locWeights to weights.
	weights.conservativeResize(weights.size() + locWeights.size());
	for(index_t j=0; j<locWeights.size(); j++)
	    weights(start + j) = locWeights(j);
    }

    params = globParams.row(0);
    //gsInfo << "params:\n" << params << std::endl;
}

template <class T>
void sampleDataGauss(const gsTensorBSplineBasis<2, T>& basis,
		     gsMatrix<T>& params,
		     gsMatrix<T>& points,
		     gsVector<T>& uWeights,
		     gsVector<T>& vWeights,
		     index_t sample,
		     T       quA = 1.0,
		     index_t quB = 1)
{
    bool verbose = false;

    const gsBSplineBasis<T> uBasis = basis.component(0);
    const gsBSplineBasis<T> vBasis = basis.component(1);

    gsVector<T> uParams, vParams;

    sampleParamsAndWeights(uBasis, uParams, uWeights, quA, quB, verbose);
    sampleParamsAndWeights(vBasis, vParams, vWeights, quA, quB, verbose);

    GISMO_ASSERT(uParams.size() == uWeights.size(), "Different number of params and weights in u-direction.");
    GISMO_ASSERT(vParams.size() == vWeights.size(), "Different number of params and weights in v-direction.");

    // FIXME:
    GISMO_ASSERT(uParams.size() == vParams.size(), "Problem not symmetric. This is possible in theory but not in the current implementation.");

    // for(index_t i=0; i<uWeights.size(); i++)
    // {
    // 	uWeights(i) = math::sqrt(uWeights(i));
    // 	vWeights(i) = math::sqrt(vWeights(i));
    // }

    sampleData(uParams, vParams, params, points, sample);
}

template <class T>
void sampleDataMeshInterpolated(gsMatrix<T>& params,
				gsMatrix<T>& points,
				const std::string& meshFile,
				index_t uNum,
				index_t vNum,
				index_t sample)
{
    gsFileData<> fd(meshFile);
    gsMesh<real_t>::uPtr mm = fd.getFirst<gsMesh<real_t>>();

    params.resize(2, uNum * vNum);
    points.resize(1, uNum * vNum);

    for(index_t i=0; i<uNum; i++)
    {
	for(index_t j=0; j<vNum; j++)
	{
	    index_t glob = j * vNum + i;
	    real_t u = real_t(i+1) / (uNum + 1);
	    real_t v = real_t(j+1) / (vNum + 1);

	    params(0, glob) = u;
	    params(1, glob) = v;

	    gsVertex<real_t> vert(u, v, 0);
	    const auto faces = mm->faces();
	    for(auto fi=faces.begin(); fi != faces.end(); ++fi)
	    {
		if((*fi)->inside(&vert))
		{
		    points(0, glob) = evalSample(u, v, sample);
		}
	    }
	}
    }
}

template <class T>
void sampleDataMeshVertices(gsMatrix<T>& params,
			    gsMatrix<T>& points,
			    const std::string& meshFile,
			    index_t sample)
{
    gsFileData<> fd(meshFile);
    gsMesh<real_t>::uPtr mm = fd.getFirst<gsMesh<real_t>>();

    const auto vertices = mm->vertices();
    params.resize(2, vertices.size());
    points.resize(1, vertices.size());

    index_t glob = 0;
    for(auto vi = vertices.begin(); vi != vertices.end(); vi++, glob++)
    {
	real_t u = (*vi)->x();
	real_t v = (*vi)->y();

	params(0, glob) = u;
	params(1, glob) = v;
	points(0, glob) = evalSample(u, v, sample);
    }
}		

template <class T>
void sampleDataUniform(const gsGeometry<T>& geometry,
		       index_t uNum, index_t vNum,
		       gsMatrix<T>& params, gsMatrix<T>& points)
{
    gsVector<T> uPar = sampleUniform(uNum, T(0), T(1));
    gsVector<T> vPar = sampleUniform(vNum, T(0), T(1));

    sampleData(uPar, vPar, params, points, geometry);
}

real_t stdFit(const gsMatrix<real_t>& params,
	      const gsMatrix<real_t>& points,
	      index_t numKnots,
	      index_t deg,
	      index_t sample,
	      real_t minU = 0.0,
	      real_t maxU = 1.0)
{
    gsKnotVector<real_t> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);
    
    gsFitting<real_t> fitting(params, points, basis);
    fitting.compute();
    fitting.computeErrors();
    //real_t L2Err = L2Error(*static_cast<gsTensorBSpline<2, real_t>*>(fitting.result()), sample);
    // gsInfo << "max err std: " << fitting.maxPointError() << ",\n"
    // 	   << "L2 err std: "  << L2Err << ",\n"
    // 	   << "l2 err std: "  << fitting.get_l2Error()
    // 	   << std::endl;
    //gsWriteParaview(*fitting.result(), "fitting", 10000, false, true);
    // gsInfo << "just checking:\n";
    // printErrors(fitting.result(), params, points);
    //return L2Err;
    return fitting.get_l2Error();
    //return fitting.getSolutionTime();
}

real_t stdFit(const gsMatrix<real_t>& params,
	      const gsMatrix<real_t>& points,
	      index_t uNumKnots,
	      index_t vNumKnots,
	      index_t deg,
	      index_t sample,
	      real_t uMin = 0.0,
	      real_t uMax = 1.0,
	      real_t vMin = 0.0,
	      real_t vMax = 1.0)
{
    gsKnotVector<real_t> uKnots(uMin, uMax, uNumKnots, deg+1);
    gsKnotVector<real_t> vKnots(vMin, vMax, vNumKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(uKnots, vKnots);
    
    gsFitting<real_t> fitting(params, points, basis);
    fitting.compute();
    fitting.computeErrors();
    //gsWriteParaview(*fitting.result(), "fitting", 10000, false, true);
    return fitting.get_l2Error();
    //return fitting.maxPointError();
}

void lowSVDFit(const gsMatrix<real_t>& params,
	       const gsMatrix<real_t>& points,
	       index_t numKnots,
	       index_t deg,
	       index_t sample,
	       index_t maxIter, // unused now
	       const std::string& filename,
	       real_t minU = 0.0,
	       real_t maxU = 1.0)
{
    gsKnotVector<> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsInfo << "SVD fitting:\n";
    gsLowRankFitting<real_t> fitting(params, points, basis);
    fitting.computeSVD(sample, filename);

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void lowCrossAppFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg,
		    index_t sample,
		    index_t maxIter,
		    const std::string& filename,
		    bool pivot,
		    real_t minU = 0.0,
		    real_t maxU = 1.0)
{
    gsKnotVector<> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << "CrossApp fitting";
    if(pivot)
	gsInfo << " with pivoting";
    gsInfo << ":\n";
    fitting.computeCross(pivot, maxIter, sample);
    if(pivot)
	fitting.exportl2Err(filename + "piv_L2.dat");
    else
	fitting.exportl2Err(filename + "full_L2.dat");

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void lowCrossResFit(const gsMatrix<real_t>& params,
		    const gsMatrix<real_t>& points,
		    index_t numKnots,
		    index_t deg,
		    real_t minU = 0.0,
		    real_t maxU = 1.0)
{
    gsKnotVector<> knots(minU, maxU, numKnots, deg+1);
    gsTensorBSplineBasis<2> basis(knots, knots);

    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << "CrossApp residual fitting:\n";
    fitting.computeRes();

    //gsWriteParaview(*fitting.result(), "low-rank", 10000, false, true);

    // gsFileData<real_t> fd;
    // fd << *fitting.result();
    // fd.dump("low-rank");
}

void param()
{
    gsFileData<> fd("example-2-rank2.xml");
    gsTensorBSpline<2> bspline;
    fd.getId<gsTensorBSpline<2>>(0, bspline);
    gsVector<real_t> shift(2);
    shift << -0.5, -0.5;
    bspline.translate(shift);
    //bspline.rotate(5 * EIGEN_PI / 14);
    gsWriteParaview(bspline, "bspline", 1000, false, true);

    gsBSpline<> bBott, bLeft, bRght;
    bspline.slice(1, 0.0, bBott);
    bspline.slice(0, 0.0, bLeft);
    bspline.slice(0, 1.0, bRght);

    gsMatrix<> cBott = bBott.coefs();
    gsMatrix<> cLeft = bLeft.coefs();
    gsMatrix<> cRght_0 = bRght.coefs();

    gsInfo << "slice:\n" << cRght_0 << std::endl;
    // Get-around, since slice gives wrong results for par = 1.0,
    // cf. https://github.com/gismo/gismo/issues/504
    gsMatrix<> cRght(5, 2), cTopp(5, 2);
    gsMatrix<> coefs = bspline.coefs();
    for(index_t i=0; i<5; i++)
    {
	for(index_t j=0; j<2; j++)
	{
	    cRght(i, j) = coefs(5 * i + 4, j);
	    cTopp(i, j) = coefs(20 + i, j);
	}
    }
    gsInfo << "manual:\n" << cRght << std::endl;
    //gsInfo << bspline.coefs().rows() << " x " << bspline.coefs().cols() << std::endl;

    // Compatibility check:
    // gsInfo << cBott.row(0) << std::endl << cLeft.row(0) << std::endl;
    // gsInfo << cTopp.row(4) << std::endl << cRght.row(4) << std::endl;

    // Figuring out the axial shift (do it on the original and change the sign!).
    //gsInfo << 0.25 * (cBott.row(0) + cBott.row(4) + cTopp.row(0) + cTopp.row(4)) << std::endl;

    gsLowRankFitting<real_t> fitting;
    fitting.CR2I_old(cBott, cLeft, cRght, cTopp);
    fitting.CR2I_new(cBott, cLeft, cRght, cTopp);
}

void development()
{
    //checkSvd();
    gsMatrix<real_t> params, points;
    real_t minT = -1.0; // -1 leads to a confusion index_t / real_t.
    index_t sample = 4;
    //real_t minT = 0;
    //sampleData(10, params, points, 5, minT);
    sampleDataGre(10, params, points, sample, minT, 1.0, 2);
    //sampleDataGre(50, params, points, 6, minT, 1.0, 2);
    // Experience: for examples 0 and 1 (rank 1 and 2, respectively),
    // we obtain the same precision as the standard fit after rank
    // iterations. Cool! Can we prove this to be true in general?

    index_t numKnots = 0;
    index_t deg = 2;
    index_t maxIter = 5;
    std::string filename = "old";
    stdFit(        params, points, numKnots, deg, sample, minT);
    lowSVDFit(     params, points, numKnots, deg, sample, maxIter, filename, minT);
    lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, false, minT);
    lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, true,  minT);
    //lowCrossResFit(params, points, numKnots, deg);
    //param();
}

void example_1()
{
    gsMatrix<real_t> params, points;
    real_t minT = -1.0;
    index_t sample = 4;
    sampleData(500, params, points, sample, minT);

    std::vector<index_t> numKnots(2);
    numKnots[0] = 46;
    numKnots[1] = 96;

    // for(auto it=dataSizes.begin(); it!=dataSizes.end(); ++it)
    // {
    // 	// TODO: Hack here.
    // }
}

void example_2()
{
    index_t sample = 6;
    std::vector<index_t> dataSizes(2);
    dataSizes[0] = 50;
    dataSizes[1] = 100;
    //dataSizes[2] = 200;
    //dataSizes[3] = 400;

    for(auto it=dataSizes.begin(); it!=dataSizes.end(); ++it)
    {
	gsMatrix<real_t> params, points;
	real_t minT = -1.0; // -1 leads to a confusion index_t / real_t.
	sampleDataGre(*it, params, points, sample, minT, 1.0, 2);

	index_t deg = 2;
	index_t numKnots = *it - deg - 1;
	index_t maxIter = 25;

	std::string filename = std::to_string(*it);

	//stdFit(        params, points, numKnots, deg, minT);
	// lowSVDFit(     params, points, numKnots, deg, maxIter, filename, minT);
	//lowCrossAppFit(params, points, numKnots, deg, maxIter, filename, false, minT);
	lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, true,  minT);
    }
}

void example_3()
{
    // Integrate 1/4 exp(sqrt(x^2+y^2) over the unit square.
    // The correct answer (from Wolfram cloud) is 0.55886658581726677503.
    index_t numDOF = 100;

    // 1. Take a tensor-product basis for fitting.
    index_t deg = 3;
    index_t numKnots = numDOF - deg -1;
    real_t tMin = -1.0;
    real_t tMax =  1.0;
    gsKnotVector<real_t> kv(tMin, tMax, numKnots, deg + 1);
    gsTensorBSplineBasis<2, real_t> fittingBasis(kv, kv);

    // 2. Sample points and weights for a high-order quadrature rule.
    index_t sample = 6;
    gsVector<real_t> uWeights, vWeights;
    gsMatrix<real_t> params, points;
    sampleDataGauss<real_t>(fittingBasis, params, points, uWeights, vWeights, sample, 2.0, 3);
    gsInfo << "Sampled " << params.cols() << " points, fitting with "
	   << fittingBasis.size() << " DOF." << std::endl;

    // 3. Approximate in the least-squares sense.
    index_t maxIter = 20;
    std::string filename = "example-3-case-6-weighted";

    gsLowRankFitting<real_t> fitting(params, points, uWeights, vWeights, fittingBasis);
    gsInfo << "CrossApp fitting with weights and pivoting:\n";
    //gsInfo << "Params (" << params.rows() << "x" << params.cols() << "):\n" << params << std::endl;
    //gsInfo << "Points (" << points.rows() << "x" << points.cols() << "):\n" << points << std::endl;
    fitting.computeCross(true, maxIter, sample);
    fitting.exportL2Err(filename + "piv_L2.dat");

    // 4. Compare with the full fitting without weights.
    stdFit(params, points, numKnots, deg, sample, tMin);

    // 5. Compare with Irina & Clemens using the same spline space.
    sampleDataGre(numDOF, params, points, sample, tMin, tMax, deg);
    gsInfo << "Sampled " << params.cols() << " points, fitting with "
	   << fittingBasis.size() << " DOF." << std::endl;
    filename = "example-3-case-6-inter100";
    lowCrossAppFit(params, points, numKnots, deg, sample, maxIter, filename, true, tMin);

    // 6. And finally full fitting in these points.
    stdFit(params, points, numKnots, deg, sample, tMin);
}

void example_4()
{
    index_t sample = 4; // TODO: Choose!
    index_t deg = 3;
    index_t maxIter = 10;
    real_t tMin = -1;
    real_t tMax = 1;
    index_t nExperiments = 6;

    gsMatrix<real_t> params, points;

    std::vector<index_t> numDOF(nExperiments);
    numDOF[0] = 16;
    numDOF[1] = 32;
    numDOF[2] = 64;
    numDOF[3] = 128;
    numDOF[4] = 256;
    numDOF[5] = 512;
    sampleDataGre(512, params, points, sample, tMin, tMax, deg);

    std::vector<real_t> stdL2Err;
    std::vector<std::vector<real_t>> lowL2Err;
    std::vector<real_t> knotSpans;
    for(auto it=numDOF.begin(); it!=numDOF.end(); ++it)
    {
	index_t numKnots = *it - deg - 1;
	stdL2Err.push_back(stdFit(params, points, numKnots, deg, sample, tMin));

	gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	// Remember h.
	knotSpans.push_back(knots.maxIntervalLength());
	gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
	lowRankFitting.computeCross(true, maxIter, sample);
	lowL2Err.push_back(lowRankFitting.getl2Err());
    }

    // Compute total number of DOF.
    std::vector<real_t> totDOF(numDOF.size());
    for(size_t i=0; i<numDOF.size(); i++)
	totDOF[i] = numDOF[i] * numDOF[i];

    gsWriteGnuplot(totDOF, stdL2Err, "example-4-std.dat");

    for(index_t j=0; j<maxIter; j++)
    {
	std::vector<real_t> curr;
	for(size_t i=0; i<numDOF.size(); i++)
	    if(size_t(j) < lowL2Err[i].size())
		curr.push_back(lowL2Err[i][j]);

	if(curr.size() > 0)
	    gsWriteGnuplot(totDOF, curr, "example-4-rank-" + std::to_string(j+1) + ".dat");
    }

    // Print the LaTeX table with errors.
    gsInfo << "rank \\textbackslash $h$";
    for(auto it=knotSpans.begin(); it!=knotSpans.end(); ++it)
	gsInfo << " & " << *it;
    gsInfo << " \\\\" << std::endl;

    for(index_t j=0; j<maxIter; j++)
    {
	gsInfo << j+1 << " & ";
	for(index_t i=0; i<nExperiments; i++)
	{
	    if(size_t(j) < lowL2Err[i].size())
		gsInfo << lowL2Err[i][j];
	    else
		gsInfo << " - ";
	    if(i == nExperiments - 1)
		gsInfo << "\\\\" << std::endl; // We need quadruple backslash, because a double backslash prints as a single backslash.
	    else
		gsInfo << " & ";
	}
    }
}

void example_5()
{
    index_t sample = 4; // TODO: Choose!
    index_t deg = 3;
    real_t tMin = -1;
    real_t tMax = 1;

    gsMatrix<real_t> params, points;

    index_t numDOF = 32;

    std::vector<real_t> stdL2Err;
    index_t numKnots = numDOF - deg - 1;
    // TODO: Sampling 32 points and fitting them with 32 DOF does not converge (!).
    sampleDataGre(512, params, points, sample, tMin, tMax, deg);

    gsInfo << "params: " << params.rows() << " x " << params.cols() << std::endl;

    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);

    gsInfo << "basis:\n" << basis << std::endl;
    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
    lowRankFitting.computeCross(true, 50, sample);

    stdFit(params, points, numKnots, deg, sample, tMin);
}

void example_6()
{
    // 2023-06-04: Adapted to the settings of example_14.
    index_t sample  = 9;
    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 2;
    index_t nExperiments = 6;
    index_t nRanks = 5;
    index_t numSamples = 300;
    bool printErr = false;

    gsMatrix<real_t> params, points;

    std::vector<index_t> numDOF(nExperiments);
    if(nExperiments > 0) numDOF[0] = 11;
    if(nExperiments > 1) numDOF[1] = 19;
    if(nExperiments > 2) numDOF[2] = 35;
    if(nExperiments > 3) numDOF[3] = 67;
    if(nExperiments > 4) numDOF[4] = 131;
    if(nExperiments > 5) numDOF[5] = 259;

    std::vector<index_t> maxRanks(nRanks);
    if(nRanks > 0) maxRanks[0] = 4;
    if(nRanks > 1) maxRanks[1] = 8;
    if(nRanks > 2) maxRanks[2] = 16;
    if(nRanks > 3) maxRanks[3] = 32;
    if(nRanks > 4) maxRanks[4] = 64;

    std::vector<real_t> timesB;
    gsMatrix<real_t> timesC(numDOF.size(), maxRanks.size());
    gsStopwatch time;
    for(size_t i=0; i<numDOF.size(); i++)
    {
	gsInfo << "==========================================\n";
	gsInfo << "numDOF: " << numDOF[i] << std::endl;
	sampleData(numSamples, params, points, sample, tMin, tMax);
	index_t numKnots = numDOF[i] - deg - 1;

	//gsInfo << "Method A:\n";
	//stdFit(params, points, numKnots, deg, sample, tMin);

	gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	gsLowRankFitting<real_t> lowRankFitting(params, points, basis);

	gsInfo << "Method B:\n";
	timesB.push_back(lowRankFitting.methodB(printErr));

	for(size_t j=0; j<maxRanks.size(); j++)
	{
	    index_t maxRank = maxRanks[j];
	    gsInfo << "Method C, rank " << maxRank << ":\n";
	    timesC(i, j) = lowRankFitting.methodC(printErr, maxRank);
	}
    }

    gsWriteGnuplot(numDOF, timesB, "example-6-method-B-fast.dat");

    for(size_t j=0; j<maxRanks.size(); j++)
    {
	std::vector<real_t> timesCurr(numDOF.size());
	for(size_t i=0; i<numDOF.size(); i++)
	    timesCurr[i] = timesC(i, j);
	gsWriteGnuplot(numDOF, timesCurr, "example-6-method-C-fast-iter-" + std::to_string(maxRanks[j]) + ".dat");
    }
}

void example_7()
{
    index_t sample = 6; // Works well for 4 and 6 but not that well for 8.
    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 2;

    gsMatrix<real_t> params, points;

    index_t numDOF = 32;

    std::vector<real_t> stdL2Err;
    index_t numKnots = numDOF - deg - 1;
    sampleData(512, params, points, sample, tMin, tMax);

    gsInfo << "params: " << params.rows() << " x " << params.cols() << std::endl;

    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);

    gsInfo << "basis:\n" << basis << std::endl;
    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
    //lowRankFitting.computeCrossWithRef(false, 200, sample);
    lowRankFitting.computeCross_3(true, 200, sample);
    //gsWriteParaview(*lowRankFitting.result(), "cross-result", 10000, false, true);
    lowRankFitting.exportl2Err("example-7.dat");

    // stdFit(params, points,     numKnots,     deg, sample, tMin, tMax);
    // stdFit(params, points, 2 * numKnots + 1, deg, sample, tMin, tMax);
    // stdFit(params, points, 4 * numKnots + 3, deg, sample, tMin, tMax);
    // stdFit(params, points, 8 * numKnots + 7, deg, sample, tMin, tMax);
    // stdFit(params, points, 16 * numKnots + 15, deg, sample, tMin, tMax);
}

// for profiling_3
gsMatrix<real_t> convertToMN(const gsMatrix<real_t>& points, index_t rows)
{
    // gsMatrix<real_t> result(rows, points.cols() / rows);
    // for(index_t i=0; i<points.cols(); i++)
    // {
    // 	//gsInfo << "result(" << i%rows << ", " << i/rows << ") = " <<  points(0, i) << std::endl;
    // 	result(i/rows, i%rows) = points(0, i);
    // }

    gsMatrix<real_t> result(rows, points.rows() / rows);
    for(index_t i=0; i<points.rows(); i++)
	result(i%rows, i/rows) = points(i, 0);

    return result;
}

void profiling_1()
{
    gsMatrix<real_t> params, points;
    sampleData(128, params, points, 8, 0.0, 1.0);

    index_t uNum = math::sqrt(params.cols());
    index_t maxIter = 64;
    gsMatrix<real_t> ptsMN = convertToMN(points.transpose(), uNum);
    gsMatrixCrossApproximation<real_t> crossApp(ptsMN.transpose());
    gsMatrix<real_t> uMat(uNum, maxIter), vMat(uNum, maxIter), tMat(maxIter, maxIter);
    gsVector<real_t> uVec, vVec;
    real_t sigma;
    tMat.setZero();
    for(index_t i=0; i<maxIter && crossApp.nextIteration(sigma, uVec, vVec, true); i++)
    {
	//gsInfo << "i: " << i << " / " << maxIter << std::endl;
	uMat.col(i) = uVec;
	vMat.col(i) = vVec;
	tMat(i, i)  = sigma;
    }    
}

void profiling_3()
{
    gsMatrix<real_t> params, points;
    sampleData(2048, params, points, 6, 0.0, 1.0);
    gsMatrix<real_t> ptsMN = convertToMN(params, math::sqrt(params.cols()));
    gsMatrixCrossApproximation_3<real_t> crossApp(ptsMN, 0);
    crossApp.compute(true, 64);
    gsMatrix<real_t> uMat, vMat, tMat;
    crossApp.getU(uMat);
    crossApp.getV(vMat);
    crossApp.getT(tMat);
}

void profiling_4()
{
    gsMatrix<real_t> params, points;
    sampleData(12, params, points, 8, 0.0, 1.0);
    gsInfo << points << std::endl;
    gsInfo << "==========================================" << std::endl;
    gsKnotVector<real_t> knots(0.0, 1.0, 13, 4);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);
    gsLowRankFitting<real_t> fitting(params, points, basis);
    gsInfo << fitting.returnPoints() << std::endl;
    fitting.testMN(12);
}

// Bert's stopping criterium.
void example_8()
{
    index_t sample = 5;
    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 2;

    gsMatrix<real_t> params, points;

    index_t numDOF = 4;

    std::vector<real_t> stdL2Err;
    index_t numKnots = numDOF - deg - 1;
    sampleDataGre(259, params, points, sample, tMin, tMax);
    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);
    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
    lowRankFitting.computeCrossWithRefAndStop(1e-12, true);
    //lowRankFitting.computeCross(false, 200, 6);

    stdFit(params, points, 0, deg, sample, tMin, tMax);
    stdFit(params, points, 1, deg, sample, tMin, tMax);
    stdFit(params, points, 3, deg, sample, tMin, tMax);
    stdFit(params, points, 7, deg, sample, tMin, tMax);
    stdFit(params, points, 15, deg, sample, tMin, tMax);
    stdFit(params, points, 31, deg, sample, tMin, tMax);
    stdFit(params, points, 63, deg, sample, tMin, tMax);
    stdFit(params, points, 127, deg, sample, tMin, tMax);
    stdFit(params, points, 255, deg, sample, tMin, tMax);
    //stdFit(params, points, 511, deg, sample, tMin, tMax);
}

// Fitting with general weights.
void example_9()
{
    index_t sample = 5;
    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 2;
    index_t nSamples = 259;

    gsMatrix<real_t> params, points;

    index_t numDOF = 4;

    std::vector<real_t> stdL2Err;
    index_t numKnots = numDOF - deg - 1;
    sampleDataGre(nSamples, params, points, sample, tMin, tMax);
    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);
    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
    lowRankFitting.computeFull();

    // work in progress
    gsVector<real_t> weights = matrixUtils::oneVector<real_t>(nSamples * nSamples);
    lowRankFitting.setWeights(weights);
    gsInfo << "weighted l2-error: " << lowRankFitting.get_weightedl2Error() << std::endl;

    stdFit(params, points, 0, deg, sample, tMin, tMax);
}

// Fitting unstructured data by sampling a mesh.
void example_10()
{
    //gsFileData<> fd("nefertiti-parameterized.off");
    // gsMesh<real_t>::uPtr mm = fd.getFirst<gsMesh<real_t>>();

    // const auto vertices = mm->vertices();
    // for(auto vi=vertices.begin(); vi!=vertices.end(); ++vi)
    // {
    // 	real_t u = (*vi)->x();
    // 	real_t v = (*vi)->y();
    // 	//real_t disp = 0.25 * math::sin(2 * EIGEN_PI * x) * math::sin(2 * EIGEN_PI * y);
    // 	//real_t disp = 0.25 * math::exp(math::sqrt(u * u + v * v));
    // 	real_t arg = 5 * EIGEN_PI * ((u - 0.2) * (u - 0.2) + (v - 0.0) * (v - 0.0));
    //     real_t disp =  math::sin(arg) / arg;

    // 	(*vi)->move(0, 0, disp);
    // }
    //gsWriteParaview(*mm, "waves");

    index_t uNum = 64;
    index_t vNum = 64;
    index_t sample = 4;
    std::string filename = "nefertiti-parameterized.off";
    gsMatrix<real_t> vertParams, vertPoints, gridParams, gridPoints;

    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 1;
    
    sampleDataMeshInterpolated(gridParams, gridPoints, filename, uNum, vNum, sample);
    sampleDataMeshVertices(vertParams, vertPoints, filename, sample);

    index_t maxGridIter = 16;
    std::vector<real_t> totDOF(maxGridIter);
    std::vector<real_t> stdErrs, tensErrs;
    // Warning: taking 0 DOF leads to a memory overflow.
    for(index_t i=1; i<=maxGridIter; i++)
    {
	index_t numDOF = 4 * i;
	index_t numKnots = numDOF - deg - 1;
	totDOF[i] = numDOF * numDOF;

	gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	gsLowRankFitting<real_t> lowRankFitting(gridParams, gridPoints, basis);
	lowRankFitting.computeFull();

	real_t tensErr = l2Error(lowRankFitting.result(), vertParams, vertPoints);
	gsInfo << "tensor-product l2-error in vertices: " << tensErr << std::endl;
	tensErrs.push_back(tensErr);
    }

    index_t maxVertIter = 5;
    for(index_t i=1; i<maxVertIter; i++)
    {
	index_t numDOF = 4 * i;
	index_t numKnots = numDOF - deg - 1;

	real_t stdErr = stdFit(vertParams, vertPoints, numKnots, deg, sample, tMin, tMax);
	gsInfo << "full fitting   l2-error in vertices: " << stdErr << std::endl;
	stdErrs.push_back(stdErr);
    }
    gsWriteGnuplot(totDOF, stdErrs, "example-10-std.dat");
    gsWriteGnuplot(totDOF, tensErrs, "example-10-tens.dat");
}

void debugging()
{
    gsMatrix<real_t> mat(0, 0);
    // mat << 1, 0, 0,
    // 	0, 1, 0,
    // 	0, 0, 1;

    // gsVector<real_t> vec(3);
    // vec << 1, 2, 3;
    //matrixUtils::appendCol(mat, vec);
    matrixUtils::appendDiag<real_t>(mat, 1);
    matrixUtils::appendDiag<real_t>(mat, 2);
    matrixUtils::appendDiag<real_t>(mat, 3);
    gsInfo << mat << std::endl;
}

void setDomain(index_t sample, real_t &tMin, real_t &tMax)
{
    if(sample == 4 || sample == 6 || sample == 9)
    {
	tMin = -1;
	tMax = 1;
    }
    else if(sample == 8)
    {
	tMin = 0;
	tMax = 2;
    }
    else
    {
	tMin = 0;
	tMax = 1;
    }	    

    gsInfo << "Sample: " << sample << ", tMin: " << tMin << ", tMax: " << tMax << std::endl;
}

void printMessage(index_t message)
{
    if(message == 0)
	gsInfo << "success";
    else if(message == 1)
	gsInfo << "cannot converge";
    else
	gsInfo << "max iter reached";
    gsInfo << std::endl;
}

void gnuplot_11(const std::string& filename, const std::string& data, real_t stdValue)
{
    std::ofstream fout;
    fout.open(filename);
    std::string rgb("'#0072bd'");
    fout << "set style line 1 \\\n"
	 << "linecolor rgb " << rgb << " \\\n"
	 << "linetype 1 linewidth 2 \\\n"
	 << "pointtype 1 pointsize 1\n" << std::endl;

    fout << "set style line 11 \\\n"
	 << "linecolor rgb " << rgb << " \\\n"
	 << "linetype -1 dashtype 2 linewidth 2\n" << std::endl;

    fout << "set log y\n"
	 << "set format y \"10^{%L}\"\n"
	 << std::endl;

    fout << "std(p) = (p < 12 ? " << stdValue << " : 1/0)\n" << std::endl;

    fout << "plot '" << data << "' index 0 with linespoints linestyle 1 title 'low-rank-fitting',\\\n"
	 << "std(x) linestyle 11 title 'full LS fitting'" << std::endl;
    fout.close();
}

/// Algo converges to the full approximation when
/// $\varepsilon_{\text{abort}}=\infty$ and
/// $\varepsilon_{\text{accept}}=0.$
void example_11(index_t sample, index_t deg, index_t numSamples, real_t epsAcc, real_t epsAbort)
{
    // Strange: for sample = 6, our approximation converges to the std
    // solution if numDOF < numSamples but outperforms it when they
    // are equal.

    real_t tMin, tMax;
    setDomain(sample, tMin, tMax);

    gsMatrix<real_t> params, points;
    sampleDataGre(numSamples, params, points, sample, tMin, tMax);
    //sampleData(numSamples, params, points, sample, tMin, tMax);
    std::vector<real_t> stdL2Err;

    for(index_t numDOF = 10; numDOF <= numSamples; numDOF += 10)
    {
	index_t numKnots = numDOF - deg - 1;
	gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	gsLowRankFitting<real_t> lowRankFitting(params, points, basis);
	//lowRankFitting.computeCrossWithStop(0, 10);
	index_t message = lowRankFitting.computeCrossWithStop(epsAcc, epsAbort);
	if(message == 0)
	    gsInfo << "success";
	else if(message == 1)
	    gsInfo << "cannot converge";
	else
	    gsInfo << "max iter reached";
	gsInfo << std::endl;

	// TODO next time: Bert's criterium is stopping too early or not at all now.
	// The correct value seems to depend on the example but also on numDOF:
	// for small bases it stops too early, for big ones too late.
	// Maybe we can make it relative to the l2-error?

	real_t std = stdFit(params, points, numKnots, deg, sample, tMin, tMax);
	gsInfo << "std: " << std  << " with " << numDOF << " DOF" << std::endl;
	stdL2Err.push_back(std);

	std::string filename("example-11-" + std::to_string(numDOF));
	lowRankFitting.exportl2Err(filename + ".dat");
	gnuplot_11(filename + ".gnu", filename + ".dat", std);
	if(numDOF == 10)
	    lowRankFitting.exportDecompErr("example-11-decomp.dat");
    }
}

void costs_12()
{
    index_t sample  = 9;
    index_t deg = 3;
    real_t tMin = 0;
    real_t tMax = 2;
    index_t nExperiments = 6;
    index_t numSamples = 300;
    // bigger example:
    // index_t nExperiments = 9;
    // index_t numSamples = 2500;
    bool printErr = true;
    real_t nRepetitions(25); // real_t so as to prevent division problems

    gsMatrix<real_t> params, points;

    // The following three arrays must be of length nExperiments.
    // The ranks up to 259 DOF (inclusive) have been computed by stopCrit.
    index_t numDOF[] = {11, 19, 35, 67, 131, 259, 515, 1027, 2051};
    // index_t maxRankPiv[] = {36, 44, 72, 84, 96, 142, 280, 560, 1120};
    // index_t maxRankFull[] = {24, 38, 58, 76, 88, 130, 270, 540, 1080};
    index_t maxRankPiv[] = {36, 44, 72, 81, 81, 81, 81, 81, 81};
    index_t maxRankFull[] = {24, 38, 58, 76, 81, 81, 81, 81, 81};
    std::vector<index_t> DOF2D(nExperiments);
    for(size_t i=0; i<DOF2D.size(); i++)
	DOF2D[i] = numDOF[i] * numDOF[i];

    std::vector<real_t> timesA(nExperiments, 0), timesB(nExperiments, 0),
	timesCpivDecomp(nExperiments, 0), timesCfullDecomp(nExperiments, 0),
	timesCpivSolve(nExperiments, 0), timesCfullSolve(nExperiments, 0);
    sampleData(numSamples, params, points, sample, tMin, tMax);
    for(index_t j=0; j<nRepetitions; j++)
    {
	gsInfo << "Repeating the measurements for the " << j << "-th time.\n";
	for(index_t i=0; i<nExperiments; i++)
	{
	    gsInfo << "==========================================\n";
	    gsInfo << "numDOF: " << numDOF[i] << std::endl;
	    index_t numKnots = numDOF[i] - deg - 1;

	    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
	    gsTensorBSplineBasis<2, real_t> basis(knots, knots);
	    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);

	    //gsInfo << "Method A:\n";
	    //timesA[i] += stdFit(params, points, numKnots, deg, sample, tMin) / nRepetitions;

	    gsInfo << "Method B:\n";
	    timesB[i] += lowRankFitting.methodB(printErr) / nRepetitions;

	    gsInfo << "Method C (pivot), rank " << maxRankPiv[i] << ":\n";
	    timesCpivSolve[i] += lowRankFitting.methodC(printErr, maxRankPiv[i], true) / nRepetitions;
	    timesCpivDecomp[i] +=  lowRankFitting.getDecompTime() / nRepetitions;

	    gsInfo << "Method C (full), rank " << maxRankFull[i] << ":\n";
	    timesCfullSolve[i] += lowRankFitting.methodC(printErr, maxRankFull[i], false) / nRepetitions;
	    timesCfullDecomp[i] += lowRankFitting.getDecompTime() / nRepetitions;
	}
    }

    gsWriteGnuplot(DOF2D, timesA, "example-6-method-A-solutions.dat");
    gsWriteGnuplot(DOF2D, timesB, "example-6-method-B-solutions.dat");
    gsWriteGnuplot(DOF2D, timesCpivSolve, "example-6-method-C-solutions-pivot.dat");
    gsWriteGnuplot(DOF2D, timesCfullSolve, "example-6-method-C-solutions-full.dat");
    gsWriteGnuplot(DOF2D, timesCpivDecomp, "example-6-method-C-decomposition-pivot.dat");
    gsWriteGnuplot(DOF2D, timesCfullDecomp, "example-6-method-C-decomposition-full.dat");
}


/// Returns i-th colour from the bright colour scheme for qualitative
/// data from https://personal.sron.nl/~pault/.
std::string paultBright(index_t i)
{
    switch(i)
    {
    case 0:
	return "'#4477aa'";
    case 1:
	return "'#66ccee'";
    case 2:
	return "'#228833'";
    case 3:
	return "'#ccbb44'";
    case 4:
	return "'#ee6677'";
    case 5:
	return "'#aa3377'";
    case 6:
	return "'#bbbbbb'";
    default:
	gsWarn << "Index out of the scale 0, ..., 6" << std::endl;
	return "";	    
    }
}

void gnuplotWriteColourArray(std::ofstream& fout, size_t size)
{
    fout << "array Rgb[" << size << "]\n";
    for(size_t i=0; i<size; i++)
	fout << "Rgb[" << i+1 << "] = " << paultBright(i) << "\n";
    fout << std::endl;
}

void gnuplot_12(const std::string& filename,
		const std::string& data,
		real_t stdValue,
		index_t iteration,
		index_t numDOF)
{
    std::ofstream fout;
    fout.open(filename);
    gnuplotWriteColourArray(fout, 6);

    fout << "set style line 1 \\\n"
	 << "linecolor rgb Rgb[" << iteration << "] \\\n"
	 << "linetype 1 linewidth 2 \\\n"
	 << "pointtype " << iteration << " pointsize 1\n" << std::endl;

    fout << "set style line 11 \\\n"
	 << "linecolor rgb Rgb[" << iteration << "] \\\n"
	 << "linetype -1 dashtype 2 linewidth 2\n" << std::endl;

    fout << "set log y\n"
	 << "set format y \"10^{%L}\"\n"
	 << "set xrange[0:72]\n"
	 << "set yrange[1e-8:1e2]\n"
	 << "set xlabel 'rank'\n"
	 << "set ylabel 'l2-error'\n"
	 << std::endl;

    index_t rightEnd;
    if(iteration == 0)
	rightEnd = 4;
    else if(iteration == 1)
	rightEnd = 12;
    else if(iteration == 2)
	rightEnd = 22;
    else if(iteration == 3)
	rightEnd = 32;
    else if(iteration == 4)
	rightEnd = 39;
    else if(iteration == 5)
	rightEnd = 60;
    else
	rightEnd = 72;

    fout << "std(p) = (p < " << rightEnd << " ? " << stdValue << " : 1/0)\n" << std::endl;

    fout << "plot '" << data << "' index 0 with linespoints linestyle 1"
	 << " title 'Algorithm LowRankFit " << numDOF << "x" << numDOF << "DOF',\\\n"
	 << "std(x) linestyle 11 title 'the standard method'" << std::endl;
    fout.close();
}

void example_12()
{
    index_t sample      = 9;
    index_t deg         = 3;
    index_t numSamples  = 300;
    index_t numIntKnots = 7;

    real_t epsAbort = 1e-6;
    
    real_t tMin, tMax;
    setDomain(sample, tMin, tMax);

    gsMatrix<real_t> params, points;
    //sampleDataGre(numSamples, params, points, sample, tMin, tMax);
    sampleData(numSamples, params, points, sample, tMin, tMax);

    gsKnotVector<real_t> knots(tMin, tMax, numIntKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);

    std::vector<real_t> stdl2Err, totDOF, stdCost, fullACACost, pivACACost;
    index_t iteration = 1;
    do
    {
	gsInfo << "==========================================" << std::endl;
	index_t numDOF = basis.size(0);
	index_t numKnots = numDOF - deg - 1;

	// Compute the error using standard method first.
	gsStopwatch timeStd, timeApp;
	timeStd.restart();
	real_t std = stdFit(params, points, numKnots, deg, sample, tMin, tMax);
	timeStd.stop();
	stdl2Err.push_back(std);
	stdCost.push_back(math::sqrt(params.cols()) + numDOF);

	totDOF.push_back(basis.size());
	std::string filename("example-12-" + std::to_string(numDOF));

	gsLowRankFitting<real_t> lowRankFittingFull(params, points, basis, 4e-13);
	timeApp.restart();
	printMessage(lowRankFittingFull.computeCrossWithStop(1.01 * std, epsAbort, false, false));
	timeApp.stop();

	gsInfo << " in " << timeApp.elapsed() << "s." << std::endl;
	gsInfo << "std: " << std  << " with " << numDOF << " DOF "
	       << " in " << timeStd.elapsed() << "s." << std::endl;

	fullACACost.push_back(2 * lowRankFittingFull.getRank());
	lowRankFittingFull.exportl2Err(filename + "-full-l2.dat");
	gnuplot_12(filename + "-full-l2.gnu", filename + "-full-l2.dat", std, iteration, numDOF);

	gsLowRankFitting<real_t> lowRankFittingPiv(params, points, basis, 1e-12);
	printMessage(lowRankFittingPiv.computeCrossWithStop(1.01 * std, epsAbort, true, false));

	pivACACost.push_back(2 * lowRankFittingPiv.getRank());
	lowRankFittingPiv.exportl2Err(filename + "-piv-l2.dat");
	gnuplot_12(filename + "-piv-l2.gnu", filename + "-piv-l2.dat", std, iteration, numDOF);

	basis.uniformRefine();
	iteration++;
    } while(basis.size(0) <= numSamples);

    // TODO: gnu-file for costs

    gsWriteGnuplot(totDOF, fullACACost, "example-12-aca-full-cost.dat");
    gsWriteGnuplot(totDOF, pivACACost, "example-12-aca-piv-cost.dat");
    gsWriteGnuplot(totDOF, stdCost, "example-12-std-cost.dat");

    // real_t lowTotCost(0), stdTotCost(0);
    // for(auto it=lowCost.begin(); it!=lowCost.end(); ++it)
    // 	lowTotCost += *it;
    // for(auto it=stdCost.begin(); it!=stdCost.end(); ++it)
    // 	stdTotCost += *it;

    // gsInfo << "==========================================" << std::endl;
    // gsInfo << "total low cost: " << lowTotCost << ", total std cost: " << stdTotCost << std::endl;
}

void gnuplotWriteLinestyles(std::ofstream& fout, size_t size)
{
    fout << "do for [i=1:" << size << "] {\n"
	 << "    set style line i linetype i linewidth 2"
	 << " lc rgb Rgb[i]\n"
	 << "}\n";
    fout << std::endl;
}

void gnuplotWriteLabels(std::ofstream& fout, std::string xlabel, std::string ylabel)
{
    fout << "set log y\n";
    fout << "set format y \"10^{%L}\"\n";
    fout << "set xlabel \"" << xlabel << "\"\n";
    fout << "set ylabel \"" << ylabel << "\"\n";
    fout << std::endl;
}

void gnuplot_13(const std::vector<index_t> numsDOF,
		const std::vector<std::vector<real_t>>& errsL2Int,
		const std::vector<std::vector<real_t>>& errsL2Gre,
		const std::vector<std::string>& filenames,
		const std::string& filename)
{
    size_t size = numsDOF.size();
    GISMO_ASSERT(filenames.size() == size, "filenames of wrong size");
    GISMO_ASSERT(errsL2Int.size() == size, "errsL2Int of wrong size");
    GISMO_ASSERT(errsL2Gre.size() == size, "errsL2Gre of wrong size");

    std::ofstream fout;
    fout.open(filename);
    gnuplotWriteColourArray(fout, size);

    fout << "do for [i=1:" << size << "] {\n"
	 << "    set style line 2 * i - 1 linetype i linewidth 2 pointsize 0"
	 << " lc rgb Rgb[i] dashtype 2\n"
	 << "    set style line 2 * i     linetype i linewidth 2 pointsize 0"
	 << " lc rgb Rgb[i]\n"
	 << "}\n";
    fout << std::endl;

    fout << "set xrange[0:24]\n"
	 //<< "set yrange[1e-5:0.2]\n"
	 << "set log y\n"
	 << "set xlabel \"rank\"\n"
	 << "set ylabel \"L2-error\"\n"
	 << "set format y \"10^{%L}\"\n"
	 << std::endl;

    fout << "plot ";
    for(size_t i=0; i<size; i++)
    {
	std::string filenameInt = filenames[i] + "-int.dat";
	std::string filenameGre = filenames[i] + "-gre.dat";
	gsWriteGnuplot(errsL2Int[i], filenameInt);
	gsWriteGnuplot(errsL2Gre[i], filenameGre);

	fout << "'" << filenameGre << "' index 0 with linespoints linestyle " << 2*i+1
	     << " title 'low rank interpolation, "
	     << numsDOF[i] << "x" << numsDOF[i] << " DOF', \\"
	     << std::endl << "     "
	     << "'" << filenameInt << "' index 0 with linespoints linestyle " << 2*i+2
	     << " title 'Alg. LowRankFit, "
	     << numsDOF[i] << "x" << numsDOF[i] << " DOF'";
	if(i != size - 1)
	    fout << ", \\" << std::endl << "     ";
	else
	    fout << std::endl;
    }
    fout.close();
}

void example_13(index_t sample, index_t deg, real_t epsAcc, real_t epsAbort,
		real_t quA, index_t quB)
{
    real_t tMin, tMax, zero = 1e-13;
    setDomain(sample, tMin, tMax);
    bool pivot = true;

    std::vector<index_t>             numsDOF;
    std::vector<std::vector<real_t>> errsL2Int, errsL2Gre;
    std::vector<std::string>         filenames;

    for(index_t numDOF = 10; numDOF <= 160; numDOF *= 2)
    {
	numsDOF.push_back(numDOF);

	// 1. Take a tensor-product basis for fitting.
	index_t numKnots = numDOF - deg -1;
	gsKnotVector<real_t> kv(tMin, tMax, numKnots, deg + 1);
	gsTensorBSplineBasis<2, real_t> fittingBasis(kv, kv);

	// 2. Sample points and weights for a quadrature rule.
	gsVector<real_t> uWeights, vWeights;
	gsMatrix<real_t> paramsInt, pointsInt, paramsGre, pointsGre;
	sampleDataGauss<real_t>(fittingBasis, paramsInt, pointsInt, uWeights, vWeights,
				sample, quA, quB);
	gsInfo << "Sampled " << paramsInt.cols() << " points, fitting with "
	       << fittingBasis.size() << " DOF using the "
	       << quA * deg + quB << "-point Gauss-Legendre quadrature rule." << std::endl;

	// 3. Approximate in the least-squares sense.
	gsLowRankFitting<real_t> fittingInt(paramsInt, pointsInt, uWeights, vWeights,
					    fittingBasis, zero, sample);
	gsInfo << "CrossApp fitting with weights and pivoting:\n";
	fittingInt.computeCrossWithStop(epsAcc, epsAbort, pivot, false);
	errsL2Int.push_back(fittingInt.getL2Err());

	// 4. Compare with the full fitting without weights.
	//stdFit(paramsInt, pointsInt, numKnots, deg, sample, tMin);

	// 5. Compare with Irina & Clemens using the same spline space.
	sampleDataGre(numDOF, paramsGre, pointsGre, sample, tMin, tMax, deg);
	gsInfo << "Sampled " << paramsGre.cols() << " points, fitting with "
	       << fittingBasis.size() << " DOF." << std::endl;
	gsLowRankFitting<real_t> fittingGre(paramsGre, pointsGre, fittingBasis, zero, sample);
	fittingGre.computeCrossWithStop(epsAcc, epsAbort, pivot);
	errsL2Gre.push_back(fittingGre.getL2Err());

	filenames.push_back("example-13-" + std::to_string(numDOF));
    }

    gnuplot_13(numsDOF, errsL2Int, errsL2Gre, filenames, "example-13.gnu");
}

void gnuplot_14(const std::vector<real_t>& stdErrs,
		const std::vector<index_t>& numsDOF,
		const std::vector<index_t>& lowCost,
		const std::vector<real_t>& lowErr,
		const std::vector<std::string>& datfiles,
		const std::string& pivot,
		real_t tol)
{
    const std::string gnufile("example-14-" + pivot + ".gnu");
    //const std::string datfile("example-14-" + pivot + ".dat");
    size_t size = stdErrs.size();

    std::ofstream fout;
    fout.open(gnufile);
    gnuplotWriteColourArray(fout, size);
    gnuplotWriteLinestyles(fout, size);
    gnuplotWriteLabels(fout, "rank", "l2-error");

    fout << "tol(p) = " << tol << "\n";
    for(size_t i=0; i<size; i++)
    {
	fout << "std" << numsDOF[i] << "(p) = "
	     << "(p < ";
	if(i==0)
	    fout << "4";
	else if(i==1)
	    fout << "12";
	else if(i==2)
	    fout << "22";
	else if(i==3)
	    fout << "32";
	else if(i==4)
	    fout << "39";
	else
	    fout << "60";

	fout << " ? " << stdErrs[i] << " : 1/0)\n";
    }
    fout << std::endl;

    for(size_t i=0; i<size; i++)
    {
	fout << "set arrow from "
	     << lowCost[i] / 2 << ", graph 0 to "
	     << lowCost[i] / 2 << ", graph 1 nohead linecolor rgb Rgb["
	     << i+1 << "] linewidth 2 dashtype 3\n";
    }
    fout << std::endl;
    //gsWriteGnuplot(lowErr, datfile);

    // fout << "plot '" << datfile << "' index 0 with linespoints"
    // 	 << " linestyle " << size << " pointtype 1 title 'Algorithm Fast Fit',\\\n";
    fout << "plot \\\n";
    for(size_t i=0; i<datfiles.size(); i++)
	fout << "'" << datfiles[i] << "' index 0 with linespoints"
	     << " linestyle " << i+1 << " pointtype " << i+1
	     << " title  'Alg. LowRankFit, " << numsDOF[i] << "x" << numsDOF[i] << "DOF',\\\n";
    
    fout << "tol(x) linetype 1 linewidth 2 dashtype 4 linecolor rgb '#000000'"
	 << " title 'epsAcc',\\\n";
    for(size_t i=0; i<size; i++)
    {
	fout << "std" << numsDOF[i] << "(x) linestyle " << i+1 << " dashtype 2"
	     << " title 'the standard method, " << numsDOF[i] << "x" << numsDOF[i] << "DOF'";
	if(i != size - 1)
	    fout << ",\\\n";
    }
}

void costs_14_oneRow(std::ostream& fout,
		     const std::string& intro,
		     const std::vector<index_t>& values,
		     bool sumEntries)
{
    index_t sum(0);
    fout << intro << " & ";
    for(auto it=values.begin(); it!=values.end(); ++it)
    {
	fout << *it;
	sum += *it;
	fout << " & ";
    }
    if(sumEntries)
	fout << sum;
    else
	fout << "sum";
    fout << std::endl;
    fout << "      \\\\ \\hline" << std::endl;
}

void costs_14(const std::vector<index_t>& numDOFs,
	      const std::vector<index_t>& costsACAFull,
	      const std::vector<index_t>& costsACAPiv,
	      const std::vector<index_t>& costsStd,
	      const std::string filename)
{
    std::ofstream fout;
    fout.open(filename);

    fout << "    \\begin{tabular}[ht]{|l|";
    for(size_t i=0; i<numDOFs.size(); i++)
	fout << "c|";
    fout << "|c|}" << std::endl;
    fout << "      \\hline" << std::endl;

    costs_14_oneRow(fout, "      $p=q$:",                               numDOFs,      false);
    costs_14_oneRow(fout, "      Alg. \\ref{alg:fit} without pivoting", costsACAFull, true);
    costs_14_oneRow(fout, "      Alg. \\ref{alg:fit} with pivoting",    costsACAPiv,  true);
    costs_14_oneRow(fout, "      Full LS fitting",                      costsStd,     true);

    fout << "    \\end{tabular}" << std::endl;
    fout.close();
}

void example_14()
{
    index_t sample      = 9;
    index_t deg         = 3;
    index_t numSamples  = 300;
    index_t numIntKnots = 7;
    real_t  epsAcc      = 1e-7;
    real_t  epsAbort    = 1e-7;

    real_t tMin, tMax;
    setDomain(sample, tMin, tMax);

    gsMatrix<real_t> params, points;
    sampleData(numSamples, params, points, sample, tMin, tMax);

    gsKnotVector<real_t> knots(tMin, tMax, numIntKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);

    std::vector<real_t>  l2ErrStd, l2ErrACAFull, l2ErrACAPiv;
    std::vector<index_t> totDOF, costsStd, costsACAFull, costsACAPiv;
    std::vector<std::string> datfilesFull, datfilesPiv;

    do
    {
	index_t numDOF = basis.size(0);
	totDOF.push_back(numDOF);
	gsLowRankFitting<real_t> lowRankFitting(params, points, basis);

	index_t numKnots = numDOF - deg - 1;
	real_t std = stdFit(params, points, numKnots, deg, sample, tMin, tMax);
	gsInfo << "std: " << std  << " with " << numDOF << " DOF" << std::endl;
	l2ErrStd.push_back(std);
	costsStd.push_back(math::sqrt(params.cols()) + numDOF);

	// Uncomment for Example 12:
	// epsAbort = 1000000;
	// epsAcc = 1.01 * std;

	printMessage(lowRankFitting.computeCrossWithStop(epsAcc, epsAbort, false));
	costsACAFull.push_back(2 * lowRankFitting.getRank());
	l2ErrACAFull = lowRankFitting.getl2Err();
	std::string datfileFull("example-14-" + std::to_string(numDOF) + "-full-l2.dat");
	lowRankFitting.exportl2Err(datfileFull);
	datfilesFull.push_back(datfileFull);

	printMessage(lowRankFitting.computeCrossWithStop(epsAcc, epsAbort, true));
	costsACAPiv.push_back(2 * lowRankFitting.getRank());
	l2ErrACAPiv = lowRankFitting.getl2Err();
	std::string datfilePiv("example-14-" + std::to_string(numDOF) + "-piv-l2.dat");
	lowRankFitting.exportl2Err(datfilePiv);
	datfilesPiv.push_back(datfilePiv);


	basis.uniformRefine();
    } while(basis.size(0) <= numSamples);

    costs_14(totDOF, costsACAFull, costsACAPiv, costsStd, "example-14-costs.txt");

    gnuplot_14(l2ErrStd, totDOF, costsACAFull, l2ErrACAFull, datfilesFull, "full", epsAcc);
    gnuplot_14(l2ErrStd, totDOF, costsACAPiv,  l2ErrACAPiv,  datfilesPiv,  "piv",  epsAcc);
}

/// Extends \a vector by repeatedly appending its last element until
/// its size equals \a newsize.
void extendWithLast(std::vector<real_t>& vector, size_t newSize)
{
    real_t last = vector.back();
    for(size_t i=vector.size(); i<newSize; i++)
	vector.push_back(last);
}

void gnuplot_15(std::vector<real_t> xErr,
		std::vector<real_t> yErr,
		std::vector<real_t> zErr)
{
    // The three error vectors can be of different sizes.
    size_t maxSize = std::max(xErr.size(), std::max(yErr.size(), zErr.size()));
    extendWithLast(xErr, maxSize);
    extendWithLast(yErr, maxSize);
    extendWithLast(zErr, maxSize);

    std::vector<real_t> err(maxSize);
    for(size_t i=0; i<maxSize; i++)
    	err[i] = math::sqrt(xErr[i] * xErr[i] + yErr[i] * yErr[i] + zErr[i] * zErr[i]);

    // gsWriteGnuplot(err, "example-15-x.dat");
    // gsWriteGnuplot(err, "example-15-y.dat");
    // gsWriteGnuplot(err, "example-15-z.dat");
    gsWriteGnuplot(err, "example-15.dat");	
}

void gnuplotMethodComparison(const std::string& exampleName, const std::string& what)
{
    const std::string filename(exampleName + "-" + what + ".gnu");
    size_t size = 3;

    std::ofstream fout;
    fout.open(filename);
    gnuplotWriteColourArray(fout, size);
    gnuplotWriteLinestyles(fout, size);
    gnuplotWriteLabels(fout, "rank", what + "-error");

    fout << "plot '" << exampleName << "-pivot-" << what << ".dat' index 0 with linespoints"
	 << " linestyle 1 pointsize 0 title 'ACA with pivoting',\\\n"
	 <<      "'" << exampleName << "-full-"  << what << ".dat' index 0 with linespoints"
	 << " linestyle 2 pointsize 0 title 'ACA without pivoting',\\\n"
	 <<      "'" << exampleName << "-svd-"   << what << ".dat' index 0 with linespoints"
	 << " linestyle 3 pointsize 0 title 'SVD'";

    fout.close();
}

/**
   \a sample has different meaning in this example:
   0 -> sample points from tmtf-one-surf.xml uniformly;
   1 -> sample points from tmtf-one-surf.xml in Greville abscissae;
   2 -> load points from tmtf-input.xml

   When loading the points, then the parameter matrix has to have the form
   (u0, ..., un, u0, ..., un, ..., u0, ..., un)
   (v0, ..., v0, v1, ..., v1, ..., vn, ..., vn),
   otherwise partitioning into parameters is likely to lead to strange behaviour.
 */
void example_15()
{
    // Settings from the paper.
    index_t deg = 3;
    index_t uNumSamples = 41;
    index_t uNumDOF = 40;
    index_t vNumDOF = 128;
    real_t epsAcc = 5e-5;
    real_t epsAbort = 100;

    index_t uNumKnots = uNumDOF - deg - 1;
    index_t vNumKnots = vNumDOF - deg - 1;
    real_t uMin(0.0), uMax(1.0), vMin(0.0), vMax(1.0);
    gsErrType errType = gsErrType::max;
    index_t dummy = -1; // We don't compute the L2-error.
    real_t zero = 1e-13;

    gsMatrix<real_t> params, points;
    gsFileData<> fd2("crescendo/tmtf-input.xml");
    fd2.getId<gsMatrix<>>(0, params);
    fd2.getId<gsMatrix<>>(1, points);
    
    std::vector<std::vector<real_t>> maxErrs;

    gsKnotVector<real_t> uKnots(uMin, uMax, uNumKnots, deg+1);
    gsKnotVector<real_t> vKnots(vMin, vMax, vNumKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(uKnots, vKnots);
    gsMatrix<real_t> coefs(basis.size(), 3), stdCoefs(basis.size(), 3);
    for(index_t i=0; i<3; i++)
    {
	gsLowRankFitting<real_t> lowRankFitting(params, points.row(i), basis, zero, dummy, errType, uNumSamples);
	lowRankFitting.computeCrossWithStop(epsAcc, epsAbort, true);
	coefs.col(i) = lowRankFitting.result()->coefs();
	maxErrs.push_back(lowRankFitting.getMaxErr());

	gsFitting<real_t> stdFitting(params, points, basis);
	stdFitting.compute();
	stdFitting.computeErrors();
	stdCoefs.col(i) = stdFitting.result()->coefs();
	//gsInfo << "std: " << stdFitting.get_Error() << std::endl;

	if(i==0)
	{
	    const std::string fileBasis("example-15-x");
	    lowRankFitting.exportMaxErr(   fileBasis + "-pivot-max.dat");
	    lowRankFitting.exportDecompErr(fileBasis + "-pivot-decomp.dat");

	    lowRankFitting.computeCrossWithStop(epsAcc, epsAbort, false);
	    lowRankFitting.exportMaxErr(   fileBasis + "-full-max.dat");
	    lowRankFitting.exportDecompErr(fileBasis + "-full-decomp.dat");

	    lowRankFitting.computeSVD(0,   fileBasis);
	    lowRankFitting.exportMaxErr(   fileBasis + "-svd-max.dat");
	    lowRankFitting.exportDecompErr(fileBasis + "-svd-decomp.dat");

	    gnuplotMethodComparison(fileBasis, "max");
	    gnuplotMethodComparison(fileBasis, "decomp");
	}
    }

    //gnuplot_15(maxErrs[0], maxErrs[1], maxErrs[2]);
    gsFileData<> lowRankFd;
    lowRankFd << gsTensorBSpline<2>(basis, coefs);
    lowRankFd.dump("result-low-rank.xml");

    gsFileData<> stdFd;
    stdFd << gsTensorBSpline<2>(basis, coefs);
    stdFd.dump("result-std.xml");
}

/**
   Testing the influence of the decomposition method.
 */
void example_16(index_t sample, index_t deg, index_t numSamples, index_t numDOF,
		real_t epsAcc, real_t epsAbort)
{
    real_t tMin, tMax;
    setDomain(sample, tMin, tMax);

    gsMatrix<real_t> params, points;
    sampleData(numSamples, params, points, sample, tMin, tMax);
    std::vector<real_t> stdl2Err, lowl2Err;
    const std::string fileBasis("example-16");

    index_t numKnots = numDOF - deg - 1;
    gsKnotVector<real_t> knots(tMin, tMax, numKnots, deg+1);
    gsTensorBSplineBasis<2, real_t> basis(knots, knots);

    gsLowRankFitting<real_t> lowRankFitting(params, points, basis);

    gsInfo << "Pivoting ACA\n";
    index_t message = lowRankFitting.computeCrossWithStop(epsAcc, epsAbort, true);
    printMessage(message);
    lowRankFitting.exportl2Err(    fileBasis + "-pivot-l2.dat");
    lowRankFitting.exportDecompErr(fileBasis + "-pivot-decomp.dat");

    gsInfo << "Full ACA\n";
    message = lowRankFitting.computeCrossWithStop(epsAcc, epsAbort, false);
    printMessage(message);
    lowRankFitting.exportl2Err(    fileBasis + "-full-l2.dat");
    lowRankFitting.exportDecompErr(fileBasis + "-full-decomp.dat");

    gsInfo << "SVD\n";
    lowRankFitting.computeSVD(sample, fileBasis);
    lowRankFitting.exportl2Err(    fileBasis + "-svd-l2.dat");
    lowRankFitting.exportDecompErr(fileBasis + "-svd-decomp.dat");

    real_t std = stdFit(params, points, numKnots, deg, sample, tMin, tMax);
    gsInfo << "std: " << std  << " with " << numDOF << " DOF" << std::endl;

    gnuplotMethodComparison(fileBasis, "l2");
    gnuplotMethodComparison(fileBasis, "decomp");
}

int main(int argc, char *argv[])
{
    index_t numSamples = 100;
    index_t uNumSamples = 150;
    index_t vNumSamples = 40;
    index_t numDOF = 50;
    index_t uNumDOF = 75;
    index_t vNumDOF = 20;
    index_t deg = 3;
    index_t sample = 6;
    index_t example = 11;
    index_t quB = 1;

    real_t epsAbort(1);
    real_t epsAcc(0);
    real_t quA = 1;

    bool pivot = false;
    bool verbose = false;

    gsCmdLine cmd("Choose the example and its parameters.");
    cmd.addInt("m", "samples", "number of samples in each direction", numSamples);
    cmd.addInt("n", "dofs", "number of degrees of freedom in each direction", numDOF);
    cmd.addInt("d", "deg", "degree of approximation", deg);
    cmd.addInt("s", "sample", "id of the input function", sample);
    cmd.addInt("e", "example", "which example to compute", example);
    cmd.addInt("r", "quB", "quB in the quadrature rule", quB);
    cmd.addReal("a", "abort", "epsilon abort for Algorithm 1", epsAbort);
    cmd.addReal("t", "tol", "epsilon accept for Algorithm 1", epsAcc);
    cmd.addReal("q", "quA", "quA in the quadrature rule", quA);
    cmd.addSwitch("p", "piv", "activate ACA pivoting", pivot);
    cmd.addSwitch("v", "verbose", "write a lot of messages", verbose);
    cmd.addInt("b", "usamples", "number of samples in the u-direction", uNumSamples);
    cmd.addInt("c", "vsamples", "number of samples in the v-direction", vNumSamples);
    cmd.addInt("f", "udof", "number of DOF in the u-direction", uNumDOF);
    cmd.addInt("i", "vdof", "number of DOF in the v-direction", vNumDOF);
    try
    {
	cmd.getValues(argc, argv);
    }
    catch(int rv)
    {
	return rv;
    }

    switch(example)
    {
    case 1:
	example_1();
	break;
    case 2:
	example_2();
	break;
    case 3:
	example_3();
	break;
    case 4:
	example_4();
	break;
    case 5:
	example_5();
	break;
    case 6:
	example_6();
	break;
    case 7:
	example_7();
	break;
    case 8:
	example_8();
	break;
    case 9:
	example_9();
	break;
    case 10:
	example_10();
	break;
    case 11:
	example_11(sample, deg, numSamples, epsAcc, epsAbort);
	break;
    case 12:
	example_12();
	//costs_12();
	break;
    case 13:
	example_13(5, 3, 0, 10, 1, 0);
	break;
    case 14:
	example_14();
	break;
    case 15:
	example_15();
	break;
    case 16:
	example_16(sample, deg, numSamples, numDOF, epsAcc, epsAbort);
	break;
    default:
	gsWarn << "Unknown example, exiting." << std::endl;
	return -1;
    }

    //checkCrossApp(5, false);
    //profiling_1();
    //profiling_3();
    //profiling_4();
    return 0;    
}

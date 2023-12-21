/** @file embedding_massmatrix.cpp

    @brief Provide the implementation for the mass matrix corresponding to isogeometric thin
    structure which is assumed to be constant throughout the simulation
 *
 */

#include <gismo.h>
#include <iostream>

using namespace gismo;

int main(int argc, char *argv[]){
    bool plot = true;
    //------------------ Generate Geometry ------------------
    /// Master NURBS surface: degree 2, Bezier patch
    gsKnotVector<> Kv(0,2,1,4);
    gsMatrix<> C(25,3);
    C<< 0.124215, 0.0620467, -0.909323,
            0.990116, -0.112269, -0.0161917,
            1.99052, -0.213628, 1.12389,
            2.99999, -0.176554, -0.0225916,
            3.93311, -0.232422, -0.736062,
            -0.0212573, 1.14758, -0.383038,
            1, 1, 0,
            2, 1, 0,
            3, 1, 0,
            4.0773, 1.08316, -0.416151,
            -0.0554789, 2.0257, -0.138956,
            1, 2, 0,
            1.68928, 1.82642, 3.25086,
            3, 2, 0,
            3.94707, 2.11565, -0.357777,
            -0.112924, 2.99649, -0.0987988,
            1, 3, 0,
            2, 3, 0,
            3, 3, 0,
            4.0241, 3.11986, -0.280967,
            -0.209772, 4.06133, -0.303981,
            0.90333, 4.17869, -0.250397,
            1.98218, 4.13487, -0.119872,
            3.10997, 4.14081, -0.364305,
            4.09176, 4.18805, -0.43237;

    gsTensorBSpline<2> master = gsTensorBSpline<2, real_t>( Kv, Kv, give(C) );
    gsTensorBSplineBasis<2> master_basis = master.basis();
    gsInfo << "master_basis = " << master_basis << "\n";

    /// An embedding curve
    // Make a BSpline curve
    gsKnotVector<> kv(0, 1, 1, 3);//start,end,interior knots, start/end multiplicites of knots

    gsMatrix<> coefs(4, 2);
    coefs << 0, 0,
            0.1, 0.2,
            0.5, 0.25,
            0.8, 0.8;

    gsBSpline<> embedding_curve( kv, give(coefs));

    gsBSplineBasis<> curve_basis = embedding_curve.basis();


    gsInfo << "curve_basis = " << curve_basis << "\n";

    // Get the xi extension of the embedded cable (start and end knots)
    gsInfo<<embedding_curve.knots()<<"\n";
    gsVector<> xiExtension(2), etaExtension(2);
    real_t xi_min = embedding_curve.knots().first();
    real_t xi_max = embedding_curve.knots().last();
    xiExtension << xi_min,xi_max;
    gsInfo << "xiExtension = " << xiExtension << "\n";
    // Get the eta extension of the embedded cable
    etaExtension<< 0,0;

    gsMatrix<> parameterCable = embedding_curve.coefs(); // The parameterization of the cable
    // Get the running and the fixed parameters on the patch where the cable is embedded
    if(xi_min==xi_max)  // If True, coupling is in the eta direction
    {
        gsInfo << "Coupling is in the eta direction\n";
        gsVector<> couplingRegion = etaExtension;
    }
    else // If False, coupling is in the xi direction
    {
        gsVector<> couplingRegion = xiExtension;
        // Find the correct spans for the coupled region
        int spanStart = Kv.iFind(couplingRegion.at(0)) - Kv.begin();
        int spanEnd = Kv.iFind(couplingRegion.at(1)) - Kv.begin();
        // Corresponding to the coupled region surface knot span
        // Construct the basis function for the embedded geometry


    }
}



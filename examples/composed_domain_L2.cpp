/** @file composed_domain_L2.cpp

    @brief Tutorial on how to use expression assembler to solve the Poisson equation

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): A. Mantzaflaris
*/

//! [Include namespace]
#include <gismo.h>
#include <gsNurbs/gsSquareDomain.h>

using namespace gismo;
//! [Include namespace]

int main(int argc, char *argv[])
{
    //! [Parse command line]
    bool plot = false;
    index_t numRefine  = 2;
    index_t numElevate = 0;

    gsCmdLine cmd("Tutorial on solving a Poisson problem.");
    cmd.addInt( "e", "degreeElevation",
                "Number of degree elevation steps to perform before solving (0: equalize degree in all directions)", numElevate );
    cmd.addInt( "r", "uniformRefine", "Number of Uniform h-refinement loops",  numRefine );
    cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
    //! [Parse command line]

    gsTensorBSpline<2,real_t> tbspline = *gsNurbsCreator<>::BSplineSquare();
    if (numElevate!=0)
        tbspline.degreeElevate(numElevate);

    // h-refine
    for (int r =0; r < numRefine; ++r)
        tbspline.uniformRefine();

/*    // degree elevation
    if (numElevate!=0)
        tbspline.degreeElevate(numElevate);
    // local h-refine
    for (int r =0; r < numRefine; ++r)
        tbspline.uniformRefine();

    gsTHBSpline<2, real_t>  thbgeom(tbspline);
    gsInfo << thbgeom << "\n";

    // local h-refine
    std::vector<index_t> boxes;
    boxes.resize(5);
    boxes[0] = 1;
    boxes[1] = boxes[2] = 0;
    boxes[3] = boxes[4] = 4;
    thbgeom.refineElements(boxes);

    gsWriteParaview( thbgeom , "thbgeom_refined", 1000, true);
    gsInfo << thbgeom << "\n";

    gsTHBSplineBasis<2,real_t> thbbasis = thbgeom.basis();
*/
    // The domain sigma
    gsSquareDomain<2,real_t> domain;

    gsMatrix<> pars = domain.controls();
    pars *= 0.75;
    domain.controls() = pars.col(0);
    domain.updateGeom();

    // Define a composite basis and composite geometry
    // The basis is composed by the square domain
    gsComposedBasis<real_t> cbasis(domain,tbspline.basis()); // basis(u,v) = basis(sigma(xi,eta)) -> deriv will give dphi/dxi, dphi/deta
    // The geometry is defined using the composite basis and some coefficients
    gsComposedGeometry<real_t> cgeom(cbasis,tbspline.coefs()); // G(u,v) = G(sigma(xi,eta))  -> deriv will give dG/dxi, dG/deta

    gsMultiPatch<> mp;
    mp.addPatch(cgeom);

    gsMultiBasis<> dbasis(mp,false);

    //! [Refinement]

    // Source function:
    // gsFunctionExpr<> f("((tanh(20*(x^2 + y^2)^(1/2) - 5)^2 - 1)*(20*x^2 + 20*y^2)*(40*tanh(20*(x^2 + y^2)^(1/2) - 5)*(x^2 + y^2)^(1/2) - 1))/(x^2 + y^2)^(3/2)",2);

    // Exact solution
    gsFunctionExpr<> ms("tanh((0.25-sqrt(x^2+y^2))/0.05)+1",2);

    gsBoundaryConditions<> bc;
    // bc.addCondition(boundary::side::west ,condition_type::dirichlet,&ms);
    // bc.addCondition(boundary::side::east ,condition_type::dirichlet,&ms);
    // bc.addCondition(boundary::side::south,condition_type::dirichlet,&ms);
    // bc.addCondition(boundary::side::north,condition_type::dirichlet,&ms);
    bc.setGeoMap(mp);

    //! [Problem setup]
    gsExprAssembler<> A(1,1);

    typedef gsExprAssembler<>::geometryMap geometryMap;
    typedef gsExprAssembler<>::variable    variable;
    typedef gsExprAssembler<>::space       space;
    typedef gsExprAssembler<>::solution    solution;

    // Elements used for numerical integration
    A.setIntegrationElements(dbasis);
    gsExprEvaluator<> ev(A);

    // Set the geometry map
    geometryMap G = A.getMap(mp);

    // Set the discretization space
    space u = A.getSpace(dbasis);

    // Set the source term
    // auto ff = A.getCoeff(f, G);

    // Recover manufactured solution
    auto u_ex = ev.getVariable(ms, G);

    // Solution vector and solution variable
    gsMatrix<> solVector;
    solution u_sol = A.getSolution(u, solVector);

    //! [Problem setup]

    gsSparseSolver<>::CGDiagonal solver;

    u.setup(bc, dirichlet::l2Projection, 0);

    // Initialize the system
    A.initSystem();

    gsInfo<< A.numDofs() <<std::flush;

    gsMatrix<> points(2,2);
    points.col(0).setConstant(0.25);
    points.col(1).setConstant(0.50);

    // gsDebugVar(ev.eval(u_ex,points));

    // gsDebugVar(cbasis.eval(points));
    // gsDebugVar(ev.eval(u,points.col(0)));
    // gsDebugVar(ev.eval(u,points));

    // gsDebugVar(cgeom.eval(points));
    // gsDebugVar(ev.eval(G,points.col(0)));
    // gsDebugVar(ev.eval(G,points));

    // Compute the system matrix and right-hand side
    A.assemble(
        u * u.tr() * meas(G)
        // igrad(u, G) * igrad(u, G).tr() * meas(G) //matrix
        ,
        u * u_ex * meas(G) //rhs vector
        );

    solver.compute( A.matrix() );
    solVector = solver.solve(A.rhs());

    // Compute the error
    real_t L2err = math::sqrt( ev.integral( (u_ex - u_sol).sqNorm() * meas(G) ) );
    gsInfo<<"L2 error = "<<L2err<<"\n";

    //! [Export visualization in ParaView]
    if (plot)
    {
        gsInfo<<"Plotting in Paraview...\n";

        gsParaviewCollection collection("ParaviewOutput/solution", &ev);
        collection.options().setSwitch("plotElements", true);
        collection.options().setInt("plotElements.resolution", 100);
        collection.newTimeStep(&mp);
        collection.addField(u_sol,"numerical solution");
        collection.addField(u_ex, "exact solution");
        collection.addField((u_ex-u_sol).sqNorm(), "error");
        collection.saveTimeStep();
        collection.save();


        // gsFileManager::open("ParaviewOutput/solution.pvd");
    }
    else
        gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
                  "file containing the solution.\n";
    //! [Export visualization in ParaView]

    return EXIT_SUCCESS;

}// end main

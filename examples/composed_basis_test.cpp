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
#include <gsAssembler/gsExprEvaluator.h>
#include <gsCore/gsComposedFunction.h>
#include <gsNurbs/gsSquareDomain.h>

using namespace gismo;
using namespace expr;
//! [Include namespace]

int main(int argc, char *argv[])
{

    gsFunctionExpr<> S("x*y","y^2*sqrt(x)",2);
    // gsFunctionExpr<> S("(x-0.5)^2","(y-0.5)^2",2);
    // gsSquareDomain<2,real_t> S;
    // gsMatrix<> pars = S.controls();
    // pars *= 0.95;
    // // pars(0,0) -= 0.1;
    // S.controls() = pars.col(0);
    // S.updateGeom();


    gsFunctionExpr<> G("y^(1/3)","x^(1/2)","0",2);
    gsFunctionExpr<> FXI("x*y",2);
    gsFunctionExpr<> FS("x*y",2);
    gsFunctionExpr<> FG("x*y+z",3);
    gsComposedFunction<real_t> CG({&S,&G});     // Composition G∘S
    gsComposedFunction<real_t> CFG({&S,&G,&FG});
    gsComposedFunction<real_t> CSFG({&G,&FG});
    gsComposedFunction<real_t> CFCG({&CG,&FG});

    gsKnotVector<> kv(0,1,0,3);
    gsTensorBSplineBasis<2> tbasis2(kv,kv);
    gsComposedBasis<>       cbasis2(S,tbasis2);
    gsTensorBSplineBasis<3> tbasis3(kv,kv,kv);

    gsExprAssembler<> A(1,1);
    gsMultiBasis<> mb(tbasis2);
    A.setIntegrationElements(mb);
    gsExprEvaluator<> ev(A);
    auto s = A.getMap(S);
    auto g = A.getMap(G);
    auto cg = A.getMap(CG);

    auto u  = A.getSpace(tbasis2);
    auto cu = A.getSpace(cbasis2);
    auto u3 = A.getSpace(tbasis3);

    auto JSinv  = jac(s).ginv();
    auto JGinv  = jac(g).ginv();
    auto JCGinv = jac(cg).ginv();

    gsVector<> pt(2);
    pt<<0.5,0.25;
    gsMatrix<> ptS = ev.eval(s,pt);
    gsMatrix<> ptG = ev.eval(g,ptS);
    gsMatrix<> ptCG = ev.eval(cg,pt);

    gsDebug<<"Point evaluation of the maps\n";
    gsDebug<<"(ξ,η)         = "<<pt.transpose()<<"\n";
    gsDebug<<"(u,v)         = "<<ptS.transpose()<<"\n";
    gsDebug<<"(x,y,z)       = "<<ptG.transpose()<<"\n";
    gsDebug<<"(x,y,z)       = "<<ptCG.transpose()<<"\n";
    gsDebug<<"\n";
    gsDebug<<"σ(ξ,η)        = "<<ev.eval(s,pt).transpose()<<"\n";
    gsDebug<<"G(u,v)        = "<<ev.eval(g,ptS).transpose()<<"\n";
    gsDebug<<"G(σ(ξ,η))     = "<<ev.eval(cg,pt).transpose()<<"\n";
    gsDebug<<"\n";
    gsDebug<<"Function evaluations of F(x,y,z) = x*y+z\n";
    gsDebug<<"φ(σ(ξ,η))     = "<<ev.eval(u,ptS).transpose()<<"\n";
    gsDebug<<"φ(G(σ(ξ,η)))  = "<<ev.eval(cu,pt).transpose()<<"\n";

    real_t xi = pt(0,0);
    real_t eta= pt(1,0);
    real_t U  = ptS(0,0);
    real_t V  = ptS(1,0);

    gsMatrix<> anBasis(1,9);
    anBasis.row(0)<<(1-U)*(1-U)*(1-V)*(1-V),
                    2*(1-U)*U*(1-V)*(1-V),
                    U*U*(1-V)*(1-V),
                    (1-U)*(1-U)*2*(1-V)*V,
                    4*(1-U)*U*(1-V)*V,
                    U*U*2*(1-V)*V,
                    (1-U)*(1-U)*V*V,
                    2*(1-U)*U*V*V,
                    U*U*V*V;
    gsDebug<<"φ(u,v)         = "<<anBasis<<"\n";

    gsMatrix<> anBasisDer(2,9);
    anBasisDer.row(0)<< (2*U-2)*(V-1)*(V-1),
                        -2*(2*U-1)*(V-1)*(V-1),
                        2*U*(V-1)*(V-1),
                        -2*V*(2*U-2)*(V-1),
                        4*V*(2*U-1)*(V-1),
                        -4*U*V*(V-1),
                        V*V*(2*U-2),
                        -2*V*V*(2*U-1),
                        2*U*V*V;
    anBasisDer.row(1)<< (2*V-2)*(U-1)*(U-1),
                        -U*(2*U-2)*(2*V-2),
                        U*U*(2*V-2),
                        2*(1-2*V)*(U-1)*(U-1),
                        4*U*(2*V-1)*(U-1),
                        -2*U*U*(2*V-1),
                        2*V*(U-1)*(U-1),
                        -2*U*V*(2*U-2),
                        2*U*U*V;


    gsDebug<<"\n";
    // Derivatives of f w.r.t. x,y,z
    gsDebug<<"∇φ            = \n"<<anBasisDer<<"\n";
    gsDebug<<"∇φ(σ(ξ,η))    = \n"<<ev.eval(grad(u),ptS).transpose()<<"\n";
    gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(igrad(cu,s),pt).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed

    // gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(grad(u)*JSinv,pt).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed
    // gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(grad(cu),pt).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed

    gsMatrix<> JS  = ev.eval(jac(s),pt);
    gsMatrix<> JCG = ev.eval(jac(cg),pt);

    gsDebug<<"∇s            = \n"<<JS<<"\n";
    gsDebug<<"∇CG           = \n"<<JCG<<"\n";
    gsDebug<<"∇φ            = \n"<<((JCG.transpose()*JCG).inverse()*JCG.transpose()).transpose()*JS.transpose()*anBasisDer<<"\n";
    gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(grad(u)*JGinv,ptS).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed
    gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(igrad(u,g),ptS).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed


    gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(grad(cu)*JCGinv,pt).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed
    gsDebug<<"∇φ(G(σ(ξ,η))) = \n"<<ev.eval(igrad(cu,cg),pt).transpose()<<"\n"; // fg is defined in (x,y,z), so no jac transform is needed

    /////////////
    gsFunctionExpr<> B00("(1-x)^2*(1-y)^2",2); // (1-u     )^2*(1-v     )^2
    gsComposedFunction<real_t> CB00({&S,&B00});// (1-u(ξ,η))^2*(1-v(ξ,η))^2
    auto b00  = A.getCoeff(B00,s);
    auto cb00 = A.getCoeff(CB00);

    gsDebugVar(ev.integral(b00));
    gsDebugVar(ev.integral(cb00));


    ////////////////////////////////////////////////////////////////////////////////////////////////
    gsQuadRule<> QuRule;  // Quadrature rule
    gsVector<> quWeights; // quadrature weights
    gsMatrix<> quPoints, quPointsS;
    gsMatrix<> vals;

    gsOptionList opt;
    opt.addReal("quA", "Number of quadrature points: quA*deg + quB", 1.0  );
    opt.addInt ("quB", "Number of quadrature points: quA*deg + quB", 1    );
    opt.addInt ("plot.npts", "Number of sampling points for plotting", 3000 );
    opt.addSwitch("plot.elements", "Include the element mesh in plot (when applicable)", false);
    opt.addSwitch("flipSide", "Flip side of interface where evaluation is performed.", false);
    //opt.addSwitch("plot.cnet", "Include the control net in plot (when applicable)", false);


    QuRule =  gsQuadrature::get(tbasis2, opt);

    // Initialize domain element iterator
    typename gsBasis<>::domainIter domIt = tbasis2.makeDomainIterator();

    real_t resultB00 = 0;
    real_t resultCB00 = 0;
    for (; domIt->good(); domIt->next() )
    {
        // Map the Quadrature rule to the element
        QuRule.mapTo( domIt->lowerCorner(), domIt->upperCorner(),
                      quPoints, quWeights);

        S.eval_into(quPoints,quPointsS);

        // Compute the basis on the quadrature nodes
        B00.eval_into(quPointsS,vals);
        // Compute integral
        for (index_t k = 0; k != quWeights.rows(); ++k) // loop over quad. nodes
            resultB00 += vals(0,k)*quWeights[k];

        CB00.eval_into(quPoints,vals);
        // Compute integral
        for (index_t k = 0; k != quWeights.rows(); ++k) // loop over quad. nodes
            resultCB00 += vals(0,k)*quWeights[k];

    }
    gsDebugVar(resultB00);
    gsDebugVar(resultCB00);

    gsWriteParaview(tbasis2,"tbasis2");
    gsWriteParaview(cbasis2,"cbasis2");
    return EXIT_SUCCESS;

}// end main

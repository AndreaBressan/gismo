//! [Include namespace]

#include <gismo.h>
#include <gsKLShell/gsThinShellUtils.h>

using namespace gismo;

namespace gismo{
namespace expr{

template<class T>
class curve_binormal_expr  : public _expr<curve_binormal_expr<T> >
{
    typename gsGeometryMap<T>::Nested_t _G;

public:
    typedef T Scalar;
    enum {Space = 0, ScalarValued= 0, ColBlocks= 0};

    explicit curve_binormal_expr(const gsGeometryMap<T> & G) : _G(G)
    {
        GISMO_ASSERT(_G.source().domainDim()==1 && _G.source().targetDim()==3,"curve binormal only implemented for curves with domainDim==1 and targetDim==3");
    }

    mutable gsMatrix<T> der1, der2;

    const gsMatrix<Scalar> & eval(const index_t k) const
    {
        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]

        return (der1.cross(der2)).normalized();
    }

    index_t rows() const { return  _G.source().targetDim(); }
    index_t cols() const { return 1; }


    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_G);
        _G.data().flags |= NEED_DERIV | NEED_DERIV2;
    }

    const gsFeSpace<T> & rowVar() const {return gsNullExpr<T>::get();}
    const gsFeSpace<T> & colVar() const {return gsNullExpr<T>::get();}

    void print(std::ostream &os) const { os << "binormal("; _G.print(os); os<<")"; }
};

template<class T>
class curve_normal_expr  : public _expr<curve_normal_expr<T> >
{
    typename gsGeometryMap<T>::Nested_t _G;

public:
    typedef T Scalar;
    enum {Space = 0, ScalarValued= 0, ColBlocks= 0};

    explicit curve_normal_expr(const gsGeometryMap<T> & G) : _G(G)
    {
        GISMO_ASSERT(_G.source().domainDim()==1 && _G.source().targetDim()==3,"curve normal only implemented for curves with domainDim==1 and targetDim==3");
    }

    mutable gsMatrix<T> der1, der2, binormal;

    const gsMatrix<Scalar> & eval(const index_t k) const
    {
        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]

        binormal = (der1.cross(der2)).normalized();
        return (binormal.cross(der1)).normalized();
    }

    index_t rows() const { return  _G.source().targetDim(); }
    index_t cols() const { return 1; }


    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_G);
        _G.data().flags |= NEED_DERIV | NEED_DERIV2;
    }

    const gsFeSpace<T> & rowVar() const {return gsNullExpr<T>::get();}
    const gsFeSpace<T> & colVar() const {return gsNullExpr<T>::get();}

    void print(std::ostream &os) const { os << "binormal("; _G.print(os); os<<")"; }
};

template<class E>
class curve_deriv2_expr : public _expr<curve_deriv2_expr<E> >
{
    typename E::Nested_t _u;

public:
    // enum {ColBlocks = E::rowSpan }; // ????
    enum{ Space = E::Space, ScalarValued= 0, ColBlocks = (1==E::Space?1:0) };

    typedef typename E::Scalar Scalar;

    curve_deriv2_expr(const E & u) : _u(u) { }

    mutable gsMatrix<Scalar> res, tmp;

    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }

    index_t rows() const //(components)
    {
        return _u.source().domainDim() * ( _u.source().domainDim() + 1 ) / 2; // _u.dim() for space or targetDim() for geometry
    }

    index_t cols() const
    {
        return _u.source().targetDim();
    }

    void parse(gsExprHelper<Scalar> & evList) const
    {
        _u.parse(evList);
        _u.data().flags |= NEED_DERIV2;
    }

    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
    const gsFeSpace<Scalar> & colVar() const { return _u.colVar(); }

    index_t cardinality_impl() const { return _u.cardinality_impl(); }

    void print(std::ostream &os) const { os << "deriv2("; _u.print(os); os <<")"; }

private:
    template<class U> inline
    typename util::enable_if< util::is_same<U,gsGeometryMap<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k)  const
    {
        /*
            Here, we compute the hessian of the geometry map.
            The hessian of the geometry map c has the form: hess(c)
            [d11 c1, d11 c2, d11 c3]
            [d22 c1, d22 c2, d22 c3]
            [d12 c1, d12 c2, d12 c3]

            The geometry map has components c=[c1,c2,c3]
        */
        // evaluate the geometry map of U
        res = _u.data().values[2].reshapeCol(k, rows(), cols() );
        return res;
    }
};

/**
 * @brief      Expression that takes the second derivative of an expression and multiplies it with a row vector
 *
 * @tparam     E1    Expression
 * @tparam     E2    Row vector
 */
template<class E1, class E2>
class curve_deriv2dot_expr : public _expr<curve_deriv2dot_expr<E1, E2> >
{
    typename E1::Nested_t _u;
    typename E2::Nested_t _v;

public:
    enum{   Space = (E1::Space == 1 || E2::Space == 1) ? 1 : 0,
            ScalarValued= 0,
            ColBlocks= 0
        };

    typedef typename E1::Scalar Scalar;

    curve_deriv2dot_expr(const E1 & u, const E2 & v) : _u(u), _v(v) { }

    mutable gsMatrix<Scalar> res,tmp, vEv;

    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }

    index_t rows() const
    {
        return 1; //since the product with another vector is computed
    }

    index_t cols() const
    {
        return cols_impl(_u);
    }

    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_u);   // We manage the flags of _u "manually" here (sets data)
        _u.data().flags |= NEED_DERIV2; // define flags

        _v.parse(evList); // We need to evaluate _v (_v.eval(.) is called)

        // Note: evList.parse(.) is called only in exprAssembler for the global expression
    }


    const gsFeSpace<Scalar> & rowVar() const
    {
        if      (E1::Space == 1 && E2::Space == 0)
            return _u.rowVar();
        else if (E1::Space == 0 && E2::Space == 1)
            return _v.rowVar();
        else
            return gsNullExpr<Scalar>::get();
    }

    const gsFeSpace<Scalar> & colVar() const
    {
        if      (E1::Space == 1 && E2::Space == 0)
            return _v.colVar();
        else if (E1::Space == 0 && E2::Space == 1)
            return _u.colVar();
        else
            return gsNullExpr<Scalar>::get();
    }

    void print(std::ostream &os) const { os << "deriv2("; _u.print(os); _v.print(os); os <<")"; }

private:

    template<class U> inline
    typename util::enable_if< util::is_same<U,gsGeometryMap<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k)  const
    {
        /*
            Here, we multiply the hessian of the geometry map by a vector, which possibly has multiple actives.
            The hessian of the geometry map c has the form: hess(c)
            [d11 c1, d11 c2, d11 c3]
            [d22 c1, d22 c2, d22 c3]
            [d12 c1, d12 c2, d12 c3]
            And we want to compute [d11 c .v; d22 c .v;  d12 c .v] ( . denotes a dot product and c and v are both vectors)
            So we simply evaluate for every active basis function v_k the product hess(c).v_k
        */


        // evaluate the geometry map of U
        tmp =_u.data().values[2].reshapeCol(k, cols(), _u.data().dim.second );
        vEv = _v.eval(k);
        res = vEv * tmp.transpose();
        return res;
    }

    template<class U> inline
    typename util::enable_if<util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k) const
    {
        /*
            We assume that the basis has the form v*e_i where e_i is the unit vector with 1 on index i and 0 elsewhere
            This implies that hess(v) = [hess(v_1), hess(v_2), hess(v_3)] only has nonzero entries in column i. Hence,
            hess(v) . normal = hess(v_i) * n_i (vector-scalar multiplication. The result is then of the form
            [hess(v_1)*n_1 .., hess(v_2)*n_2 .., hess(v_3)*n_3 ..]. Here, the dots .. represent the active basis functions.
        */
        const index_t numAct = u.data().values[0].rows();   // number of actives of a basis function
        const index_t cardinality = u.cardinality();        // total number of actives (=3*numAct)
        res.resize(rows()*cardinality, cols() );
        tmp.transpose() =_u.data().values[2].reshapeCol(k, cols(), numAct );
        vEv = _v.eval(k);

        gsDebugVar(_u.dim());

        gsDebugVar(tmp);

        gsDebugVar(vEv);


        for (index_t i = 0; i!=_u.dim(); i++)
            res.block(i*numAct, 0, numAct, cols() ) = tmp * vEv.at(i);

        return res;
    }

    template<class U> inline
    typename util::enable_if<util::is_same<U,gsFeSolution<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k) const
    {
        GISMO_NO_IMPLEMENTATION;
    }

    template<class U> inline
    typename util::enable_if< util::is_same<U,gsGeometryMap<Scalar> >::value, index_t >::type
    cols_impl(const U & u)  const
    {
        return _u.data().dim.second;
    }

    template<class U> inline
    typename util::enable_if< !util::is_same<U,gsGeometryMap<Scalar>  >::value, index_t >::type
    cols_impl(const U & u) const
    {
        return _u.dim();
    }
};

// CONTINUE HERE
/**
 * @brief      Eq. 52 of :
 *             Raknes, S.B. et al. 2013. “Isogeometric Rotation-Free Bending-Stabilized Cables: Statics, Dynamics, Bending Strips and Coupling with Shells.” Computer Methods in Applied Mechanics and Engineering 263: 127–43. https://www.sciencedirect.com/science/article/pii/S0045782513001229 (July 30, 2018).
 *             Equivalent to eq. 56 of that paper, but in 56 they expand the cross product using Levi-Civita Symbol
 * @tparam     E     Object type
 */


template<class E>
class curve_bvar1_expr : public _expr<curve_bvar1_expr<E> >
{
public:
    typedef typename E::Scalar Scalar;

private:
    typename E::Nested_t _u;
    typename gsGeometryMap<Scalar>::Nested_t _G;

    mutable gsMatrix<Scalar> res;
    mutable gsMatrix<Scalar> bGrad, bHess, cJac, cHess;
    mutable gsVector<Scalar,3> binormal, normal;

    mutable gsMatrix<Scalar> der1, der2;
    mutable gsMatrix<Scalar,3,3> B, I;

    mutable Scalar binormal_norm, normal_norm;

public:
    enum{ Space = E::Space, ScalarValued= 0, ColBlocks= 0};

    curve_bvar1_expr(const E & u, const gsGeometryMap<Scalar> & G) : _u(u), _G(G)
    {
        GISMO_ASSERT(G.domainDim()==1,"Domain dimension should be 1, but is "<<G.domainDim());
        GISMO_ASSERT(G.targetDim()==3,"Target dimension should be 3, but is "<<G.targetDim());
    }

#   define Eigen gsEigen
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   undef Eigen

    // helper function
    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
    {
        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
        result[pos] = val;
        return result;
    }

    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }

    index_t rows() const { return 1; }
    index_t cols() const { return _u.dim(); }

    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_u);
        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // need actives for cardinality
        evList.add(_G);
        _G.data().flags |= NEED_DERIV | NEED_DERIV2;
    }

    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
    index_t cardinality_impl() const { return _u.cardinality_impl(); }

    void print(std::ostream &os) const { os << "var1("; _u.print(os); os <<")"; }

private:

    template<class U> inline
    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k)  const
    {
        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
        res.resize(_u.cardinality(), cols()); // rows()*

        // Compute first and second derivative of the geometry map
        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
        // Compute unit bi-normals
        binormal = (der1.cross(der2));
        binormal_norm = binormal.norm();
        binormal /= binormal_norm;
        // Compute unit normals
        normal = (binormal.cross(der1));
        normal_norm = normal.norm();
        normal /= normal_norm;

        // Compute B tensor
        I.setIdentity();

        gsDebugVar(binormal * binormal.transpose());
        B = ( I - binormal * binormal.transpose() ) / binormal_norm;

        bGrad = _u.data().values[1].col(k);
        bHess = _u.data().values[2].col(k);
        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]

        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
        {
            const short_t s = d*N;
            for (index_t j = 0; j!= N; ++j) // for all actives
                res.row(s+j).noalias() = B*(
                                    vecFun(d, bGrad.at(j) ).cross( cHess.col3d(0) ) +
                                    cJac.col3d(0).cross( vecFun(d, bHess.at(j) ) )
                                   );
        }
        return res;
    }
};


//Use this one
//template<class E>
//class curve_bvar2_expr : public _expr<curve_bvar2_expr<E> >
//{
//
//
//public:
//    typedef typename E::Scalar Scalar;
//
//private:
//    typename E::Nested_t _u;
//    typename gsGeometryMap<Scalar>::Nested_t _G;
//
//    mutable gsMatrix<Scalar> res;
//    mutable gsMatrix<Scalar> bGrad, bHess, cJac, cHess, u_val;
//    mutable gsVector<Scalar,3> binormal, normal, bvar1, bvar1_without_B, bvar2;
//
//    mutable gsMatrix<Scalar> der1, der2;
//    mutable gsMatrix<Scalar,3,3> B, I, Bvar1;
//
//    mutable Scalar binormal_norm, normal_norm, binormal_norm_var1;
//
//public:
//    enum{ Space = E::Space, ScalarValued= 0, ColBlocks= 0};
//
//    curve_bvar2_expr(const E & u, const gsGeometryMap<Scalar> & G) : _u(u), _G(G)
//    {
//        GISMO_ASSERT(G.domainDim()==1,"Domain dimension should be 1, but is "<<G.domainDim());
//        GISMO_ASSERT(G.targetDim()==3,"Target dimension should be 3, but is "<<G.targetDim());
//    }
//
//#   define Eigen gsEigen
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//#   undef Eigen
//
//    // helper function
//    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
//    {
//        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
//        result[pos] = val;
//        return result;
//    }
//
//    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }
//
//    index_t rows() const { return 1; }
//    index_t cols() const { return _u.dim(); }
//
//    void parse(gsExprHelper<Scalar> & evList) const
//    {
//        evList.add(_u);
//        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2 | NEED_DIV; // need actives for cardinality
//        evList.add(_G);
//        _G.data().flags |= NEED_DERIV | NEED_DERIV2;
//    }
//
//    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
//    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
//    index_t cardinality_impl() const { return _u.cardinality_impl(); }
//
//    void print(std::ostream &os) const { os << "var2("; _u.print(os); os <<")"; }
//
//private:
//
//    template<class U> inline
//    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
//    eval_impl(const U & u, const index_t k)  const
//    {
//        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
//        res.resize(_u.cardinality(), cols()); // rows()*
//
//        // Compute first and second derivative of the geometry map
//        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
//        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
//        // Compute unit bi-normals
//        binormal = (der1.cross(der2));
//        binormal_norm = binormal.norm();
//        binormal /= binormal_norm;
//        // Compute unit normals
//        normal = (binormal.cross(der1));
//        normal_norm = normal.norm();
//        normal /= normal_norm;
//
//        // Compute B tensor
//        I.setIdentity();
//
//        gsDebugVar(binormal * binormal.transpose());
//        B = ( I - binormal * binormal.transpose() ) / binormal_norm;
//
//        u_val = _u.data().values[0].col(k);
//        bGrad = _u.data().values[1].col(k);
//        bHess = _u.data().values[2].col(k);
//        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
//        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]
//
//
//        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
//        {
//            const short_t s = d*N;
//            for (index_t j = 0; j!= N; ++j) // for all actives
//            {
//                bvar1_without_B = (
//                        vecFun(d, bGrad.at(j)).cross(cHess.col3d(0)) +
//                        cJac.col3d(0).cross(vecFun(d, bHess.at(j)))
//                        ).transpose();
//            }
//        }
//        Bvar1 = (-(bvar1_without_B * binormal.transpose()+ binormal * bvar1_without_B.transpose()) * binormal_norm - (I - binormal * binormal.transpose()) * binormal_norm_var1)/(binormal_norm * binormal_norm);
//        // JL: Not too sure about '*' between (-(bvar1_v * binormal + binormal * bvar1_v) and binormal_norm. HV derived the cross product but why???
//
//        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
//        {
//            const short_t s = d*N;
//            for (index_t j = 0; j!= N; ++j) // for all actives
////                res.row(s+j).noalias() = B * (vecFun(d, bvar1_without_B)).transpose() + Bvar1*bvar1_without_B.transpose();
//        }
//
//        return res;
//    }
//};


//template<class E1, class E2>
//class curve_bvar2_expr : public _expr<curve_bvar2_expr<E1, E2> >{
//public:
//    typedef typename E1::Scalar Scalar;
//
//private:
//    typename E1::Nested_t _u;
//    typename E2::Nested_t _v;
//    typename gsGeometryMap<Scalar>::Nested_t _G;
//
//    mutable gsMatrix<Scalar> res;
//    mutable gsMatrix<Scalar> bGradu, bGradv, bHessu, bHessv, cJac, cHess;
//    mutable gsVector<Scalar,3> binormal, normal, bvar1_v, bvar1_u, bvar1_u_without_B, bvar2;// I am sorry, please refer to appendix
//
//    mutable gsMatrix<Scalar> der1, der2;
//    mutable gsMatrix<Scalar,3,3> B, I, Bvar1;
//
//    mutable Scalar binormal_norm, normal_norm, binormal_norm_var1;
//
//public:
//    enum{ Space = 3, ScalarValued= 0, ColBlocks= 0};
//
//    curve_bvar2_expr(const E1 & u, const E2 & v, const gsGeometryMap<Scalar> & G) : _u(u), _v(v), _G(G)
//    {
//        GISMO_ASSERT(G.domainDim()==2,"Domain dimension should be 2, but is "<<G.domainDim());
//        GISMO_ASSERT(G.targetDim()==3,"Target dimension should be 3, but is "<<G.targetDim());
//    }
//
//#   define Eigen gsEigen
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//#   undef Eigen
//
//    // helper function
//    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
//    {
//        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
//        result[pos] = val;
//        return result;
//    }
//
//    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }
//
//    index_t rows() const { return 1; }
//    index_t cols() const { return _u.dim(); }
//
//    void parse(gsExprHelper<Scalar> & evList) const
//    {
//        evList.add(_u);
//        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // need actives for cardinality
//        evList.add(_v);
//        _v.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // I forgot what this does
//        evList.add(_G);
//        _G.data().flags |= NEED_DERIV | NEED_DERIV2;
//    }
//
//    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
//    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
//    index_t cardinality_impl() const { return _u.cardinality_impl(); }
//
//    void print(std::ostream &os) const { os << "var1("; _u.print(os); os <<")"; }
//
//private:
//
//    template<class U> inline
//    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
//    eval_impl(const U & u, const index_t k)  const
//    {
//        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
//        const index_t N2 = _v.cadinality()/_v.dim(); // _v.data().actives.rows()
//        res.resize(_u.cardinality(), _v.cadinality()); // rows()*
//
//        const index_t cardU = _u.data().values[0].rows(); // number of actives per component of u
//        const index_t cardV = _v.data().values[0].rows(); // number of actives per component of v
//
//
//        // Compute first and second derivative of the geometry map
//        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
//        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
//        // Compute unit bi-normals
//        binormal = (der1.cross(der2));
//        binormal_norm = binormal.norm();
//        binormal /= binormal_norm;
//        // Compute unit normals
//        normal = (binormal.cross(der1));
//        normal_norm = normal.norm();
//        normal /= normal_norm;
//
//        bGradu = _u.data().values[1].col(k);
//        bHessu = _u.data().values[2].col(k);
//        bGradv = _v.data().values[1].col(k);
//        bHessv = _v.data().values[2].col(k);
//        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
//        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]
//
//
//        // Compute B tensor
//        I.setIdentity();
//
//        gsDebugVar(binormal * binormal.transpose());
//        B = ( I - binormal * binormal.transpose() ) / binormal_norm;
//
//        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
//        {
//            for (index_t j = 0; j != cardU; ++j) //for all basis functions v(1)
//            {
//                bvar1_u_without_B =  vecFun(d, bGradu.at(j) ).cross( cHess.col3d(0) ) +
//                                     cJac.col3d(0).cross( vecFun(d, bHessu.at(j) ) ).transpose();
//                bvar1_u = B*(
//                        vecFun(d, bGradu.at(j) ).cross( cHess.col3d(0) ) +
//                        cJac.col3d(0).cross( vecFun(d, bHessu.at(j) ) )
//                ).transpose();
//
//            }
//
//            for (index_t i = 0; i != cardV; ++i) //for all basis functions v(1)
//            {
//                binormal_norm_var1 = (vecFun(d, bGradv.at(i)).cross(cHess.col3d(0))) +
//                                    cJac.col3d(0).cross(vecFun(d, bHessv.at(i))) * binormal/binormal_norm;
//                bvar1_v = B*(
//                        vecFun(d, bGradv.at(i) ).cross( cHess.col3d(0) ) +
//                        cJac.col3d(0).cross( vecFun(d, bHessv.at(i) ) )
//                ).transpose();
//            }
//
//        }
//        Bvar1 = (-(bvar1_v * binormal + binormal * bvar1_v) * binormal_norm - (I - binormal*binormal) * binormal_norm_var1)/binormal_norm^2;
//        // JL: Not too sure about '*' between (-(bvar1_v * binormal + binormal * bvar1_v) and binormal_norm. HV derived the cross product but why???
//
//        // Compute B tensor variation
//
////        bvar2 = B * (vecFun(d, grad(bvar1_u))) + Bvar1*bvar1_v;
//        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
//        {
//            for (index_t i = 0; i != cardV; ++i) //for all basis functions v(1)
//            {
//                res = B * (vecFun(d, grad(bvar1_u))) + Bvar1*bvar1_v;
//            }
//
//        }
//        return res;
//    }
//};


template<class E>
class curve_nvar1_expr : public _expr<curve_nvar1_expr<E> >
{
public:
    typedef typename E::Scalar Scalar;

private:
    typename E::Nested_t _u;
    typename gsGeometryMap<Scalar>::Nested_t _G;

    mutable gsMatrix<Scalar> res;
    mutable gsMatrix<Scalar> bGrad, bHess, cJac, cHess;
    mutable gsVector<Scalar,3> binormal, normal, bvar1;

    mutable gsMatrix<Scalar> der1, der2;
    mutable gsMatrix<Scalar,3,3> A, B, I;

    mutable Scalar binormal_norm, normal_norm;

public:
    enum{ Space = E::Space, ScalarValued= 0, ColBlocks= 0};

    curve_nvar1_expr(const E & u, const gsGeometryMap<Scalar> & G) : _u(u), _G(G)
    {
        GISMO_ASSERT(_G.data().dim.first==1,"Domain dimension should be 1, but is "<<_G.data().dim.first);
        GISMO_ASSERT(_G.data().dim.second==3,"Target dimension should be 3, but is "<<_G.data().dim.second);
    }

#   define Eigen gsEigen
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   undef Eigen

    // helper function
    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
    {
        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
        result[pos] = val;
        return result;
    }

    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }

    index_t rows() const { return 1; }
    index_t cols() const { return _u.dim(); }

    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_u);
        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // need actives for cardinality
        evList.add(_G);
        _G.data().flags |= NEED_DERIV | NEED_2ND_DER;
    }

    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
    index_t cardinality_impl() const { return _u.cardinality_impl(); }

    void print(std::ostream &os) const { os << "var1("; _u.print(os); os <<")"; }

private:

    template<class U> inline
    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k)  const
    {
        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
        res.resize(_u.cardinality(), cols()); // rows()*

        // Compute first and second derivative of the geometry map
        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
        // Compute unit bi-normals
        binormal = (der1.cross(der2));
        binormal_norm = binormal.norm();
        binormal /= binormal_norm;
        // Compute unit normals
        normal = (binormal.cross(der1));
        normal_norm = normal.norm();
        normal /= normal_norm;

        // Compute B tensor
        I.setIdentity();

        gsDebugVar(binormal * binormal.transpose());
        B = ( I - binormal * binormal.transpose() ) / binormal_norm;
        A = ( I - normal * normal.transpose() ) / normal_norm;


        bGrad = _u.data().values[1].col(k);
        bHess = _u.data().values[2].col(k);
        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]

        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
        {
            const short_t s = d*N;
            for (index_t j = 0; j!= N; ++j) // for all actives
            {
                bvar1 = B*(
                            vecFun(d, bGrad.at(j) ).cross( cHess.col3d(0) ) +
                            cJac.col3d(0).cross( vecFun(d, bHess.at(j) ) )
                           ).transpose();

                res.row(s+j).noalias() = A * (
                                                bvar1.cross( cJac.col(0) ) +
                                                binormal.cross( vecFun(d, bGrad.at(j) ) )
                                              );

            }
        }
        return res;
    }
};

template<class E>
class curve_Avar1_expr : public _expr<curve_Avar1_expr<E> >
{
public:
    typedef typename E::Scalar Scalar;

private:
    typename E::Nested_t _u;
    typename gsGeometryMap<Scalar>::Nested_t _G;

    mutable gsMatrix<Scalar> res;
    mutable gsMatrix<Scalar> bGrad, bHess, cJac, cHess;
    mutable gsVector<Scalar,3> binormal, normal, bvar1, bvar1_without_B, bvar2, nvar1, normal_norm_var1;

    mutable gsMatrix<Scalar> der1, der2;
    mutable gsMatrix<Scalar,3,3> A, B, I, Bvar1, Avar1;

    mutable Scalar binormal_norm, normal_norm; //binormal_norm_var1, normal_norm_var1;

public:
    enum{ Space = E::Space, ScalarValued= 0, ColBlocks= 0};

    curve_Avar1_expr(const E & u, const gsGeometryMap<Scalar> & G) : _u(u), _G(G)
    {
        GISMO_ASSERT(_G.data().dim.first==1,"Domain dimension should be 1, but is "<<_G.data().dim.first);
        GISMO_ASSERT(_G.data().dim.second==3,"Target dimension should be 3, but is "<<_G.data().dim.second);
    }

#   define Eigen gsEigen
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   undef Eigen

    // helper function
    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
    {
        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
        result[pos] = val;
        return result;
    }

    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }

    index_t rows() const { return 1; }
    index_t cols() const { return _u.dim(); }

    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_u);
        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // need actives for cardinality
        evList.add(_G);
        _G.data().flags |= NEED_DERIV | NEED_2ND_DER;
    }

    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
    index_t cardinality_impl() const { return _u.cardinality_impl(); }

    void print(std::ostream &os) const { os << "var1("; _u.print(os); os <<")"; }

private:

    template<class U> inline
    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k)  const
    {
        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
        res.resize(_u.cardinality(), cols()); // rows()*

        // Compute first and second derivative of the geometry map
        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
        // Compute unit bi-normals
        binormal = (der1.cross(der2));
        binormal_norm = binormal.norm();
        binormal /= binormal_norm;
        // Compute unit normals
        normal = (binormal.cross(der1));
        normal_norm = normal.norm();
        normal /= normal_norm;

        // Compute B tensor
        I.setIdentity();

        gsDebugVar(binormal * binormal.transpose());
        B = ( I - binormal * binormal.transpose() ) / binormal_norm;
        A = ( I - normal * normal.transpose() ) / normal_norm;


        bGrad = _u.data().values[1].col(k);
        bHess = _u.data().values[2].col(k);
        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]

        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
        {
            const short_t s = d*N;
            for (index_t j = 0; j!= N; ++j) // for all actives
            {
                bvar1 = B*(
                        vecFun(d, bGrad.at(j) ).cross( cHess.col3d(0) ) +
                        cJac.col3d(0).cross( vecFun(d, bHess.at(j) ) )
                ).transpose();

                nvar1 = A * (
                        bvar1.cross( cJac.col(0) ) +
                        binormal.cross( vecFun(d, bGrad.at(j) ) )
                );

                normal_norm_var1 = (
                        bvar1.cross( cJac.col(0) ) +
                        binormal.cross( vecFun(d, bGrad.at(j) ) )
                )*normal/normal_norm;

                Avar1 = (-(nvar1 * normal.transpose() + normal * nvar1.transpose()) * normal_norm - (I - normal*normal.transpose()) * normal_norm_var1.transpose())/(normal_norm* normal_norm);
            }
        }
        return Avar1;
    }
};
//template<class E1, class E2>
//class curve_nvar2_expr : public _expr<curve_nvar2_expr<E1, E2> >{
//public:
//    typedef typename E1::Scalar Scalar;
//
//private:
//    typename E1::Nested_t _u;
//    typename E2::Nested_t _v;
//    typename gsGeometryMap<Scalar>::Nested_t _G;
//
//    mutable gsMatrix<Scalar> res;
//    mutable gsMatrix<Scalar> bGradu, bGradv, bHessu, bHessv, cJac, cHess;
//    mutable gsVector<Scalar,3> binormal, normal, bvar1_v, bvar1_u, bvar1_u_without_B, bvar2, nvar1;// I am sorry, please refer to appendix
//
//    mutable gsMatrix<Scalar> der1, der2;
//    mutable gsMatrix<Scalar,3,3> A, B, I, Bvar1, Avar1;
//
//    mutable Scalar binormal_norm, normal_norm, binormal_norm_var1, normal_norm_var1;
//
//public:
//    enum{ Space = 3, ScalarValued= 0, ColBlocks= 0};
//
//    curve_nvar2_expr(const E1 & u, const E2 & v, const gsGeometryMap<Scalar> & G) : _u(u), _v(v), _G(G)
//    {
//        GISMO_ASSERT(G.domainDim()==2,"Domain dimension should be 2, but is "<<G.domainDim());
//        GISMO_ASSERT(G.targetDim()==3,"Target dimension should be 3, but is "<<G.targetDim());
//    }
//
//#   define Eigen gsEigen
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//#   undef Eigen
//
//    // helper function
//    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
//    {
//        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
//        result[pos] = val;
//        return result;
//    }
//
//    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }
//
//    index_t rows() const { return 1; }
//    index_t cols() const { return _u.dim(); }
//
//    void parse(gsExprHelper<Scalar> & evList) const
//    {
//        evList.add(_u);
//        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // need actives for cardinality
//        evList.add(_v);
//        _v.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // I forgot what this does
//        evList.add(_G);
//        _G.data().flags |= NEED_DERIV | NEED_DERIV2;
//    }
//
//    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
//    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
//    index_t cardinality_impl() const { return _u.cardinality_impl(); }
//
//    void print(std::ostream &os) const { os << "var1("; _u.print(os); os <<")"; }
//
//private:
//
//    template<class U> inline
//    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
//    eval_impl(const U & u, const index_t k)  const
//    {
////        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
////        const index_t N2 = _v.cadinality()/_v.dim(); // _v.data().actives.rows()
//        res.resize(rows()*_v.cadinality(), cols()); // total number of actives
//
//        const index_t cardU = _u.data().values[0].rows(); // number of actives per component of u
//        const index_t cardV = _v.data().values[0].rows(); // number of actives per component of v
//
//
//        // Compute first and second derivative of the geometry map
//        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
//        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
//        // Compute unit bi-normals
//        binormal = (der1.cross(der2));
//        binormal_norm = binormal.norm();
//        binormal /= binormal_norm;
//        // Compute unit normals
//        normal = (binormal.cross(der1));
//        normal_norm = normal.norm();
//        normal /= normal_norm;
//
//        bGradu = _u.data().values[1].col(k);
//        bHessu = _u.data().values[2].col(k);
//        bGradv = _v.data().values[1].col(k);
//        bHessv = _v.data().values[2].col(k);
//        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
//        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]
//
//
//        // Compute B tensor
//        I.setIdentity();
//
//        gsDebugVar(binormal * binormal.transpose());
//        A = ( I - normal * normal.transpose() ) / normal_norm;
//        B = ( I - binormal * binormal.transpose() ) / binormal_norm;
//
//
//        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
//        {
//            for (index_t j = 0; j != cardU; ++j) //for all basis functions v(1)
//            {
//                bvar1_u_without_B =  vecFun(d, bGradu.at(j) ).cross( cHess.col3d(0) ) +
//                                     cJac.col3d(0).cross( vecFun(d, bHessu.at(j) ) ).transpose();
//                bvar1_u = B*(
//                        vecFun(d, bGradu.at(j) ).cross( cHess.col3d(0) ) +
//                        cJac.col3d(0).cross( vecFun(d, bHessu.at(j) ) )
//                ).transpose();
//                nvar1 = A * (
//                        bvar1_u.cross( cJac.col(0) ) +
//                        binormal.cross( vecFun(d, bGradu.at(j) ) )
//                );
//
//            }
//
//            for (index_t i = 0; i != cardV; ++i) //for all basis functions v(1)
//            {
//                binormal_norm_var1 = (vecFun(d, bGradv.at(i)).cross(cHess.col3d(0))) +
//                                     cJac.col3d(0).cross(vecFun(d, bHessv.at(i))) * binormal/binormal_norm;
//                normal_norm_var1 = bvar1_v.cross(cJac.col3d(0)) + binormal.cross(bGradv.at(i)) * normal/normal_norm;
//                bvar1_v = B*(
//                        vecFun(d, bGradv.at(i) ).cross( cHess.col3d(0) ) +
//                        cJac.col3d(0).cross( vecFun(d, bHessv.at(i) ) )
//                ).transpose();
//            }
//
//        }
//        Avar1 = (-(nvar1*normal + normal*nvar1)*normal_norm - (I - normal*normal)*normal_norm_var1)/normal_norm^2;
//        Bvar1 = (-(bvar1_v * binormal + binormal * bvar1_v) * binormal_norm - (I - binormal*binormal) * binormal_norm_var1)/binormal_norm^2;
//        // JL: Not too sure about '*' between (-(bvar1_v * binormal + binormal * bvar1_v) and binormal_norm. HV derived the cross product but why???
//
//        // Compute B tensor variation
//
////        bvar2 = B * (vecFun(d, grad(bvar1_u))) + Bvar1*bvar1_v;
//        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
//        {
//            for (index_t i = 0; i != cardV; ++i) //for all basis functions v(1)
//            {
//                bvar2 = B * (vecFun(d, grad(bvar1_u))) + Bvar1*bvar1_v;
//                res = A * (vecFun(d, grad(nvar1))) +  Avar1*nvar1;
//            }
//        }
//        return res;
//    }
//};


template<class E>
class curve_nvar2_expr : public _expr<curve_nvar2_expr<E> >
{
public:
    typedef typename E::Scalar Scalar;

private:
    typename E::Nested_t _u;
    typename gsGeometryMap<Scalar>::Nested_t _G;

    mutable gsMatrix<Scalar> res;
    mutable gsMatrix<Scalar> bGrad, bHess, cJac, cHess;
    mutable gsVector<Scalar,3> binormal, normal, bvar1, bvar1_without_B, bvar2, nvar1;

    mutable gsMatrix<Scalar> der1, der2;
    mutable gsMatrix<Scalar,3,3> A, B, I, Bvar1, Avar1;

    mutable Scalar binormal_norm, normal_norm, binormal_norm_var1, normal_norm_var1;

public:
    enum{ Space = E::Space, ScalarValued= 0, ColBlocks= 0};

    curve_nvar2_expr(const E & u, const gsGeometryMap<Scalar> & G) : _u(u), _G(G)
    {
        GISMO_ASSERT(_G.data().dim.first==1,"Domain dimension should be 1, but is "<<_G.data().dim.first);
        GISMO_ASSERT(_G.data().dim.second==3,"Target dimension should be 3, but is "<<_G.data().dim.second);
    }

#   define Eigen gsEigen
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   undef Eigen

    // helper function
    static inline gsVector<Scalar,3> vecFun(index_t pos, Scalar val)
    {
        gsVector<Scalar,3> result = gsVector<Scalar,3>::Zero();
        result[pos] = val;
        return result;
    }

    const gsMatrix<Scalar> & eval(const index_t k) const {return eval_impl(_u,k); }

    index_t rows() const { return 1; }
    index_t cols() const { return _u.dim(); }

    void parse(gsExprHelper<Scalar> & evList) const
    {
        evList.add(_u);
        _u.data().flags |= NEED_ACTIVE | NEED_GRAD | NEED_DERIV2; // need actives for cardinality
        evList.add(_G);
        _G.data().flags |= NEED_DERIV | NEED_2ND_DER;
    }

    const gsFeSpace<Scalar> & rowVar() const { return _u.rowVar(); }
    const gsFeSpace<Scalar> & colVar() const {return gsNullExpr<Scalar>::get();}
    index_t cardinality_impl() const { return _u.cardinality_impl(); }

    void print(std::ostream &os) const { os << "var1("; _u.print(os); os <<")"; }

private:

    template<class U> inline
    typename util::enable_if< util::is_same<U,gsFeSpace<Scalar> >::value, const gsMatrix<Scalar> & >::type
    eval_impl(const U & u, const index_t k)  const
    {
        const index_t N = _u.cardinality()/_u.dim(); // _u.data().actives.rows()
        res.resize(_u.cardinality(), cols()); // rows()*

        // Compute first and second derivative of the geometry map
        der1 = _G.data().values[1].col(k); // [dG_1/dxi1, dG_2/dxi1, dG_3/dxi1]
        der2 = _G.data().values[2].col(k); // [d^2 G_1/d^2 xi1, d^2 G_2/d^2 xi1, d^2 G_3/d^2 xi1]
        // Compute unit bi-normals
        binormal = (der1.cross(der2));
        binormal_norm = binormal.norm();
        binormal /= binormal_norm;
        // Compute unit normals
        normal = (binormal.cross(der1));
        normal_norm = normal.norm();
        normal /= normal_norm;

        // Compute B tensor
        I.setIdentity();

        gsDebugVar(binormal * binormal.transpose());
        B = ( I - binormal * binormal.transpose() ) / binormal_norm;
        A = ( I - normal * normal.transpose() ) / normal_norm;


        bGrad = _u.data().values[1].col(k);
        bHess = _u.data().values[2].col(k);
        cJac = _G.data().values[1].reshapeCol(k, 1, 3).transpose();  // [dG1/dx;     dG2/dx;     dG3/dx]
        cHess = _G.data().values[2].reshapeCol(k, 1, 3).transpose(); // [d^2G1/dx^2; d^2G2/dx^2; d^2G3/dx^2]

        for (index_t d = 0; d!= cols(); ++d) // for all basis function components [Target dim]
        {
            const short_t s = d*N;
            for (index_t j = 0; j!= N; ++j) // for all actives
            {
                bvar1 = B*(
                        vecFun(d, bGrad.at(j) ).cross( cHess.col3d(0) ) +
                        cJac.col3d(0).cross( vecFun(d, bHess.at(j) ) )
                ).transpose();

                res.row(s+j).noalias() = A * (
                        bvar1.cross( cJac.col(0) ) +
                        binormal.cross( vecFun(d, bGrad.at(j) ) )
                );

            }
        }
        return res;
    }
};

template<class T> EIGEN_STRONG_INLINE
curve_binormal_expr<T> binormal(const gsGeometryMap<T> & G)
{ return curve_binormal_expr<T>(G); }


template<class T> EIGEN_STRONG_INLINE
curve_normal_expr<T> normal(const gsGeometryMap<T> & G)
{ return curve_normal_expr<T>(G); }

template<class E> EIGEN_STRONG_INLINE
curve_deriv2_expr<E> cderiv2(const E & u)
{ return curve_deriv2_expr<E>(u); }

template<class E1, class E2> EIGEN_STRONG_INLINE
curve_deriv2dot_expr<E1, E2> cderiv2dot(const E1 & u, const E2 & v) { return curve_deriv2dot_expr<E1, E2>(u,v); }

/// Curve Bi-Normal (CBN) first variation (var1)
template<class E> EIGEN_STRONG_INLINE
curve_bvar1_expr<E> cbnvar1(const E & u, const gsGeometryMap<typename E::Scalar> & G) { return curve_bvar1_expr<E>(u,G); }

/// Curve Normal (CN) first variation (var1)
template<class E> EIGEN_STRONG_INLINE
curve_nvar1_expr<E> cnvar1(const E & u, const gsGeometryMap<typename E::Scalar> & G) { return curve_nvar1_expr<E>(u,G); }

template<class E> EIGEN_STRONG_INLINE
curve_Avar1_expr<E> Avar1(const E & u, const gsGeometryMap<typename E::Scalar> & G) { return curve_Avar1_expr<E>(u,G); }

/// Curve Normal (CN) first variation (var1)
//template<class E> EIGEN_STRONG_INLINE
//curve_bvar2_expr<E> cbvar2(const E & u, const gsGeometryMap<typename E::Scalar> & G) { return curve_bvar2_expr<E>(u,G); }


}
}

int main(int argc, char *argv[])
{
    //! [Parse command line]
    bool plot = false;
    index_t numRefine  = 1;
    index_t numElevate = 1;
    index_t testCase = 1;
    bool weak = false;
    bool nonlinear = false;
    std::string fn;

    real_t E_modulus = 1.0;
    real_t PoissonRatio = 0.0;
    real_t thickness = 1.0;

    gsCmdLine cmd("Tutorial on solving a Poisson problem.");
    cmd.addInt( "e", "degreeElevation",
                "Number of degree elevation steps to perform before solving (0: equalize degree in all directions)", numElevate );
    cmd.addInt( "r", "uniformRefine", "Number of Uniform h-refinement steps to perform before solving",  numRefine );
    cmd.addInt( "t", "testCase", "Test case to run: 1 = unit square; 2 = Scordelis Lo Roof",  testCase );
    cmd.addSwitch("nl", "Solve nonlinear problem", nonlinear);
    cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);
    cmd.addSwitch("weak", "Weak BCs", weak);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
    //! [Parse command line]

    //! [set test case data]
    gsMultiPatch<> mp_ori;
    gsMultiPatch<> mp_def;

    gsKnotVector<> KV(0,1,0,3) ;
    gsBSplineBasis<> bbasis(KV);
    gsMatrix<> coefs_ori(3,2), coefs_def(3,2);

    coefs_ori.row(0)<<0,1;
    coefs_ori.row(1)<<1,1;
    coefs_ori.row(2)<<2,1;

    mp_ori.addPatch(bbasis.makeGeometry(give(coefs_ori)));
    mp_def = mp_ori;

    //! [Refinement]
    // p-refine
    if (numElevate!=0)
        mp_ori.degreeElevate(numElevate);

    // h-refine
    for (int r =0; r < numRefine; ++r)
        mp_ori.uniformRefine();
    //! [Refinement]

    gsMultiBasis<> mb(mp_ori);

    gsBoundaryConditions<> bc;
    bc.addCondition(boundary::west, condition_type::dirichlet, 0, 0, false, -1 ); // unknown 0 - x
    bc.addCondition(boundary::east, condition_type::dirichlet, 0, 0, false, -1 ); // unknown 1 - y
    bc.setGeoMap(mp_ori);

    // Cross sectional parameters
    real_t b = 0.1;
    real_t h = 0.2;
    real_t E = 1.0;

    // Make expression assembler
    gsExprAssembler<> assembler(1,1);
    // Defines the Gauss rule (based on the basis mb)
    assembler.setIntegrationElements(mb);

    // Make expressiom evaluator
    gsExprEvaluator<> ev(assembler);

    gsVector<> pt(1);
    pt<<0.5;

    gsMatrix<> solVector;

    auto X = assembler.getMap(mp_ori);
    auto x = assembler.getMap(mp_def);
    auto u = assembler.getSpace(mb,2);
    auto u_sol = assembler.getSolution(u,solVector);
    gsFunctionExpr<> force("0","x",2);
    auto ff= assembler.getCoeff(force,X); // evaluates in the physical domain

    // Assign variables to the Expression Assembler
    gsConstantFunction<> Area(b*h,2);
    auto area= assembler.getCoeff(Area,X); // evaluates in the physical domain
    gsConstantFunction<> Inertia(b*h*h*h/12.,2);
    auto inertia= assembler.getCoeff(Inertia,X); // evaluates in the physical domain
    gsConstantFunction<> Youngs(E,2);
    auto youngs= assembler.getCoeff(Youngs,X); // evaluates in the physical domain

    // G: R^1 -> R^3
    // jac(G): [dG_x/dxi,dG_y/dxi,dG_z/dxi]

    // NOTE: sn does not work for domainDim=1 && targetDim=3...
    // sn: surface normal vector
    // gsDebugVar(ev.eval(deriv2(u,sn(x).normalized().tr()),pt));

    // gsDebugVar(ev.eval(deriv2(x,var1(u,x)),pt));


    // Second variation of the normal (var2)
    // gsDebugVar(ev.eval(var2(u,u,x,deriv2(x)),pt));
    // Assembly
    // Initialize the system
    u.setup(bc, dirichlet::interpolation, 0);
    // Initialize LHS, RHS
    assembler.initSystem();

    gsInfo<<"Number of degrees of freedom: "<< assembler.numDofs() <<"\n"<<std::flush;

    /*
     We provide the following functions:
     E_m  membrane strain tensor.       [Works]
     E_m_der first variation of E_m     [Works]
     E_m_der2 second variation of E_m   [Works]

     E_b bending strain tensor.         [Works]
     E_b_der first variation of E_b     [Needs bvar1 (nvar1)]
     E_b_der2 second variation of E_b   [Needs bvar1 (nvar1), bvar2 (nvar2)]

     Where:
     x the deformed configuration
     X the undeformed configuration
     G the contravariant basis vector
     */


    GISMO_ASSERT(mp_ori.targetDim()==2,"Target dimension must be 2");
    auto normalX = sn(X);
    auto normalx = sn(x);

//
//     gsDebugVar(ev.eval(cbnvar1(u,X),pt));
    // gsDebugVar(ev.eval(cnvar1(u,X),pt));

//    gsDebugVar(ev.eval(Avar1(u, X),pt));

    // cbnvar2dot(u,v,deriv2(G));
    // cbnvar2dot(u,v,G);

    // GISMO_ASSERT(mp_ori.targetDim()==3,"Target dimension must be 3");
    // auto normalX = normal(X);
    // auto normalx = normal(x);
    auto E_m = jac(x).tr()*jac(x) - jac(X).tr()*jac(X); // HV: why no 1/2? maybe you mean x in front of the equation
    auto E_b = cderiv2(x)*normalx - cderiv2(X)*normalX;
    auto S_m = area*youngs*E_m;
    // gsDebugVar(ev.eval(S_m,pt));
    auto S_b = inertia*youngs*E_b;
    // gsDebugVar(ev.eval(S_b,pt));


    auto E_m_der = jac(x).tr() * jac(u);
    // gsDebugVar(ev.eval(cderiv2dot(u,normalx.normalized().tr()),pt));
//     gsDebugVar(ev.eval(cderiv2dot(x,var1(u,x)),pt));

    auto E_b_der = deriv2(u,normalx.normalized().tr()) + deriv2(x,var1(u,x));
    // gsDebugVar(ev.eval(E_b_der,pt));

    auto S_m_der = area.val()*youngs.val()*E_m_der;
    auto S_b_der = inertia*youngs*E_b_der;
    // gsDebugVar(ev.eval(S_b_der,pt));

    auto G1 = jac(x); // HV: correct??, maybe replace by measure/ JL: should be binormal vector
    auto E_m_der2 = 0.5 * (E_m.val() * (jac(u).cwisetr() * jac(u).cwisetr().tr()) );
    auto S_m_der2 = (area*youngs).val()*E_m_der2;

    //JL: Added deriv2(u,sn(x).normalized())

    /*
     The following is the force vector:
     F_m: Membrane force vector
     F_b: Bending force vector

     F_int = F_m + F_b
    */
    auto F_ext = u*ff*area.val();

    auto F_m = 0.5 * E_m.val() * S_m_der.nocb();
    auto F_b = E_b.val() * S_b_der.nocb();


    gsDebugVar(ev.eval(E_m * S_m_der,pt));
    gsDebugVar(ev.eval(E_m * S_m_der.cwisetr(),pt));
    gsDebugVar(ev.eval(E_m * S_m_der.tr(),pt));
    gsDebugVar(ev.eval(E_m * S_m_der.cwisetr().tr(),pt));
    gsDebugVar(ev.eval(S_m_der.nocb(),pt));
    gsDebugVar(ev.eval(E_m,pt));
    // gsDebugVar(ev.eval(E_m.val() * S_m_der.nocb(),pt));

    gsDebugVar((E_m.val() * S_m_der.nocb()).rows());
    gsDebugVar((E_m.val() * S_m_der.nocb()).cols());
    // gsDebugVar((E_m.val() * S_m_der.nocb()).rows());


    // gsDebugVar(ev.eval(E_m * S_m_der.nocb(),pt));

    // gsDebugVar((E_m * S_m_der).ColBlocks);
    // gsDebugVar((E_m * S_m_der.cwisetr()).ColBlocks);
    // gsDebugVar((E_m * S_m_der.tr()).ColBlocks);
    // gsDebugVar((E_m * S_m_der.cwisetr().tr()).ColBlocks);
    // gsDebugVar((E_m * S_m_der.trace()).ColBlocks);

    // gsDebugVar(ev.eval(E_m.val() * S_m_der,pt));
    // gsDebugVar(ev.eval(E_m.val() * S_m_der.cwisetr(),pt));
    // gsDebugVar(ev.eval(E_m.val() * S_m_der.tr(),pt));
    // gsDebugVar(ev.eval(E_m.val() * S_m_der.cwisetr().tr(),pt));



    gsDebugVar(ev.eval(F_m,pt));
    gsDebugVar(ev.eval(F_m.cwisetr().tr(),pt));

    gsDebugVar(F_m.rows());
    gsDebugVar(F_m.cols());

    assembler.assemble(F_m);
    gsDebugVar(assembler.matrix());

    // Assemble K_m (linear)
    gsDebug<<"1\n";
    assembler.assemble( S_m_der * E_m_der.tr() * G1.norm() * G1.norm() );
    // Assemble K_m (nonlinear)
    gsDebug<<"2\n";
    assembler.assemble( S_m_der2 * G1.norm() * G1.norm() );

/*
    // assembler.assemble( S_m_der2 * G1.norm() * G1.norm() );

    // Assemble K_b (linear)
    // assembler.assemble( S_b_der * E_b_der.tr() * G1.norm() * G1.norm() );
    // Assemble K_b (nonlinear)
    // assembler.assemble(inertia * youngs * E_b.tr() * E_b_der() * G1.norm() * G1.norm());

    // Assemble F_int
    gsDebug<<"4\n";
    assembler.assemble( (
                            F_m
                            // +
                            // F_b
                            ) * G1.norm() * G1.norm()
                        );
*/

    // Assemble F_ext
    gsDebug<<"3\n";
    assembler.assemble( F_ext * G1.norm() * G1.norm() );

    // Assemble F_int
//    gsDebug<<"4\n";
//    assembler.assemble( (
//                                F_m
//                                // +
//                                // F_b
//                        ) * G1.norm() * G1.norm()
//    );

    //! [Linear solve]
    gsSparseMatrix<> K(assembler.numDofs(),assembler.numDofs());
    K.setIdentity();
    K *= 1e-6;
    K += assembler.matrix();

    gsDebugVar(K.toDense());

    gsSparseSolver<>::CGDiagonal solver(K);

    gsDebugVar(assembler.rhs());
    solVector = solver.solve(assembler.rhs());

    mp_def = mp_ori;
    gsMatrix<> cc;
    for ( size_t k =0; k!=mp_ori.nPatches(); ++k) // Deform the geometry
    {
        u_sol.extract(cc, k);
        mp_def.patch(k).coefs() += cc;  // defG points to mp_def, therefore updated
    }
    //! [Linear solve]

    //! [Nonlinear solve]
    real_t residual = assembler.rhs().norm();
    real_t residual0 = residual;
    real_t residualOld = residual;
    gsMatrix<> updateVector = solVector;
    if (nonlinear)
    {
        index_t itMax = 100;
        real_t tol = 1e-8;
        for (index_t it = 0; it != itMax; ++it)
        {
            assembler.initSystem();

            assembler.assemble( S_m_der * E_m_der.tr() * G1.norm() * G1.norm());
            assembler.assemble( S_m_der2 * G1.norm() * G1.norm() );
            assembler.assemble( F_ext * G1.norm() * G1.norm() );
            assembler.assemble(  -F_m * G1.norm() * G1.norm() );
            gsDebug<<"Finished\n";


            // solve system
            solver.compute( assembler.matrix() );
            updateVector = solver.solve(assembler.rhs()); // this is the UPDATE


            solVector += updateVector;
            residual = assembler.rhs().norm();

            gsInfo<<"Iteration: "<< it
                   <<", residue: "<< residual
                   <<", update norm: "<<updateVector.norm()
                   <<", log(Ri/R0): "<< math::log10(residualOld/residual0)
                   <<", log(Ri+1/R0): "<< math::log10(residual/residual0)
                   <<"\n";

            residualOld = residual;

            // update deformed patch
            u_sol.setSolutionVector(updateVector);
            for ( size_t k =0; k!=mp_def.nPatches(); ++k) // Deform the geometry
            {
                // // extract deformed geometry
                u_sol.extract(cc, k);
                mp_def.patch(k).coefs() += cc;  // defG points to mp_def, therefore updated
            }

            if (residual < tol)
                break;
        }
    }
    //! [Nonlinear solve]
    gsDebugVar(solVector);
    gsDebugVar(updateVector);

    //! [Construct solution]
    u_sol.setSolutionVector(solVector);
    mp_def = mp_ori;
    for ( size_t k =0; k!=mp_ori.nPatches(); ++k) // Deform the geometry
    {
        u_sol.extract(cc, k);
        mp_def.patch(k).coefs() += cc;  // defG points to mp_def, therefore updated
    }

    gsMultiPatch<> deformation = mp_def;
    for (size_t k = 0; k != mp_def.nPatches(); ++k)
        deformation.patch(k).coefs() -= mp_ori.patch(k).coefs();

    gsInfo <<"Maximum deformation coef: "
           << deformation.patch(0).coefs().colwise().maxCoeff() <<".\n";
    gsInfo <<"Minimum deformation coef: "
           << deformation.patch(0).coefs().colwise().minCoeff() <<".\n";
    //! [Construct solution]


    gsWriteParaview(mp_ori,"mp_ori");
    gsWriteParaview(mp_def,"mp_def");



    return EXIT_SUCCESS;
}// end main
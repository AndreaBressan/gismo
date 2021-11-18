/** @file gsC1Argyris.h

    @brief Creates the C1 Argyris space.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): P. Weinmueller
*/

#pragma once

#include <gsC1Basis/gsC1ArgyrisEdge.h>
#include <gsC1Basis/gsC1ArgyrisVertex.h>

namespace gismo
{
template<short_t d,class T>
class gsC1Argyris //: public gsGeoTraits<d,T>::GeometryBase
{

private:
    typedef gsC1ArgyrisBasis<d,T> Basis;
    typedef typename std::vector<Basis> ArgyrisBasisContainer;

    /// Shared pointer for gsC1Argyris
    typedef memory::shared_ptr< gsC1Argyris > Ptr;

    /// Unique pointer for gsC1Argyris
    typedef memory::unique_ptr< gsC1Argyris > uPtr;


public:

    /// Empty constructor
    gsC1Argyris() { }

    gsC1Argyris(gsMultiPatch<T> const & mp,
                const gsOptionList & optionList)
                : m_mp(mp), m_optionList(optionList)
    {
        multiBasis = gsMultiBasis<>(m_mp);

        // p-refine
        for (size_t np = 0; np < m_mp.nPatches(); ++np)
            multiBasis.basis(np).setDegree(m_optionList.getInt("discreteDegree"));

        p_tilde = m_optionList.getInt("gluingDataDegree");//math::max(m_optionList.getInt("discreteDegree") - 1, 2);
        r_tilde = m_optionList.getInt("gluingDataRegularity");//p_tilde - 1;
        if (p_tilde == -1 || r_tilde == -1)
        {
            p_tilde = math::max(m_optionList.getInt("discreteDegree") - 1, 2);
            r_tilde = p_tilde - 1;
        }
    }

    gsC1Argyris(gsMultiPatch<T> const & mp,
                gsMultiBasis<T> const & mb,
                const gsOptionList & optionList)
            : m_mp(mp), multiBasis(mb), m_optionList(optionList)
    {
        p_tilde = m_optionList.getInt("gluingDataDegree");//math::max(m_optionList.getInt("discreteDegree") - 1, 2);
        r_tilde = m_optionList.getInt("gluingDataRegularity");//p_tilde - 1;

        if (p_tilde == -1 || r_tilde == -1)
        {
            p_tilde = math::max(m_optionList.getInt("discreteDegree") - 1, 2);
            r_tilde = p_tilde - 1;
        }
    }

    void createPlusMinusSpace(gsKnotVector<T> & kv1, gsKnotVector<T> & kv2,
                           gsKnotVector<T> & kv1_patch, gsKnotVector<T> & kv2_patch,
                           gsKnotVector<T> & kv1_result, gsKnotVector<T> & kv2_result);

    void createPlusMinusSpace(gsKnotVector<T> & kv1,
                           gsKnotVector<T> & kv1_patch,
                           gsKnotVector<T> & kv1_result, gsKnotVector<T> & kv2_result);

    void createGluingDataSpace(gsKnotVector<T> & kv1, gsKnotVector<T> & kv2,
                           gsKnotVector<T> & kv1_patch, gsKnotVector<T> & kv2_patch,
                           gsKnotVector<T> & kv_result);

    void createLokalEdgeSpace(gsKnotVector<T> & kv_plus, gsKnotVector<T> & kv_minus,
                           gsKnotVector<T> & kv_gD_1, gsKnotVector<T> & kv_gD_2,
                           gsKnotVector<T> & kv_1, gsKnotVector<T> & kv_2,
                           gsKnotVector<T> & kv1_result, gsKnotVector<T> & kv2_result);

    void createLokalEdgeSpace(gsKnotVector<T> & kv_plus, gsKnotVector<T> & kv_minus,
                              gsKnotVector<T> & kv_1,
                              gsKnotVector<T> & kv1_result);

    void createLokalVertexSpace(gsTensorBSplineBasis<d,T> & basis_vertex_1, gsTensorBSplineBasis<d,T> & result_1);

    void init()
    {
        m_bases.reserve(m_mp.nPatches()); // For each Patch
        for (size_t np = 0; np < m_mp.nPatches(); np++)
        {
            gsC1ArgyrisBasis<d,T> c1ArgyrisBasis(m_mp, np, m_optionList);
            m_bases.push_back(c1ArgyrisBasis);
        }

        // Create interior spline space
        for (size_t np = 0; np < m_mp.nPatches(); np++)
        {
            gsTensorBSplineBasis<d, T> basis_inner = dynamic_cast<gsTensorBSplineBasis<d, T> &>(multiBasis.basis(np));

            // Construct special space for r = p - 1:
            // The first and the last knot (not 0,1) are repeated +1, e.g.
            // deg 3, r = 2: |||| || | [...] | || ||||
            index_t r = m_optionList.getInt("discreteRegularity"); // Assume same reg for each direction
            for (index_t uv = 0; uv < 2; uv++)
            {
                if (basis_inner.degree(uv) - r == 1 && !m_optionList.getSwitch("twoPatch"))
                {
                    T knot_u = basis_inner.knot(uv,basis_inner.degree(uv)+1);
                    if (knot_u != 1)
                        basis_inner.insertKnot(knot_u,uv,1);

                    if (knot_u != 0.5 && knot_u != 1)
                        basis_inner.insertKnot(1-knot_u,uv,1);
                }
            }

            //gsInfo << "basis u " << basis_inner.knots(0).asMatrix() << "\n";
            //gsInfo << "basis v " << basis_inner.knots(1).asMatrix() << "\n";

            m_bases[np].setInnerBasis(basis_inner);
        }

        // For loop over Interface to construct the spaces
        for (size_t numInt = 0; numInt < m_mp.interfaces().size(); numInt++)
        {
            const boundaryInterface & item = m_mp.interfaces()[numInt];

            const index_t side_1 = item.first().side().index();
            const index_t side_2 = item.second().side().index();
            const index_t patch_1 = item.first().patch;
            const index_t patch_2 = item.second().patch;

            const index_t dir_1 = side_1 > 2 ? 0 : 1;
            const index_t dir_2 = side_2 > 2 ? 0 : 1;

            gsBSplineBasis<> basis_1 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_1).component(dir_1));
            gsBSplineBasis<> basis_2 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_2).component(dir_2));

            gsBSplineBasis<> basis_geo_1 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_1).component(1-dir_1));
            gsBSplineBasis<> basis_geo_2 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_2).component(1-dir_2));

            gsKnotVector<T> kv_1 = basis_1.knots();
            gsKnotVector<T> kv_2 = basis_2.knots();

            gsBSplineBasis<> patch_basis_1 = dynamic_cast<gsBSplineBasis<> &>(m_mp.patch(patch_1).basis().component(dir_1));
            gsKnotVector<T> kv_patch_1 = patch_basis_1.knots();

            gsBSplineBasis<> patch_basis_2 = dynamic_cast<gsBSplineBasis<> &>(m_mp.patch(patch_2).basis().component(dir_2));
            gsKnotVector<T> kv_patch_2 = patch_basis_1.knots();

            gsKnotVector<T> kv_plus, kv_minus, kv_gluingData;
            createPlusMinusSpace(kv_1, kv_2, kv_patch_1, kv_patch_2, kv_plus, kv_minus);

            gsBSplineBasis<> basis_plus(kv_plus); // S(p,r+1,h)
            gsBSplineBasis<> basis_minus(kv_minus); // S(p-1,r,h)

            //\tilde{p} = max(p-1,2)
            //\tilde{r} = \tilde{p}-1
            createGluingDataSpace(kv_1, kv_2, kv_patch_1, kv_patch_2, kv_gluingData);

            gsBSplineBasis<> basis_gluingData(kv_gluingData); // S(\tilde{p},\tilde{r},h)

            m_bases[patch_1].setBasisPlus(basis_plus, side_1);
            m_bases[patch_2].setBasisPlus(basis_plus, side_2);

            m_bases[patch_1].setBasisMinus(basis_minus, side_1);
            m_bases[patch_2].setBasisMinus(basis_minus, side_2);

            m_bases[patch_1].setBasisGeo(basis_geo_1, side_1);
            m_bases[patch_2].setBasisGeo(basis_geo_2, side_2);

            m_bases[patch_1].setBasisGluingData(basis_gluingData, side_1);
            m_bases[patch_2].setBasisGluingData(basis_gluingData, side_2);
/*
            if (m_optionList.getSwitch("info"))
            {
                gsInfo << "Basis geo 1 : " << basis_geo_1.knots().asMatrix() << "\n";
                gsInfo << "Basis geo 2 : " << basis_geo_2.knots().asMatrix() << "\n";
                gsInfo << "Basis plus : " << basis_plus.knots().asMatrix() << "\n";
                gsInfo << "Basis minus : " << basis_minus.knots().asMatrix() << "\n";

                gsInfo << "Basis gluingData : " << basis_gluingData.knots().asMatrix() << "\n";
            }
*/

            if (m_optionList.getSwitch("isogeometric"))
            {
                gsTensorBSplineBasis<d, T> basis_edge_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(patch_1));
                m_bases[patch_1].setEdgeBasis(basis_edge_1, side_1);

                gsTensorBSplineBasis<d, T> basis_edge_2 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(patch_2));
                m_bases[patch_2].setEdgeBasis(basis_edge_2, side_2);
            }
            else
            {
                gsKnotVector<T> kv_geo_1 = basis_geo_1.knots();
                gsKnotVector<T> kv_geo_2 = basis_geo_2.knots();

                gsKnotVector<T> kv_edge_1, kv_edge_2;

                createLokalEdgeSpace(kv_plus, kv_minus, kv_gluingData, kv_gluingData, kv_patch_1, kv_patch_2, kv_edge_1, kv_edge_2);
                gsBSplineBasis<> basis_edge(kv_edge_1);
/*
                if (m_optionList.getSwitch("info"))
                    gsInfo << "Basis edge : " << basis_edge.knots().asMatrix() << "\n";
*/
                gsTensorBSplineBasis<d, T> basis_edge_1(dir_1 == 0 ? kv_edge_1 : kv_geo_1, dir_1 == 0 ? kv_geo_1 : kv_edge_1);
                gsTensorBSplineBasis<d, T> basis_edge_2(dir_2 == 0 ? kv_edge_2 : kv_geo_2, dir_2 == 0 ? kv_geo_2 : kv_edge_2);

                m_bases[patch_1].setEdgeBasis(basis_edge_1, side_1);
                m_bases[patch_2].setEdgeBasis(basis_edge_2, side_2);
            }

        }
        // For loop over the Edge to construct the spaces
        for (size_t numBdy = 0; numBdy < m_mp.boundaries().size(); numBdy++)
        {
            const patchSide & bit = m_mp.boundaries()[numBdy];

            index_t patch_1 = bit.patch;
            index_t side_1 = bit.side().index();

            index_t dir_1 = m_mp.boundaries()[numBdy].m_index < 3 ? 1 : 0;

            // Using Standard Basis for boundary edges
            gsTensorBSplineBasis<d, T> basis_edge_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(patch_1));

            gsBSplineBasis<> basis_1 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_1).component(dir_1));
            gsBSplineBasis<> basis_geo_1 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_1).component(1-dir_1));

            // Assume that plus/minus space is the same as the inner space
            gsBSplineBasis<> basis_plus, basis_minus;

            if (m_optionList.getSwitch("twoPatch"))
            {
                basis_plus = basis_1;
                basis_minus = basis_1;
            }
            else
            {
                gsKnotVector<T> kv_1 = basis_1.knots();

                gsBSplineBasis<> patch_basis_1 = dynamic_cast<gsBSplineBasis<> &>(m_mp.patch(patch_1).basis().component(dir_1));
                gsKnotVector<T> kv_patch_1 = patch_basis_1.knots();

                gsKnotVector<T> kv_plus, kv_minus;
                createPlusMinusSpace(kv_1,  kv_patch_1,  kv_plus, kv_minus);

                basis_plus = gsBSplineBasis<>(kv_plus);
                basis_minus = gsBSplineBasis<>(kv_minus);

                if (basis_geo_1.degree(0) - m_optionList.getInt("discreteRegularity") == 1 && !m_optionList.getSwitch("twoPatch"))
                {
/*
                    T knot_u = basis_edge_1.knot(0,basis_edge_1.degree(0)+1);
                    if (knot_u != 1)
                    {
                        basis_geo_1.insertKnot(knot_u,1); // the first
                        //basis_geo_1.insertKnot(knot_u+knot_u,1);
                    }
                    if (knot_u != 0.5)
                    {
                        basis_geo_1.insertKnot(1-knot_u,1); // the last
                        //basis_geo_1.insertKnot(1-knot_u-knot_u,1);
                    }
*/
                    basis_geo_1.reduceContinuity(1);
                }
                m_bases[patch_1].setBasisGeo(basis_geo_1, side_1);

                if (m_optionList.getSwitch("simplified"))
                {
                    m_bases[patch_1].setEdgeBasis(basis_edge_1, side_1);

                    m_bases[patch_1].setBasisPlus(basis_plus, side_1);
                    m_bases[patch_1].setBasisMinus(basis_minus, side_1);
/*
                    if (m_optionList.getSwitch("info"))
                        gsInfo << "Basis boundary: " << basis_edge_1 << "\n";
*/
                }
                else
                {
                    gsKnotVector<T> kv_edge_1;
                    createLokalEdgeSpace(kv_plus, kv_minus, kv_patch_1, kv_edge_1);

                    gsKnotVector<T> kv_geo_1 = basis_geo_1.knots();
                    gsTensorBSplineBasis<d, T> basis_edge_1_temp(dir_1 == 0 ? kv_edge_1 : kv_geo_1, dir_1 == 0 ? kv_geo_1 : kv_edge_1);

                    m_bases[patch_1].setEdgeBasis(basis_edge_1_temp, side_1);

                    m_bases[patch_1].setBasisPlus(basis_plus, side_1);
                    m_bases[patch_1].setBasisMinus(basis_minus, side_1);
/*
                    if (m_optionList.getSwitch("info"))
                        gsInfo << "Basis boundary: " << basis_edge_1_temp << "\n";
*/
                }
            }
        }

        // For loop over the Vertex to construct the spaces
        for (size_t numVer = 0; numVer < m_mp.vertices().size(); numVer++)
        {
            std::vector<patchCorner> allcornerLists = m_mp.vertices()[numVer];
            std::vector<size_t> patchIndex;
            std::vector<size_t> vertIndex;
            for (size_t j = 0; j < allcornerLists.size(); j++)
            {
                patchIndex.push_back(allcornerLists[j].patch);
                vertIndex.push_back(allcornerLists[j].m_index);
            }

            if (patchIndex.size() == 1) // Boundary vertex
            {
                index_t patch_1 = patchIndex[0];
                index_t vertex_1 = vertIndex[0];

                gsTensorBSplineBasis<d, T> basis_vertex_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(patch_1));

                if (m_optionList.getSwitch("isogeometric")) {

                    m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                    m_bases[patch_1].setKindOfVertex(-1, vertex_1);
                }
                else
                {
                    if (m_optionList.getSwitch("simplified"))
                    {
                        m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                        m_bases[patch_1].setKindOfVertex(-1, vertex_1);
                    }
                    else
                    {
                        index_t r = m_optionList.getInt("discreteRegularity");
                        if (basis_vertex_1.degree(0) - r == 1 && !m_optionList.getSwitch("twoPatch")) // == basis_vertex_1.degree(1)
                            basis_vertex_1.reduceContinuity(1); // In the case for the max. smoothness

                        m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                        m_bases[patch_1].setKindOfVertex(-1, vertex_1);
                    }
                }
            }
            else if (patchIndex.size() > 1)
            {
                gsMultiPatch<> temp_mp;
                for (size_t j = 0; j < patchIndex.size(); j++)
                    temp_mp.addPatch(m_mp.patch(patchIndex[j]));

                temp_mp.computeTopology();
                if (patchIndex.size() == temp_mp.interfaces().size()) // Internal vertex
                {
                    for (size_t j = 0; j < patchIndex.size(); j++)
                    {
                        index_t patch_1 = patchIndex[j];
                        index_t vertex_1 = vertIndex[j];
                        if (m_optionList.getSwitch("isogeometric"))
                        {
                            gsTensorBSplineBasis<d, T> basis_vertex_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(
                                    patch_1));
                            m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                            m_bases[patch_1].setKindOfVertex(0, vertex_1);
                        }
                        else
                        {
                            gsTensorBSplineBasis<d, T> basis_vertex_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(
                                    patch_1));

                            //createLokalVertexSpace(basis_vertex_1, basis_vertex_result);

                            //index_t p_tilde_1 = math::max(multiBasis.basis(patch_1).degree(0)-1,3);
                            //index_t p_tilde_2 = math::max(multiBasis.basis(patch_1).degree(1)-1,3);

                            basis_vertex_1.degreeElevate(p_tilde-1,0); // Keep smoothness
                            basis_vertex_1.degreeElevate(p_tilde-1,1);

                            index_t r = m_optionList.getInt("discreteRegularity");
                            //if (multiBasis.basis(patch_1).degree(0) - r == 1)
                            //    basis_vertex_1.reduceContinuity(r-1); // In the case for the max. smoothness
                            //else if (r > 2)
                             // 1600
                            {
                                if (r != 1)
                                    basis_vertex_1.reduceContinuity(1); // bcs of minus space
                                else if (r_tilde < r-1)
                                    basis_vertex_1.reduceContinuity(r-r_tilde-1);
                            }

                            m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                            m_bases[patch_1].setKindOfVertex(0, vertex_1);
                        }

                        if (m_optionList.getSwitch("C1Vertex"))
                            m_bases[patch_1].setValenceOfVertex(temp_mp.nInterfaces(), vertex_1);
                    }
                }
                else if (patchIndex.size() > temp_mp.interfaces().size() && !m_optionList.getSwitch("twoPatch"))// Interface-Boundary vertex
                {
                    for (size_t j = 0; j < patchIndex.size(); j++)
                    {
                        index_t patch_1 = patchIndex[j];
                        index_t vertex_1 = vertIndex[j];

                        if (m_optionList.getSwitch("isogeometric"))
                        {
                            gsTensorBSplineBasis<d, T> basis_vertex_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(patch_1));
                            m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                            m_bases[patch_1].setKindOfVertex(1, vertex_1);
                        }
                        else
                        {
                            gsTensorBSplineBasis<d, T> basis_vertex_1 = dynamic_cast<gsTensorBSplineBasis<d, real_t> &>(multiBasis.basis(patch_1));

                            //index_t p_tilde_1 = math::max(multiBasis.basis(patch_1).degree(0)-1,3);
                            //index_t p_tilde_2 = math::max(multiBasis.basis(patch_1).degree(1)-1,3);
/*
                            gsBSplineBasis<> basis_1 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(patch_1).component(0));
                            gsBSplineBasis<> basis_geo_1 = dynamic_cast<gsBSplineBasis<> &>(multiBasis.basis(0).component(1));

                            gsKnotVector<T> kv_1 = basis_1.knots();
                            gsKnotVector<T> kv_2 = basis_geo_1.knots();
                            gsTensorBSplineBasis<d, T> basis_vertex_1(kv_1,kv_2);
*/
                            basis_vertex_1.degreeElevate(p_tilde-1,0); // Keep smoothness
                            basis_vertex_1.degreeElevate(p_tilde-1,1);

                            index_t r = m_optionList.getInt("discreteRegularity");
                            //if (multiBasis.basis(patch_1).degree(0) - r == 1)
                            //    basis_vertex_1.reduceContinuity(r-1); // In the case for the max. smoothness
                            //else if (r > 2)
                            // 1600
                            {
                                if (r != 1)
                                    basis_vertex_1.reduceContinuity(1); // bcs of minus space
                                else if (r_tilde < r-1)
                                    basis_vertex_1.reduceContinuity(r-r_tilde-1);
                            }

/*
                            if (m_optionList.getSwitch("info"))
                            {
                                gsBSplineBasis<> basis_geo_1 = dynamic_cast<gsBSplineBasis<> &>(basis_vertex_1.component(0));
                                gsInfo << "Basis vertex 1: " << basis_geo_1.knots().asMatrix() << "\n";
                                gsBSplineBasis<> basis_geo_2 = dynamic_cast<gsBSplineBasis<> &>(basis_vertex_1.component(1));
                                gsInfo << "Basis vertex 2: " << basis_geo_2.knots().asMatrix() << "\n";
                            }
*/
                            m_bases[patch_1].setVertexBasis(basis_vertex_1, vertex_1);
                            m_bases[patch_1].setKindOfVertex(1, vertex_1);
                        }
                    }
                }
            }
        }

        // Init local Basis
        for (size_t np = 0; np < m_mp.nPatches(); np++)
            m_bases[np].init();


        m_system.clear();
        index_t dim_col = 0, dim_row = 0;
        for (size_t i = 0; i < m_bases.size(); i++)
        {
            dim_col += m_bases[i].size_cols();
            dim_row += m_bases[i].size_rows();
        }

        m_system.resize(dim_row, dim_col);
        const index_t nz = 7*dim_row; // TODO
        m_system.reserve(nz);

        if (m_optionList.getSwitch("info"))
            for (size_t np = 0; np < m_mp.nPatches(); np++)
               m_bases[np].print_spaces();

    }

    void createArgyrisSpace()
    {
        // Compute Inner Basis functions
        index_t shift_row = 0, shift_col = 0;
        for(size_t np = 0; np < m_mp.nPatches(); ++np)
        {
            index_t dim_u = m_bases[np].getInnerBasis().component(0).size();
            index_t dim_v = m_bases[np].getInnerBasis().component(1).size();

            index_t row_i = 0;
            for (index_t j = 2; j < dim_v-2; ++j)
                for (index_t i = 2; i < dim_u-2; ++i)
                {
                    m_system.insert(shift_row + row_i,shift_col + j*dim_u+i) = 1.0;
                    ++row_i;
                }

            shift_row += m_bases[np].size_rows();
            shift_col += m_bases[np].size_cols();
        }

        // Compute Interface Basis functions
        /*
         *  (Side-1) * 2 + 0/1 = Index
         *  0 == for lower vertex index, 1 == higher vertex index
         *
         *  Side 1, Vertex 1 == 0
         *  Side 1, Vertex 3 == 1
         *  Side 2, Vertex 2 == 2
         *  Side 2, Vertex 4 == 3
         *  Side 3, Vertex 1 == 4
         *  ...
         */
        std::vector<std::vector<gsMultiPatch<T>>> vertex_bf(m_mp.nPatches(), std::vector<gsMultiPatch<T>>(8));
        for (size_t numInt = 0; numInt < m_mp.interfaces().size(); numInt++)
        {
            const boundaryInterface & item = m_mp.interfaces()[numInt];

            gsC1ArgyrisEdge<d, T> c1ArgyrisEdge(m_mp, m_bases, item, numInt, m_optionList);
            c1ArgyrisEdge.saveBasisInterface(m_system);
            if (m_optionList.getSwitch("C1Vertex"))
                c1ArgyrisEdge.saveBasisVertex(vertex_bf);
        }
        // Compute Edge Basis functions
        for (size_t numBdy = 0; numBdy < m_mp.boundaries().size(); numBdy++)
        {
            const patchSide & bit = m_mp.boundaries()[numBdy];

            if (m_optionList.getSwitch("simplified"))
            {
                gsC1ArgyrisEdge<d, T> c1ArgyrisEdge(m_mp, m_bases, bit, numBdy, m_optionList);
                c1ArgyrisEdge.saveSimplifiedBasisBoundary(bit, m_system);
            }
            else
            {
                gsC1ArgyrisEdge<d, T> c1ArgyrisEdge(m_mp, m_bases, bit, numBdy, m_optionList);
                c1ArgyrisEdge.saveBasisBoundary(m_system);
            }
        }
        // Compute Vertex Basis functions
        for (size_t numVer = 0; numVer < m_mp.vertices().size(); numVer++)
        {
            std::vector<patchCorner> allcornerLists = m_mp.vertices()[numVer];
            std::vector<size_t> patchIndex;
            std::vector<size_t> vertIndex;
            for (size_t j = 0; j < allcornerLists.size(); j++)
            {
                patchIndex.push_back(allcornerLists[j].patch);
                vertIndex.push_back(allcornerLists[j].m_index);
            }

            if (patchIndex.size() > 2 && m_optionList.getSwitch("C1Vertex"))
            {
                gsC1ArgyrisVertex<d, T> c1ArgyrisVertex(m_mp, m_bases, patchIndex, vertIndex, numVer, vertex_bf, m_optionList);
                c1ArgyrisVertex.saveBasisVertex(m_system);
            }
            else if (patchIndex.size() == 1 && m_optionList.getSwitch("simplified"))
            {
                index_t np = patchIndex[0];
                index_t dim_u = m_bases[np].getVertexBasis(vertIndex[0]).component(0).size();
                index_t dim_v = m_bases[np].getVertexBasis(vertIndex[0]).component(1).size();

                shift_row = 0, shift_col = 0;
                for(index_t np_temp = 0; np_temp < np; ++np_temp) {
                    shift_row += m_bases[np_temp].size_rows();
                    shift_col += m_bases[np_temp].size_cols();
                }
                shift_row += m_bases[np].rowBegin(vertIndex[0]+4);
                shift_col += m_bases[np].colBegin(vertIndex[0]+4);

                switch (vertIndex[0]) {
                    case 1:
                        m_system.insert(shift_row + 0,shift_col + 1*dim_u+1) = 1.0; // interior
                        m_system.insert(shift_row + 1,shift_col + 0*dim_u+0) = 1.0; // bdy
                        m_system.insert(shift_row + 2,shift_col + 0*dim_u+1) = 1.0;
                        m_system.insert(shift_row + 3,shift_col + 0*dim_u+2) = 1.0;
                        m_system.insert(shift_row + 4,shift_col + 1*dim_u+0) = 1.0;
                        m_system.insert(shift_row + 5,shift_col + 2*dim_u+0) = 1.0;
                        break;
                    case 2:
                        m_system.insert(shift_row + 0,shift_col + 1*dim_u+dim_u-2) = 1.0; // interior
                        m_system.insert(shift_row + 1,shift_col + 0*dim_u+dim_u-1) = 1.0; // bdy
                        m_system.insert(shift_row + 2,shift_col + 0*dim_u+dim_u-2) = 1.0;
                        m_system.insert(shift_row + 3,shift_col + 0*dim_u+dim_u-3) = 1.0;
                        m_system.insert(shift_row + 4,shift_col + 1*dim_u+dim_u-1) = 1.0;
                        m_system.insert(shift_row + 5,shift_col + 2*dim_u+dim_u-1) = 1.0;
                        break;
                    case 3:
                        m_system.insert(shift_row + 0,shift_col + (dim_v-2)*dim_u+1) = 1.0; // interior
                        m_system.insert(shift_row + 1,shift_col + (dim_v-1)*dim_u+0) = 1.0; // bdy
                        m_system.insert(shift_row + 2,shift_col + (dim_v-1)*dim_u+1) = 1.0;
                        m_system.insert(shift_row + 3,shift_col + (dim_v-1)*dim_u+2) = 1.0;
                        m_system.insert(shift_row + 4,shift_col + (dim_v-2)*dim_u+0) = 1.0;
                        m_system.insert(shift_row + 5,shift_col + (dim_v-3)*dim_u+0) = 1.0;
                        break;
                    case 4:
                        m_system.insert(shift_row + 0,shift_col + (dim_v-2)*dim_u+dim_u-2) = 1.0; // interior
                        m_system.insert(shift_row + 1,shift_col + (dim_v-1)*dim_u+dim_u-1) = 1.0; // bdy
                        m_system.insert(shift_row + 2,shift_col + (dim_v-1)*dim_u+dim_u-2) = 1.0;
                        m_system.insert(shift_row + 3,shift_col + (dim_v-1)*dim_u+dim_u-3) = 1.0;
                        m_system.insert(shift_row + 4,shift_col + (dim_v-2)*dim_u+dim_u-1) = 1.0;
                        m_system.insert(shift_row + 5,shift_col + (dim_v-3)*dim_u+dim_u-1) = 1.0;
                        break;
                    default:
                        gsInfo << "Wrong side index!\n";
                }
            }
            else
            {
                gsC1ArgyrisVertex<d, T> c1ArgyrisVertex(m_mp, m_bases, patchIndex, vertIndex, numVer, m_optionList);
                c1ArgyrisVertex.saveBasisVertex(m_system);
            }
        }

        m_system.makeCompressed();

        if (m_optionList.getSwitch("info"))
        {
            gsInfo << "Dim for Patches: \n";
            for(size_t np = 0; np < m_mp.nPatches(); ++np)
            {
                gsInfo << "(" << m_bases[np].size_rows() << "," << m_bases[np].size_cols() << "), ";
            }
            gsInfo << "\n";
        }


    }

    void uniformRefine()
    {
        index_t p = multiBasis.minCwiseDegree();
        index_t r = m_optionList.getInt("discreteRegularity");

        multiBasis.uniformRefine(1,p-r);
    }

    void writeParaviewSinglePatch( index_t patchID, std::string type )
    {
        std::string fileName;
        std::string basename = "BasisFunctions_" + type + "_" + util::to_string(patchID);
        gsParaviewCollection collection(basename);

        index_t shift_row = 0, shift_col = 0;
        for (index_t np = 0; np < patchID; ++np)
        {
            shift_row += m_bases[np].size_rows();
            shift_col += m_bases[np].size_cols();
        }

        if (type == "inner")
        {
            index_t ii = 0;
            for (index_t i = m_bases[patchID].rowBegin(0);
                     i < m_bases[patchID].rowEnd(0); i++, ii++) // Single basis function
            {
                index_t start_j = m_bases[patchID].colBegin(0);
                index_t end_j = m_bases[patchID].colEnd(0);

                gsMatrix<> coefs = m_system.block(shift_row + i, shift_col + start_j, 1, end_j - start_j);

                gsGeometry<>::uPtr geo_temp;
                geo_temp = m_bases[patchID].getInnerBasis().makeGeometry(coefs.transpose());

                gsTensorBSpline<d, T> patch_single = dynamic_cast<gsTensorBSpline<d, T> &> (*geo_temp);

                fileName = basename + "_0_" + util::to_string(ii);
                gsField<> temp_field(m_mp.patch(patchID), patch_single);
                gsWriteParaview(temp_field, fileName, 5000);
                collection.addTimestep(fileName, ii, "0.vts");
            }
        }
        else if (type == "edge" || type == "vertex")
        {
            index_t ii = 0;
            for (index_t side = 1; side < 5; ++side) {
                index_t side_shift = type == "edge" ? 0 : 4;
                for (index_t i = m_bases[patchID].rowBegin(side + side_shift);
                     i < m_bases[patchID].rowEnd(side + side_shift); i++, ii++) // Single basis function
                {
                    index_t start_j = m_bases[patchID].colBegin(side + side_shift);
                    index_t end_j = m_bases[patchID].colEnd(side + side_shift);

                    gsMatrix<> coefs = m_system.block(shift_row + i, shift_col + start_j, 1, end_j - start_j);

                    gsGeometry<>::uPtr geo_temp;
                    if (type == "edge")
                        geo_temp = m_bases[patchID].getEdgeBasis(side).makeGeometry(coefs.transpose());
                    else if (type == "vertex")
                        geo_temp = m_bases[patchID].getVertexBasis(side).makeGeometry(coefs.transpose());

                    gsTensorBSpline<d, T> patch_single = dynamic_cast<gsTensorBSpline<d, T> &> (*geo_temp);

                    fileName = basename + "_0_" + util::to_string(ii);
                    gsField<> temp_field(m_mp.patch(patchID), patch_single);
                    gsWriteParaview(temp_field, fileName, 5000);
                    collection.addTimestep(fileName, ii, "0.vts");
                }
            }
        }
        collection.save();
    }

    void plotParaview( std::string fn, index_t npts = 1000 )
    {
        gsParaviewCollection collection2(fn);
        std::string fileName2;

        for ( size_t pp = 0; pp < m_mp.nPatches(); ++pp ) // Patches
        {
            index_t shift_row = 0, shift_col = 0;
            for (size_t np = 0; np < pp; ++np)
            {
                shift_row += m_bases[np].size_rows();
                shift_col += m_bases[np].size_cols();
            }

            fileName2 = fn + util::to_string(pp);

            const gsFunction<T> & geometry = m_mp.patch(pp);

            const int n = geometry.targetDim();

            gsMatrix<T> ab = geometry.support();
            gsVector<T> a = ab.col(0);
            gsVector<T> b = ab.col(1);

            gsVector<unsigned> np = uniformSampleCount(a, b, npts);
            gsMatrix<T> pts = gsPointGrid(a, b, np);

            gsMatrix<T> eval_geo = geometry.eval(pts);//pts
            gsMatrix<T> eval_field;

            // Here add g1 basis
            eval_field.setZero(1, pts.cols());

            index_t ii = 0;
            for (index_t i = m_bases[pp].rowBegin(0);
                     i < m_bases[pp].rowEnd(0); i++, ii++) // Single basis function
            {
                index_t start_j = m_bases[pp].colBegin(0);
                index_t end_j = m_bases[pp].colEnd(0);

                gsMatrix<> coefs = m_system.block(shift_row + i, shift_col + start_j, 1, end_j - start_j);

                gsGeometry<>::uPtr geo_temp;
                geo_temp = m_bases[pp].getInnerBasis().makeGeometry(coefs.transpose());

                gsTensorBSpline<d, T> patch_single = dynamic_cast<gsTensorBSpline<d, T> &> (*geo_temp);
                gsField<> temp_field(m_mp.patch(pp), patch_single);
                eval_field += temp_field.value(pts);
            }
            std::string type = "edge";
            ii = 0;
            for (index_t side = 1; side < 5; ++side) {
                for (index_t i = m_bases[pp].rowBegin(side);
                     i < m_bases[pp].rowEnd(side); i++, ii++) // Single basis function
                {
                    index_t start_j = m_bases[pp].colBegin(side);
                    index_t end_j = m_bases[pp].colEnd(side);

                    gsMatrix<> coefs = m_system.block(shift_row + i, shift_col + start_j, 1, end_j - start_j);

                    gsGeometry<>::uPtr geo_temp;
                    if (type == "edge")
                        geo_temp = m_bases[pp].getEdgeBasis(side).makeGeometry(coefs.transpose());
                    else if (type == "vertex")
                        geo_temp = m_bases[pp].getVertexBasis(side).makeGeometry(coefs.transpose());

                    gsTensorBSpline<d, T> patch_single = dynamic_cast<gsTensorBSpline<d, T> &> (*geo_temp);
                    gsField<> temp_field(m_mp.patch(pp), patch_single);
                    eval_field += temp_field.value(pts);
                }
            }

            type = "vertex";
            ii = 0;
            for (index_t side = 1; side < 5; ++side) {
                for (index_t i = m_bases[pp].rowBegin(side+4);
                     i < m_bases[pp].rowEnd(side+4); i++, ii++) // Single basis function
                {
                    index_t start_j = m_bases[pp].colBegin(side+4);
                    index_t end_j = m_bases[pp].colEnd(side+4);

                    gsMatrix<> coefs = m_system.block(shift_row + i, shift_col + start_j, 1, end_j - start_j);

                    gsGeometry<>::uPtr geo_temp;
                    if (type == "edge")
                        geo_temp = m_bases[pp].getEdgeBasis(side).makeGeometry(coefs.transpose());
                    else if (type == "vertex")
                        geo_temp = m_bases[pp].getVertexBasis(side).makeGeometry(coefs.transpose());

                    gsTensorBSpline<d, T> patch_single = dynamic_cast<gsTensorBSpline<d, T> &> (*geo_temp);
                    gsField<> temp_field(m_mp.patch(pp), patch_single);
                    eval_field += temp_field.value(pts);
                }
            }

            //gsFunctionExpr<> solVal("x",2);
            //gsFunctionExpr<> solVal("-1*cos(pi*x)*sin(pi*y)",2);
            gsFunctionExpr<> solVal("(cos(4*pi*x) - 1) * (cos(4*pi*y) - 1)",2);
            //eval_field -= solVal.eval(eval_geo);

            /*
            for (size_t numSpaces = 0; numSpaces < m_bases[pp].getBasisG1Container().size(); ++numSpaces)
            {
                gsTensorBSplineBasis<d, T> basis = m_bases[pp].getBasisG1Container()[numSpaces];
                for (index_t i = 0; i < m_bases[pp].getRowContainer()[numSpaces]; ++i)
                {
                    gsMatrix<> coefs = m_system.block(shift_row + i, shift_col, 1, basis.size());
                    gsMultiPatch<> geo;
                    geo.addPatch(basis.makeGeometry(coefs));
                    if (pp == 0 && i == 0 && numSpaces == 0)
                        gsWriteParaview(geo.patch(0),"test_geo",1000);
                    gsField<> temp_field(m_mp.patch(pp), geo.patch(0));
                    eval_field += temp_field.value(pts);
                }
                shift_row += m_bases[pp].getRowContainer()[numSpaces];
                shift_col += m_bases[pp].getColContainer()[numSpaces]; // == basis.size()
            }
            */
            if ( 3 - d > 0 )
            {
                np.conservativeResize(3);
                np.bottomRows(3-d).setOnes();
            }
            else if (d > 3)
            {
                gsWarn<< "Cannot plot 4D data.\n";
                return;
            }

            if ( 3 - n > 0 )
            {
                eval_geo.conservativeResize(3,eval_geo.cols() );
                eval_geo.bottomRows(3-n).setZero();
            }
            else if (n > 3)
            {
                gsWarn<< "Data is more than 3 dimensions.\n";
            }

            if ( eval_field.rows() == 2)
            {
                eval_field.conservativeResize(3,eval_geo.cols() );
                eval_field.bottomRows(1).setZero(); // 3-field.dim()
            }

            gsWriteParaviewTPgrid(eval_geo, eval_field, np.template cast<index_t>(), fileName2);


            collection2.addPart(fileName2, ".vts");
        }
        collection2.save();
    }

public:

    //GISMO_CLONE_FUNCTION(gsC1Argyris)

    /// getter for m_bases
    void getMultiBasis(gsMultiBasis<> & multiBasis_result)
    {
        multiBasis_result.clear();

        std::vector<gsBasis<T> *> basis_temp = std::vector<gsBasis<T> *>(m_mp.nPatches());
        for (size_t np = 0; np < m_mp.nPatches(); np++) {
            gsC1ArgyrisBasis<2, real_t>::uPtr mbasis = gsC1ArgyrisBasis<2, real_t>::make(m_bases[np]);
            basis_temp[np] = static_cast<gsBasis<> *>(mbasis.release());
        }

        multiBasis_result = gsMultiBasis<>(basis_temp, m_mp.topology());
    };

    gsSparseMatrix<T> & getSystem() { return m_system; };
    void setSystem(gsSparseMatrix<T> & system) { m_system = system; };

    T getMinMeshSize()
    {
        T meshSize = 1.0;
        for (size_t np = 0; np < m_mp.nPatches(); np++)
            if (multiBasis.basis(np).getMinCellLength() < meshSize)
                meshSize = multiBasis.basis(np).getMinCellLength();
        return meshSize;
    }

protected:
    /// Multipatch
    gsMultiPatch<T> m_mp;
    gsMultiBasis<> multiBasis;

    /// Optionlist
    gsOptionList m_optionList;

    ArgyrisBasisContainer m_bases;

    gsSparseMatrix<T> m_system;

    index_t p_tilde, r_tilde;

}; // Class gsC1Argyris


template<short_t d,class T>
void gsC1Argyris<d,T>::createPlusMinusSpace(gsKnotVector<T> & kv1, gsKnotVector<T> & kv2,
                           gsKnotVector<T> & kv1_patch, gsKnotVector<T> & kv2_patch,
                           gsKnotVector<T> & kv1_result, gsKnotVector<T> & kv2_result)
{
    std::vector<real_t> knots_unique_1 = kv1.unique();
    std::vector<real_t> knots_unique_2 = kv2.unique();

    std::vector<index_t> knots_mult_1 = kv1.multiplicities();
    std::vector<index_t> knots_mult_2 = kv2.multiplicities();

    std::vector<real_t> patch_kv_unique_1 = kv1_patch.unique();
    std::vector<index_t> patch_kv_mult_1 = kv1_patch.multiplicities();

    std::vector<real_t> patch_kv_unique_2 = kv2_patch.unique();
    std::vector<index_t> patch_kv_mult_2 = kv2_patch.multiplicities();

    index_t p = math::max(kv1.degree(), kv2.degree());

    std::vector<real_t> knot_vector_plus, knot_vector_minus;

    if (knots_unique_1 != knots_unique_2)
        gsInfo << "NOT IMPLEMENTED YET 1: Plus, Minus space \n";

    if (kv1.degree() != kv2.degree())
        gsInfo << "NOT IMPLEMENTED YET 2: Plus, Minus space \n";

    // patch_kv_unique_1 should be for now 0, 1
    if (patch_kv_unique_1[1] != 1)
        gsInfo << "NOT IMPLEMENTED YET 3: Plus, Minus space \n";

    if (knots_mult_1 != knots_mult_2)
        gsInfo << "NOT IMPLEMENTED YET 4: Plus, Minus space \n";

    kv1_result = kv2; // == kv2
    if (p - m_optionList.getInt("discreteRegularity") != 1)
        kv1_result.reduceMultiplicity(1);

    kv2_result = kv2; // == kv2
    kv2_result.degreeDecrease(1);
    if (p - m_optionList.getInt("discreteRegularity") != 1)
        kv2_result.reduceMultiplicity(1);

/*
 * TODO Add geometry inner knot regularity
 *
    index_t i_3 = 0, i_4 = 0;

    std::vector<real_t>::iterator it3 = patch_kv_unique_1.begin();
    std::vector<real_t>::iterator it4 = patch_kv_unique_2.begin();

    std::vector<real_t>::iterator it2 = knots_unique_2.begin();
    for(std::vector<real_t>::iterator it = knots_unique_1.begin(); it != knots_unique_1.end(); ++it)
    {
        if (*it == *it2)
        {
            knot_vector_plus.push_back(*it);
            knot_vector_minus.push_back(*it);
            ++it2;
        }
        else if (*it < *it2)
        {
            //knot_vector_plus.push_back(*it);
            //knot_vector_minus.push_back(*it);
        }
        else if (*it > *it2)
        {
            while (*it > *it2)
            {
                //knot_vector_plus.push_back(*it2);
                //knot_vector_minus.push_back(*it2);
                ++it2;
            }
            knot_vector_plus.push_back(*it2);
            knot_vector_minus.push_back(*it2);
            ++it2;
        }
    }

    // Repeat the first and the last vector p or p-1 times
    kv1_result = gsKnotVector<>(knot_vector_plus);
    kv1_result.degreeIncrease(p);
    if (kv1.multiplicities()[1] > p-2 && knots_unique_1[1] != 1) // TODO Check
        kv1_result.increaseMultiplicity(kv1.multiplicities()[1]-2);
    kv2_result = gsKnotVector<>(knot_vector_minus);
    kv2_result.degreeIncrease(p-1);
    if (kv2.multiplicities()[1] > p-2 && knots_unique_2[1] != 1) // TODO Check
        kv2_result.increaseMultiplicity(kv2.multiplicities()[1]-2);
*/
}


template<short_t d,class T>
void gsC1Argyris<d,T>::createPlusMinusSpace(gsKnotVector<T> & kv1,
                           gsKnotVector<T> & kv1_patch,
                           gsKnotVector<T> & kv1_result, gsKnotVector<T> & kv2_result)
{
    std::vector<real_t> knots_unique_1 = kv1.unique();

    std::vector<real_t> patch_kv_unique_1 = kv1_patch.unique();
    std::vector<index_t> patch_kv_mult_1 = kv1_patch.multiplicities();

    index_t p = math::max(kv1.degree(), 0);

    kv1_result = kv1;
    if (p - m_optionList.getInt("discreteRegularity") != 1)
        kv1_result.reduceMultiplicity(1);

    kv2_result = kv1;
    kv2_result.degreeDecrease(1);
    if (p - m_optionList.getInt("discreteRegularity") != 1)
        kv2_result.reduceMultiplicity(1);


    /*
    * TODO Add geometry inner knot regularity
    *
    index_t i_3 = 0, i_4 = 0;

    std::vector<real_t>::iterator it3 = patch_kv_unique_1.begin();
    std::vector<real_t>::iterator it4 = patch_kv_unique_2.begin();

    std::vector<real_t> knot_vector_plus, knot_vector_minus;

    for(std::vector<real_t>::iterator it = knots_unique_1.begin(); it != knots_unique_1.end(); ++it)
    {
        knot_vector_plus.push_back(*it);
        knot_vector_minus.push_back(*it);
    }

    // Repeat the first and the last vector p or p-1 times
    kv1_result = gsKnotVector<>(knot_vector_plus);
    kv1_result.degreeIncrease(p);
    kv2_result = gsKnotVector<>(knot_vector_minus);
    kv2_result.degreeIncrease(p-1);
     */
}

template<short_t d,class T>
void gsC1Argyris<d,T>::createGluingDataSpace(gsKnotVector<T> & kv1, gsKnotVector<T> & kv2,
                           gsKnotVector<T> & kv1_patch, gsKnotVector<T> & kv2_patch,
                           gsKnotVector<T> & kv_result)
{
    //index_t p_tilde = math::max(math::max(kv1.degree(), kv2.degree())-1,3); // max(p-1,3)
    //index_t r_tilde = math::max(p_tilde - 2, 1); // C^2 gluing data

    std::vector<real_t> knots_unique_1 = kv1.unique();
    std::vector<real_t> knots_unique_2 = kv2.unique();

    std::vector<real_t> knot_vector;

    /*
     * TODO Add geometry inner knot regularity
     */
    if (knots_unique_1 != knots_unique_2)
        gsInfo << "\n\n ERROR: Interfaces are not matching!!! \n\n";

    knot_vector = knots_unique_2; // = knots_unique_1
/*
    std::vector<real_t>::iterator it2 = knots_unique_2.begin();
    for(std::vector<real_t>::iterator it = knots_unique_1.begin(); it != knots_unique_1.end(); ++it)
    {
        if (*it == *it2)
        {
            knot_vector.push_back(*it);
            ++it2;
        }
        else if (*it < *it2)
        {
            //knot_vector.push_back(*it);
        }
        else if (*it > *it2)
        {
            while (*it > *it2)
            {
                //knot_vector.push_back(*it2);
                ++it2;
            }
            knot_vector.push_back(*it2);
            ++it2;
        }
    }
*/

    kv_result = gsKnotVector<>(knot_vector);
    kv_result.degreeIncrease(p_tilde);
    kv_result.increaseMultiplicity(p_tilde-r_tilde-1);
} // createGluingDataSpace


template<short_t d,class T>
void gsC1Argyris<d,T>::createLokalEdgeSpace(gsKnotVector<T> & kv_plus, gsKnotVector<T> & kv_minus,
                           gsKnotVector<T> & kv_gD_1, gsKnotVector<T> & kv_gD_2,
                           gsKnotVector<T> & kv_patch_1, gsKnotVector<T> & kv_patch_2,
                           gsKnotVector<T> & kv1_result, gsKnotVector<T> & kv2_result)
{
    index_t p_1 = math::max(kv_plus.degree()+kv_gD_1.degree()-1, kv_minus.degree()+kv_gD_1.degree() );
    //index_t p_2 = math::max(kv_plus.degree()+kv_gD_2.degree()-1, kv_minus.degree()+kv_gD_2.degree() ); == p_1

    std::vector<real_t> knots_unique_plus = kv_plus.unique(); // == kv_minus.unique()

    if (knots_unique_plus != kv_minus.unique())
        gsInfo << "ERROR LOKAL EDGE SPACE \n";

    kv1_result = gsKnotVector<>(knots_unique_plus);
    kv1_result.degreeIncrease(p_1);
    if (knots_unique_plus[1] != 1)
    {
        index_t r_plus = kv_plus.degree() - kv_plus.multiplicities()[1]; // The same for all
        index_t r_minus = kv_minus.degree() - kv_minus.multiplicities()[1]; // The same for all
        index_t r_tilde = kv_gD_1.degree() - kv_gD_1.multiplicities()[1]; // The same for all
        //gsInfo << "R_tilde " << r_tilde << "\n";
        //gsInfo << "r_plus " << r_plus << "\n";
        //gsInfo << "r_minus " << r_minus << "\n";

        index_t r = math::min(r_tilde, math::min(r_plus, r_minus));
        //gsInfo << "r " << r << "\n";

        kv1_result.increaseMultiplicity(p_1-r-1);
        //kv1_result.insert(T(0.5),2); // for g1600
    }
    // ==
    kv2_result = kv1_result;

/*
    index_t p_plus_diff = p_1 - kv_plus.degree();
    index_t p_gD_diff = p_1 - kv_gD_1.degree();
    index_t p_patch_diff = p_1 - kv_1.degree();

    std::vector<real_t> knots_unique_plus = kv_plus.unique();
    std::vector<real_t> knots_unique_gD = kv_gD_1.unique();

    std::vector<real_t> knots_unique_1 = kv_1.unique();

    knots_unique_1.erase(knots_unique_1.begin()); // First
    knots_unique_1.pop_back(); // Last

    std::vector<index_t> patch_kv_mult_plus = kv_plus.multiplicities();
    std::vector<index_t> patch_kv_mult_gD = kv_gD_1.multiplicities();

    std::vector<index_t> patch_kv_mult_1 = kv_1.multiplicities();

    if (knots_unique_plus != knots_unique_gD)
        gsInfo << "\n\nERROR: TODO \n\n";

    std::vector<real_t> knot_vector;

    index_t i_plus = 0;
    index_t i_1 = 1;
    std::vector<real_t>::iterator it_1 = knots_unique_1.begin();
    for(std::vector<real_t>::iterator it = knots_unique_plus.begin(); it != knots_unique_plus.end(); ++it, ++i_plus)
    {
        if (*it_1 == *it && it_1 != knots_unique_1.end())
        {
            index_t i_temp = 0;
            while(i_temp < math::max(patch_kv_mult_1[i_1]+p_patch_diff, math::max(patch_kv_mult_plus[i_plus]+p_plus_diff, patch_kv_mult_gD[i_plus]+p_gD_diff)))
            {
                knot_vector.push_back(*it);
                ++i_temp;
            }

            ++it_1;
            ++i_1;
        }
        else
        {
            index_t i_temp = 0;
            while(i_temp < math::max(patch_kv_mult_plus[i_plus]+p_plus_diff, patch_kv_mult_gD[i_plus]+p_gD_diff))
            {
                knot_vector.push_back(*it);
                ++i_temp;
            }
        }



    }


    kv1_result = gsKnotVector<>(knot_vector);
    // ==
    kv2_result = gsKnotVector<>(knot_vector);
*/
} // createLokalEdgeSpace

    template<short_t d,class T>
    void gsC1Argyris<d,T>::createLokalEdgeSpace(gsKnotVector<T> & kv_plus, gsKnotVector<T> & kv_minus,
                                                gsKnotVector<T> & kv_patch_1,
                                                gsKnotVector<T> & kv1_result)
    {
        index_t p_1 = math::max(kv_plus.degree(), kv_minus.degree() );

        std::vector<real_t> knots_unique_plus = kv_plus.unique(); // == kv_minus.unique()

        if (knots_unique_plus != kv_minus.unique())
            gsInfo << "ERROR LOKAL EDGE SPACE \n";

        kv1_result = gsKnotVector<>(knots_unique_plus);
        kv1_result.degreeIncrease(p_1);
        if (knots_unique_plus[1] != 1)
        {
            index_t r_plus = kv_plus.degree() - kv_plus.multiplicities()[1]; // The same for all
            index_t r_minus = kv_minus.degree() - kv_minus.multiplicities()[1]; // The same for all

            index_t r = math::min(r_plus, r_minus);

            kv1_result.increaseMultiplicity(p_1-r-1);
            //kv1_result.insert(T(0.5),2); // for g1600
        }
    } // createLokalEdgeSpace

    template<short_t d,class T>
    void gsC1Argyris<d,T>::createLokalVertexSpace(gsTensorBSplineBasis<d, T> & basis_vertex, gsTensorBSplineBasis<d, T> & basis_vertex_result)
    {
        index_t p_1 = basis_vertex.degree(0); // == basis_vertex.degree(1)
        //index_t p_tilde = math::max(p_1 - 1, 3); // TODO more general

        index_t r = m_optionList.getInt("discreteRegularity");

        if (basis_vertex.degree(0) != basis_vertex.degree(1))
            gsInfo << "ERROR LOKAL Vertex SPACE \n";

        basis_vertex_result = basis_vertex;

        //gsInfo << "basis u " << basis_vertex_result.knots(0).asMatrix() << "\n";
        //gsInfo << "basis v " << basis_vertex_result.knots(1).asMatrix() << "\n";

        basis_vertex_result.degreeElevate(p_tilde, 0); // p + \tilde{p} - 1
        basis_vertex_result.degreeElevate(p_tilde, 1); // p + \tilde{p} - 1
        basis_vertex_result.reduceContinuity(r-1);



    } // createLokalVertexSpace

} // Namespace gismo

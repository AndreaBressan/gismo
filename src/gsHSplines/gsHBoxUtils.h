
/** @file gsAdaptiveRefUtils.h

    @brief Provides generic routines for adaptive refinement.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): S. Kleiss
*/

#pragma once


#include <iostream>
// #include <gsHSplines/gsHBox.h>
#include <gsHSplines/gsHNeighborhood.h>

namespace gismo
{

template<short_t d, class T>
struct gsHBoxUtils
{
protected:
    typedef typename gsHBox<d,T>::Container         Container;
    typedef typename gsHBox<d,T>::SortedContainer   SortedContainer;
    typedef typename gsHBox<d,T>::HContainer        HContainer;
    typedef typename gsHBox<d,T>::Iterator          Iterator;
    typedef typename gsHBox<d,T>::cIterator         cIterator;
    typedef typename gsHBox<d,T>::rIterator         rIterator;
    typedef typename gsHBox<d,T>::HIterator         HIterator;
    typedef typename gsHBox<d,T>::cHIterator        cHIterator;
    typedef typename gsHBox<d,T>::rHIterator        rHIterator;

public:
	static typename gsHBox<d, T>::SortedContainer Sort(const Container & container);
	static typename gsHBox<d, T>::Container Unique(const Container & container);
	static typename gsHBox<d, T>::HContainer Unique(const HContainer & container);
	static gsHBoxContainer<d, T> Unique(const gsHBoxContainer<d,T> & container);

	static typename gsHBox<d, T>::Container Union(const Container & container1, const Container & container2);
	static typename gsHBox<d, T>::HContainer Union(const HContainer & container1, const HContainer & container2);
	static gsHBoxContainer<d, T> Union(const gsHBoxContainer<d,T> & container1, const gsHBoxContainer<d,T> & container2);
	// Takes the difference container1 \ container2
	static typename gsHBox<d, T>::Container Difference(const Container & container1, const Container & container2);
	static typename gsHBox<d, T>::HContainer Difference(const HContainer & container1, const HContainer & container2);
	static gsHBoxContainer<d, T> Difference(const gsHBoxContainer<d,T> & container1, const gsHBoxContainer<d,T> & container2);
	/**
	 * @brief      Performs an intersection. Keeps the smallest boxes in overlapping regions, can also intersect partial boxes
	 *
	 * @param[in]  container1  The container 1
	 * @param[in]  container2  The container 2
	 *
	 * @tparam     d           { description }
	 * @tparam     T           { description }
	 *
	 * @return     { description_of_the_return_value }
	 */
	static typename gsHBox<d, T>::Container Intersection(const Container & container1, const Container & container2);
	static typename gsHBox<d, T>::Container ContainedIntersection(const Container & container1, const Container & container2);
	/**
	 * @brief      Performs an intersection; only keeps the boxes that are EXACTLY the same (also level is the same)
	 *
	 * @param[in]  container1  The container 1
	 * @param[in]  container2  The container 2
	 *
	 * @tparam     d           { description }
	 * @tparam     T           { description }
	 *
	 * @return     { description_of_the_return_value }
	 */
	static typename gsHBox<d, T>::Container ExactIntersection(const Container & container1, const Container & container2);
	// typename gsHBox<d, T>::Container gsHBoxIntersection(const gsHBox<d,T> & box1, const gsHBox<d,T> & box2)

	static Container HContainer2Container( const HContainer & container );
	static HContainer Container2HContainer( const Container & container );

	/**
	 * @brief      Returns a container representation of the object.
	 *
	 * @return     Container representation of the object.
	 */
	static Container toContainer(const HContainer & container);

	/**
	 * @brief      Transforms the boxes in \a container as unit boxes
	 *
	 * @param[in]  container  A hierarchical container of boxes
	 *
	 * @return     A hierarchical container containing the unit boxes.
	 */
	static Container toUnitBoxes(const HContainer & container);

    /**
     * @brief      Transforms the boxes in \a container as unit boxes
     *
     * @param[in]  container  A hierarchical container of boxes
     *
     * @return     A hierarchical container containing the unit boxes.
     */
    static HContainer toUnitHBoxes(const HContainer & container);

    /**
     * @brief      Performs T-admissible refinement
     *
     * @param      marked  The marked boxes
     * @param[in]  m       The jump parameter
     *
     * @return     The resulting hierarchical container.
     */
    static HContainer markTadmissible(const HContainer & marked, index_t m);

    /**
     * @brief      Performs T-admissible refinement
     *
     * @param      marked  The marked boxes
     * @param[in]  m       The jump parameter
     *
     * @return     The resulting hierarchical container.
     */
    static HContainer markTadmissible(const gsHBox<d,T> & marked, index_t m);

    /**
     * @brief      Performs H-admissible refinement
     *
     * @param      marked  The marked boxes
     * @param[in]  m       The jump parameter
     *
     * @return     The resulting hierarchical container.
     */
    static HContainer markHadmissible(const HContainer & marked, index_t m);

    /**
     * @brief      Performs H-admissible refinement
     *
     * @param      marked  The marked boxes
     * @param[in]  m       The jump parameter
     *
     * @return     The resulting hierarchical container.
     */
    static HContainer markHadmissible(const gsHBox<d,T> & marked, index_t m);

    /**
     * @brief      Performs H-admissible refinement
     *
     * @param      marked  The marked boxes
     * @param[in]  m       The jump parameter
     *
     * @return     The resulting hierarchical container.
     */
    template<gsHNeighborhood _mode>
    static HContainer markAdmissible(const gsHBox<d,T> & marked, index_t m);
    template<gsHNeighborhood _mode>
    static HContainer markAdmissible(const HContainer & marked, index_t m);

    static HContainer markAdmissible(const gsHBox<d,T> & marked, index_t m);
    static HContainer markAdmissible(const HContainer & marked, index_t m);

    static bool allActive(const  Container & elements);
    static bool allActive(const HContainer & elements);

protected:
	/**
	 * @brief      Marks Recursively
	 *
	 * @param      marked  The marked boxes
	 * @param[in]  lvl     The level
	 * @param[in]  m       The jump parameter
	 *
	 * @tparam     _mode   see gsHNeighborhood for T or H
	 *
	 * @return     The resulting hierarchical container.
	 */
	template<gsHNeighborhood _mode>
    static
    typename std::enable_if<_mode==gsHNeighborhood::T || _mode==gsHNeighborhood::H, HContainer>::type
	_markRecursive(const HContainer & marked, index_t lvl, index_t m);

	template<gsHNeighborhood _mode>
    static
    typename std::enable_if<_mode!=gsHNeighborhood::T && _mode!=gsHNeighborhood::H, HContainer>::type
	_markRecursive(const HContainer & marked, index_t lvl, index_t m);
};


// template <short_t d, class T>
// void removeDuplicates( gsHBox<d,T>::Container & Container )
// {
//     typedef Container::iterator Iterator;
//     Iterator it=Container.begin();
//     while (it!=Container.end())
//     {
//         if (it)
//             Container.erase(it++);
//     }

// }

// void removeDuplicates( gsHBox<d,T>::HContainer & HContainer )
// {
//     typedef HContainer::iterator HIterator;
//     for (HIterator it!=Container.begin())

// }

// template <short_t d, class T>
// gsHBox<d, T>::Container _boxUnion(const gsHBox<d, T>::Container & container1, const gsHBox<d, T>::Container & container2) const
// {
//     gsHBox<d, T>::SortedContainer sortedResult;

//     gsHBox<d, T>::SortedContainer scontainer1(container1.begin(), container1.end());
//     gsHBox<d, T>::SortedContainer scontainer2(container2.begin(), container2.end());

//     auto comp = [](auto & a, auto & b)
//                     {
//                         return
//                         (a.level() < b.level())
//                         ||
//                         ((a.level() == b.level()) &&
//                         std::lexicographical_compare(  a.lowerIndex().begin(), a.lowerIndex().end(),
//                                                     b.lowerIndex().begin(), b.lowerIndex().end())   )
//                         ||
//                         ((a.level() == b.level()) && (a.lowerIndex() == b.lowerIndex()) &&
//                         std::lexicographical_compare(  a.upperIndex().begin(), a.upperIndex().end(),
//                                                     b.upperIndex().begin(), b.upperIndex().end())    );
//                     };

//     sortedResult.reserve(scontainer1.size() + scontainer2.size());
//     if (scontainer1.size()!=0 && scontainer2.size()!=0)
//     {
//         // First sort (otherwise union is wrong)
//         std::sort(scontainer1.begin(),scontainer1.end(),comp);
//         std::sort(scontainer2.begin(),scontainer2.end(),comp);

//         std::set_union( scontainer1.begin(),scontainer1.end(),
//                         scontainer2.begin(),scontainer2.end(),
//                         std::inserter(sortedResult,sortedResult.begin()),
//                         comp);
//     }
//     else if (scontainer1.size()!=0 && container2.size()==0)
//         sortedResult.insert(sortedResult.end(),scontainer1.begin(),scontainer1.end());
//     else if (scontainer1.size()==0 && container2.size()!=0)
//         sortedResult.insert(sortedResult.end(),scontainer2.begin(),scontainer2.end());
//     else    { /* Do nothing */ }

//     gsHBox<d, T>::Container result(sortedResult.begin(),sortedResult.end());

//     return result;
// }

template <short_t d, class T>
struct gsHBoxCompare
{
    bool operator()(const gsHBox<d,T> & a, const gsHBox<d,T> & b) const;
};

template <short_t d, class T>
struct gsHBoxEqual
{
    bool operator()(const gsHBox<d,T> & a, const gsHBox<d,T> & b) const;
};

// template <short_t d, class T>
// struct gsHBoxOverlaps
// {
//     bool operator()(const gsHBox<d,T> & a, const gsHBox<d,T> & b) const;
// };

template <short_t d, class T>
struct gsHBoxContains
{
    bool operator()(const gsHBox<d,T> & a, const gsHBox<d,T> & b) const;
};

template <short_t d, class T>
struct gsHBoxIsContained
{
    bool operator()(const gsHBox<d,T> & a, const gsHBox<d,T> & b) const;
};


} // namespace gismo

#ifndef GISMO_BUILD_LIB
#include GISMO_HPP_HEADER(gsHBoxUtils.hpp)
#endif

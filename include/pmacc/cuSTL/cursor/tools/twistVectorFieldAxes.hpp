/* Copyright 2013-2021 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/accessor/TwistAxesAccessor.hpp"

#include "pmacc/cuSTL/cursor/navigator/compile-time/TwistAxesNavigator.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace tools
        {
            /** Returns a new cursor which looks like a vector field rotated version of the one passed
             *
             * When rotating a vector field in physics the coordinate system and the vectors themselves
             * have to be rotated. This is the idea behind this function. It is assuming that the cursor
             * which is passed returns in its access call a vector type of the same dimension as in
             * the jumping call. In other words, the field and the vector have the same dimension.
             *
             * e.g.: new_cur = twistVectorFieldAxes<math::CT::Int<1,2,0> >(cur); // x -> y, y -> z, z -> x
             *
             * @tparam T_Permutation compile-time vector (pmacc::math::CT::Int) that describes the mapping.
             * x-axis -> T_Permutation::at<0>, y-axis -> T_Permutation::at<1>, ...
             * @param cursor cursor to permute
             * @param navigatorPermutation compile time permutation vector for the navigator
             * @param accessorPermutation compile time permutation vector for the accessor
             */
            template<typename T_NavigatorPerm, typename T_Cursor, typename T_AccessorPerm = T_NavigatorPerm>
            HDINLINE Cursor<
                TwistAxesAccessor<T_Cursor, T_AccessorPerm>,
                CT::TwistAxesNavigator<T_NavigatorPerm>,
                T_Cursor>
            twistVectorFieldAxes(
                const T_Cursor& cursor,
                T_NavigatorPerm /*navigatorPermutation*/ = {},
                T_AccessorPerm /*accessorPermutation*/ = {})
            {
                return {
                    TwistAxesAccessor<T_Cursor, T_AccessorPerm>(),
                    CT::TwistAxesNavigator<T_NavigatorPerm>(),
                    cursor};
            }
        } // namespace tools
    } // namespace cursor
} // namespace pmacc

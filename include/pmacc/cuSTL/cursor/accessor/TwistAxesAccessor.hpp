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

#include "pmacc/math/vector/TwistComponents.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace cursor
    {
        template<typename TCursor, typename Axes>
        struct TwistAxesAccessor
        {
            using Reference = decltype(math::twistComponents<Axes>(*std::declval<TCursor&>()));

            /** Returns a reference to the result of '*cursor' (with twisted axes).
             *
             * Be aware that the underlying cursor must not be a temporary object if '*cursor'
             * refers to something inside the cursor.
             */
            HDINLINE Reference operator()(TCursor& cursor)
            {
                return math::twistComponents<Axes>(*cursor);
            }

            ///\todo: implement const method here with a const TCursor& argument and 'Reference' as return type.
        };

    } // namespace cursor
} // namespace pmacc

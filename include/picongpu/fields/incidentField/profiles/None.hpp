/* Copyright 2022 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/None.def"

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace detail
            {
                /** Get type of incident field E functor for the none profile type
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE<profiles::None, T_axis, T_direction>
                {
                    using type = ZeroFunctor;
                };

                /** Get type of incident field B functor for the none profile type
                 *
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 */
                template<uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB<profiles::None, T_axis, T_direction>
                {
                    using type = ZeroFunctor;
                };
            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
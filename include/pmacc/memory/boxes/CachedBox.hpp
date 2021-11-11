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

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/SharedBox.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace CachedBox
    {
        template<
            uint32_t Id_,
            SharedDataBoxMapping T_Mapping,
            typename ValueType_,
            class BlockDescription_,
            typename T_Acc>
        DINLINE auto create(T_Acc const& acc, const BlockDescription_ block)
        {
            using OffsetOrigin = typename BlockDescription_::OffsetOrigin;
            using Type = DataBox<SharedBox<ValueType_, typename BlockDescription_::FullSuperCellSize, Id_, T_Mapping>>;
            return Type{Type::init(acc)}.shift(DataSpace<OffsetOrigin::dim>{OffsetOrigin::toRT()});
        }

        template<
            uint32_t Id_,
            SharedDataBoxMapping T_Mapping,
            typename ValueType_,
            class BlockDescription_,
            typename T_Acc>
        DINLINE auto create(T_Acc const& acc, const ValueType_& value, const BlockDescription_ block)
        {
            return create<Id_, ValueType_, BlockDescription_, T_Mapping>(acc);
        }
    } // namespace CachedBox
} // namespace pmacc

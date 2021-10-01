/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
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

#include "SharedBox.hpp"
#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/shared/Allocate.hpp"

#include <llama/llama.hpp>

namespace pmacc
{
    namespace detail
    {
        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<1> const& idx = {})
        {
            return db[idx.x()];
        }

        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<2> const& idx = {})
        {
            return db[idx.y()][idx.x()];
        }

        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<3> const& idx = {})
        {
            return db[idx.z()][idx.y()][idx.x()];
        }
    } // namespace detail

    template<typename Base>
    struct DataBox : Base
    {
        HDINLINE DataBox() = default;

        HDINLINE DataBox(Base base) : Base{std::move(base)}
        {
        }

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {}) const
        {
            ///@todo(bgruber): inline and replace this by if constexpr in C++17
            return detail::access(*this, idx);
        }

        HDINLINE DataBox shift(DataSpace<Base::Dim> const& offset) const
        {
            DataBox result(*this);
            result.fixedPointer = &((*this) (offset));
            return result;
        }

        HDINLINE DataBox<typename Base::ReducedType> reduceZ(const int zOffset) const
        {
            return Base::reduceZ(zOffset);
        }
    };

    namespace internal
    {
        template<typename X, typename Y, typename Z>
        HDINLINE auto toAE(math::CT::Vector<X, Y, Z>)
        {
            constexpr auto dim = math::CT::Vector<X, Y, Z>::dim;
            if constexpr(dim == 1)
                return llama::ArrayExtents<X::value>{};
            else if constexpr(dim == 2)
                return llama::ArrayExtents<Y::value, X::value>{};
            else if constexpr(dim == 3)
                return llama::ArrayExtents<Z::value, Y::value, X::value>{};
        }

        // LLAMA and DataSpace indices have the same semantic, fast moving index is first.
        HDINLINE auto toAI(DataSpace<1> idx)
        {
            return llama::ArrayIndex<1>{static_cast<std::size_t>(idx[0])};
        }

        HDINLINE auto toAI(DataSpace<2> idx)
        {
            return llama::ArrayIndex<2>{static_cast<std::size_t>(idx[1]), static_cast<std::size_t>(idx[0])};
        }

        HDINLINE auto toAI(DataSpace<3> idx)
        {
            return llama::ArrayIndex<3>{
                static_cast<std::size_t>(idx[2]),
                static_cast<std::size_t>(idx[1]),
                static_cast<std::size_t>(idx[0])};
        }

        template<typename DataBox>
        struct LlamaAccessor
        {
            static constexpr auto dim = DataBox::Dim;

            using Reference = decltype(std::declval<DataBox&>()(DataSpace<dim>{}));
            using ValueType = typename DataBox::ValueType;

            HDINLINE decltype(auto) operator()(DataSpace<dim> idx) const
            {
                return db(idx);
            }

            DataBox db; // A cursor can outlive its DataBox, so it must carry on a copy of it.
        };

        template<std::size_t Dim>
        struct DataspaceNavigator
        {
            template<typename Jump>
            HDINLINE auto operator()(DataSpace<Dim> ds, const Jump& jump) const
            {
                return ds + jump;
            }
        };

        template<typename DataBox>
        using LlamaCursor
            = cursor::Cursor<LlamaAccessor<DataBox>, DataspaceNavigator<DataBox::Dim>, DataSpace<DataBox::Dim>>;
    } // namespace internal

    // handle DataBox wrapping SharedBox with LLAMA
    template<typename T_TYPE, class T_SizeVector, SharedDataBoxMapping T_Mapping, uint32_t T_id, uint32_t T_dim>
    struct DataBox<SharedBox<T_TYPE, T_SizeVector, T_id, T_Mapping, T_dim>>
    {
        using SB = SharedBox<T_TYPE, T_SizeVector, T_id, T_Mapping, T_dim>;

        static constexpr std::uint32_t Dim = T_dim;
        using ValueType = T_TYPE;
        using RefValueType = ValueType&;
        using Size = T_SizeVector;

        using SplitRecordDim = llama::TransformLeaves<T_TYPE, math::ReplaceVector>;
        using ArrayExtents = decltype(internal::toAE(T_SizeVector{}));

        using AoS = llama::mapping::AoS<ArrayExtents, T_TYPE>;
        using AoSSplit = llama::mapping::AoS<ArrayExtents, SplitRecordDim>;
        using AoSSplitFortran
            = llama::mapping::AoS<ArrayExtents, SplitRecordDim, true, llama::mapping::LinearizeArrayDimsFortran>;
        using SoA = llama::mapping::SoA<ArrayExtents, T_TYPE, false>;
        using SoASplit = llama::mapping::SoA<ArrayExtents, SplitRecordDim, false>;
        using SoASplitFortran
            = llama::mapping::SoA<ArrayExtents, SplitRecordDim, false, llama::mapping::LinearizeArrayDimsFortran>;
        using Mapping = std::conditional_t<
            T_Mapping == SharedDataBoxMapping::SoA,
            SoA,
            std::conditional_t<
                T_Mapping == SharedDataBoxMapping::SoASplitVector,
                SoASplit,
                std::conditional_t<
                    T_Mapping == SharedDataBoxMapping::AoS,
                    AoS,
                    std::conditional_t<
                        T_Mapping == SharedDataBoxMapping::AoSSplitVector,
                        AoSSplit,
                        std::conditional_t<
                            T_Mapping == SharedDataBoxMapping::AoSSplitVectorFortran,
                            AoSSplitFortran,
                            std::conditional_t<
                                T_Mapping == SharedDataBoxMapping::SoASplitVectorFortran,
                                SoASplitFortran,
                                void>>>>>>;

        using View = llama::View<Mapping, std::byte*>;

        View view;

        HDINLINE DataBox() = default;

        HDINLINE DataBox(SB sb)
            : view{
                Mapping{{}},
                llama::Array{const_cast<std::byte*>(reinterpret_cast<const std::byte*>(sb.fixedPointer))}}
        {
        }

        HDINLINE decltype(auto) operator()(DataSpace<T_dim> idx = {}) const
        {
            auto&& v = const_cast<View&>(view)(internal::toAI(idx + offset));
            using ReturnType = std::remove_reference_t<decltype(v)>;
            if constexpr(math::isVector<T_TYPE> && llama::is_VirtualRecord<ReturnType>)
                return math::makeVectorWithLlamaStorage<T_TYPE>(v);
            else
                return v;
        }

        HDINLINE auto toCursor() const
        {
            return internal::LlamaCursor<DataBox>{
                internal::LlamaAccessor<DataBox>{*this},
                internal::DataspaceNavigator<T_dim>{},
                DataSpace<T_dim>{}};
        }

        HDINLINE DataBox shift(const DataSpace<T_dim>& offset) const
        {
            DataBox result(*this);
            result.offset += offset;
            return result;
        }

        template<typename T_Acc>
        static DINLINE SB init(T_Acc const& acc)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Size>::type::value>>(acc);
            return {mem_sh.data()};
        }

        DataSpace<T_dim> offset{};
    };
} // namespace pmacc
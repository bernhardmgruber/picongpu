/* Copyright 2013-2021 Rene Widera, Alexander Grund
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

#include "pmacc/math/MapTuple.hpp"
#include "pmacc/meta/GetKeyFromAlias.hpp"
#include "pmacc/meta/conversion/OperateOnSeq.hpp"
#include "pmacc/meta/conversion/SeqToMap.hpp"
#include "pmacc/particles/ParticleDescription.hpp"
#include "pmacc/particles/boostExtension/InheritLinearly.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/dataTypes/Particle.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/types.hpp"

#include <boost/mp11.hpp>
#include <boost/mp11/mpl.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/map.hpp>
#include <boost/utility/result_of.hpp>

#include <llama/llama.hpp>

namespace pmacc
{
    namespace pmath = pmacc::math;

    namespace detail
    {
        template<typename PICValueType>
        using MakeLlamaField = llama::Field<PICValueType, typename traits::Resolve<PICValueType>::type::type>;

        template<typename ValueTypeSeq>
        using ValueTypeSeqToList =
            typename boost::mpl::copy<ValueTypeSeq, boost::mpl::back_inserter<boost::mp11::mp_list<>>>::type;

        template<typename ValueTypeSeq>
        using RecordDimFromValueTypeSeq = boost::mp11::
            mp_rename<boost::mp11::mp_transform<MakeLlamaField, ValueTypeSeqToList<ValueTypeSeq>>, llama::Record>;

        template<typename T>
        inline constexpr auto nonTypeArgOf = nullptr; // = delete;

        template<template<auto> typename T, auto I>
        inline constexpr auto nonTypeArgOf<T<I>> = I;

        template<typename T_CreatePairOperator, typename T_ParticleDescription>
        struct ViewHolder
        {
            static constexpr std::size_t particlesPerFrame
                = nonTypeArgOf<T_CreatePairOperator>; // T_ParticleDescription::SuperCellSize
            static constexpr ParticleFrameMapping frameMapping = T_ParticleDescription::frameMapping;

            using RawRecordDim = RecordDimFromValueTypeSeq<typename T_ParticleDescription::ValueTypeSeq>;
            using RecordDim = llama::TransformLeaves<RawRecordDim, pmath::ReplaceVector>;
            using ArrayExtents = llama::ArrayExtents<particlesPerFrame>;

            using AoS = llama::mapping::AoS<
                ArrayExtents,
                RawRecordDim,
                true,
                llama::mapping::LinearizeArrayDimsCpp,
                llama::mapping::FlattenRecordDimMinimizePadding>;
            using AoSSplit = llama::mapping::AoS<
                ArrayExtents,
                RecordDim,
                true,
                llama::mapping::LinearizeArrayDimsCpp,
                llama::mapping::FlattenRecordDimMinimizePadding>;
            using SoA = llama::mapping::SoA<ArrayExtents, RawRecordDim, false>;
            using SoASplit = llama::mapping::SoA<ArrayExtents, RecordDim, false>;
            using AoSoA16 = llama::mapping::AoSoA<ArrayExtents, RawRecordDim, 16>;
            using AoSoA32 = llama::mapping::AoSoA<ArrayExtents, RawRecordDim, 32>;
            using AoSoA64 = llama::mapping::AoSoA<ArrayExtents, RawRecordDim, 64>;
            using One = llama::mapping::One<ArrayExtents, RawRecordDim>;
            using Mapping = boost::mp11::mp_if_c<
                particlesPerFrame == 1,
                One,
                boost::mp11::mp_at_c<
                    boost::mp11::mp_list<AoS, AoSSplit, SoA, SoASplit, AoSoA16, AoSoA32, AoSoA64>,
                    static_cast<int>(frameMapping)>>;
            using LlamaViewType = decltype(llama::allocView(
                Mapping{ArrayExtents{}},
                llama::bloballoc::Stack<Mapping{{}}.blobSize(0)>{}));
            LlamaViewType view;
        };

        template<typename VirtualRecord>
        struct LlamaParticleAttribute
        {
            template<typename OtherVirtualRecord>
            auto operator=(const LlamaParticleAttribute<OtherVirtualRecord>& lpa) -> LlamaParticleAttribute&
            {
                vr = lpa.vr;
                return *this;
            }

            template<typename OtherVirtualRecord>
            auto operator=(LlamaParticleAttribute<OtherVirtualRecord>&& lpa) -> LlamaParticleAttribute&
            {
                vr = lpa.vr;
                return *this;
            }

            template<typename T>
            auto operator=(T&& t) -> LlamaParticleAttribute&
            {
                vr.store(std::forward<T>(t));
                return *this;
            }

            template<typename T>
            operator T() const
            {
                return vr.template loadAs<T>();
            }

            VirtualRecord vr;
        };
    } // namespace detail

    /** Frame is a storage for arbitrary number >0 of Particles with attributes
     *
     * @tparam T_CreatePairOperator unary template operator to create a boost pair
     *                              from single type ( pair<name,dataType> )
     *                              @see MapTupel
     * @tparam T_ValueTypeSeq sequence with value_identifier
     * @tparam T_MethodsList sequence of classes with particle methods
     *                       (e.g. calculate mass, gamma, ...)
     * @tparam T_Flags sequence with identifiers to add flags on a frame
     *                 (e.g. useSolverXY, calcRadiation, ...)
     */
    template<typename T_CreatePairOperator, typename T_ParticleDescription>
    struct Frame;

    template<typename T_CreatePairOperator, typename T_ParticleDescription>
    struct Frame
        : public InheritLinearly<typename T_ParticleDescription::MethodsList>
        , public detail::ViewHolder<T_CreatePairOperator, T_ParticleDescription>
        , public InheritLinearly<typename OperateOnSeq<
              typename T_ParticleDescription::FrameExtensionList,
              bmpl::apply1<bmpl::_1, Frame<T_CreatePairOperator, T_ParticleDescription>>>::type>
    {
        using ParticleDescription = T_ParticleDescription;
        using Name = typename ParticleDescription::Name;
        using SuperCellSize = typename ParticleDescription::SuperCellSize;
        using ValueTypeSeq = typename ParticleDescription::ValueTypeSeq;
        using MethodsList = typename ParticleDescription::MethodsList;
        using FlagList = typename ParticleDescription::FlagsList;
        using FrameExtensionList = typename ParticleDescription::FrameExtensionList;
        using ThisType = Frame<T_CreatePairOperator, ParticleDescription>;
        using Map = typename SeqToMap<typename T_ParticleDescription::ValueTypeSeq, T_CreatePairOperator>::type;

        /* type of a single particle*/
        using ParticleType = pmacc::Particle<ThisType>;

        /** access the Nth particle*/
        HDINLINE ParticleType operator[](const uint32_t idx)
        {
            return ParticleType(*this, idx);
        }

        /** access the Nth particle*/
        HDINLINE const ParticleType operator[](const uint32_t idx) const
        {
            return ParticleType(*this, idx);
        }

        template<typename Frame, typename T_Key>
        static HDINLINE decltype(auto) at(Frame& f, uint32_t i, const T_Key key)
        {
            using Key = typename GetKeyFromAlias<ValueTypeSeq, T_Key, errorHandlerPolicies::ThrowValueNotFound>::type;
            auto&& v = f.view(i)(Key{});

            using OldDstType = typename traits::Resolve<Key>::type::type;
            using ReturnType = std::remove_reference_t<decltype(v)>;

            if constexpr(pmath::isVector<OldDstType> && llama::is_VirtualRecord<ReturnType>)
                return pmath::makeVectorWithLlamaStorage<OldDstType>(v);
            else if constexpr(llama::is_VirtualRecord<ReturnType>)
                return detail::LlamaParticleAttribute<ReturnType>{v};
            else
                return v;
        }

        template<typename T_Key>
        HDINLINE decltype(auto) get(uint32_t i, const T_Key)
        {
            return at(*this, i, T_Key{});
        }

        template<typename T_Key>
        HDINLINE decltype(auto) get(uint32_t i, const T_Key) const
        {
            return at(*this, i, T_Key{});
        }

        HINLINE static std::string getName()
        {
            return Name::str();
        }
    };

    namespace traits
    {
        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct HasIdentifier<pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>;

        public:
            using ValueTypeSeq = typename FrameType::ValueTypeSeq;
            /* if T_IdentifierName is void_ than we have no T_IdentifierName in our Sequence.
             * check is also valid if T_Key is a alias
             */
            using SolvedAliasName = typename GetKeyFromAlias<ValueTypeSeq, T_IdentifierName>::type;

            using type = bmpl::contains<ValueTypeSeq, SolvedAliasName>;
        };

        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct HasFlag<pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>;
            using SolvedAliasName = typename GetFlagType<FrameType, T_IdentifierName>::type;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = bmpl::contains<FlagList, SolvedAliasName>;
        };

        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct GetFlagType<pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = typename GetKeyFromAlias<FlagList, T_IdentifierName>::type;
        };

    } // namespace traits

} // namespace pmacc

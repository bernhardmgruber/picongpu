/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera
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

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/result_of_Functor.hpp>
#include <pmacc/types.hpp>
#include "experimental/simd"
#include <type_traits>

namespace stdx = std::experimental;

// forward declaration
namespace picongpu
{
    struct AssignedTrilinearInterpolation;
} // namespace picongpu

namespace pmacc
{
    namespace result_of
    {
        template<typename T_Cursor>
        struct Functor<picongpu::AssignedTrilinearInterpolation, T_Cursor>
        {
            using type = typename T_Cursor::ValueType;
        };

    } // namespace result_of
} // namespace pmacc

namespace picongpu
{
    struct AssignedTrilinearInterpolation
    {
        /** Does a trilinear field-to-point interpolation for
         * arbitrary assignment function and arbitrary field_value types.
         *
         * @tparam T_begin lower margin for interpolation
         * @tparam T_end upper margin for interpolation
         * @tparam T_AssignmentFunction type of shape functors
         *
         * @param cursor cursor pointing to the field
         * @param shapeFunctors Array with d shape functors, where d is the dimensionality of the field represented by
         *                      cursor. The shape functor must have the interface to call
         *                      `operator()(relative_grid_point)` and return the assignment value for the given grid
         *                      point.
         * @return sum over: field_value * assignment
         *
         * interpolate on grid points in range [T_begin;T_end]
         *
         * @{
         */
        template<int T_begin, int T_end, typename T_Cursor, typename T_AssignmentFunction>
        HDINLINE static auto interpolate(
            const T_Cursor& cursor,
            const pmacc::memory::Array<T_AssignmentFunction, 3>& shapeFunctors)
        {
            [[maybe_unused]] constexpr auto iterations = T_end - T_begin + 1;

            using type = decltype(*cursor(0, 0, 0) * shapeFunctors[0](0));

#if 0
            auto result_z = type(0.0);
            PMACC_UNROLL(iterations)
            for(int z = T_begin; z <= T_end; ++z)
            {
                auto result_y = type(0.0);
                PMACC_UNROLL(iterations)
                for(int y = T_begin; y <= T_end; ++y)
                {
                    auto result_x = type(0.0);
                    PMACC_UNROLL(iterations)
                    for(int x = T_begin; x <= T_end; ++x)
                        /* a form factor is the "amount of particle" that is affected by this cell
                         * so we have to sum over: cell_value * form_factor
                         */
                        result_x += *cursor(x, y, z) * shapeFunctors[0](x);

                    result_y += result_x * shapeFunctors[1](y);
                }
                result_z += result_y * shapeFunctors[2](z);
            }
            return result_z;
#else
            static_assert(iterations == 4, "Use particle shape PQS!");

            /* The implementation assumes that x is the fastest moving index to iterate over contiguous memory
             * e.g. a row, to optimize memory fetch operations.
             */
            using Simd = stdx::fixed_size_simd<type, 16>;
            //using Simd = stdx::native_simd<type>;
            static_assert(Simd{}.size() == 16);

            const auto sfx = Simd(
                [&](auto ic) LLAMA_LAMBDA_INLINE
                {
                    constexpr auto i = static_cast<int>(decltype(ic)::value);
                    return shapeFunctors[0](T_begin + i % 4);
                });

            const auto sfy = Simd(
                [&](auto ic) LLAMA_LAMBDA_INLINE
                {
                    constexpr auto i = static_cast<int>(decltype(ic)::value);
                    return shapeFunctors[1](T_begin + i / 4);
                });

            auto result_z = Simd(0);

            PMACC_UNROLL(iterations)
            for(int z = T_begin; z <= T_end; ++z)
            {
                const auto fieldData = Simd(
                    [&](auto ic) LLAMA_LAMBDA_INLINE
                    {
                        constexpr auto i = static_cast<int>(decltype(ic)::value);
                        return *cursor(T_begin + i % 4, T_begin + i / 4, z);
                    });

                Simd result_y = fieldData * sfx * sfy;
                result_z += result_y * Simd(shapeFunctors[2](z));
            }

            return stdx::reduce(result_z);
#endif
        }

        /** Implementation for 2D position*/
        template<int T_begin, int T_end, class T_Cursor, class T_AssignmentFunction>
        HDINLINE static auto interpolate(
            T_Cursor const& cursor,
            const pmacc::memory::Array<T_AssignmentFunction, 2>& shapeFunctors)
        {
            [[maybe_unused]] constexpr int iterations = T_end - T_begin + 1;

            using type = decltype(*cursor(0, 0) * shapeFunctors[0](0));
            /* The implementation assumes that x is the fastest moving index to iterate over contiguous memory
             * e.g. a row, to optimize memory fetch operations.
             */
            auto result_y = type(0.0);
            PMACC_UNROLL(iterations)
            for(int y = T_begin; y <= T_end; ++y)
            {
                auto result_x = type(0.0);
                PMACC_UNROLL(iterations)
                for(int x = T_begin; x <= T_end; ++x)
                    // a form factor is the "amount of particle" that is affected by this cell
                    // so we have to sum over: cell_value * form_factor
                    result_x += *cursor(x, y) * shapeFunctors[0](x);

                result_y += result_x * shapeFunctors[1](y);
            }
            return result_y;
        }

        static auto getStringProperties() -> pmacc::traits::StringProperty
        {
            pmacc::traits::StringProperty propList("name", "uniform");
            return propList;
        }
    };

    //! @}

} // namespace picongpu

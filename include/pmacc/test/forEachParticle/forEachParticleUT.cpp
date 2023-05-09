/* Copyright 2016-2022 Alexander Grund
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

#include <picongpu/simulation_defines.hpp>

#include <pmacc/meta/String.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/ParticlesBase.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <picongpu/param/mallocMC.param>

namespace pmacc::test
{
    constexpr auto simDim = 3;

    void runKernel()
    {
        using SuperCellSize = typename math::CT::shrinkTo<math::CT::Int<8, 8, 4>, simDim>::type;
        using MappingDesc = MappingDescription<simDim, SuperCellSize>;

        using T_Name = PMACC_CSTRING("e");
        using T_Flags = MakeSeq_t<
            //                particlePusher<UsedParticlePusher>,
            //                shape<UsedParticleShape>,
            //                interpolation<UsedField2Particle>,
            //                current<UsedParticleCurrentSolver>,
            //                massRatio<MassRatioElectrons>,
            //                chargeRatio<ChargeRatioElectrons>
            >;

        using T_Attributes
            = MakeSeq_t<picongpu::position<picongpu::position_pic>, picongpu::momentum, picongpu::weighting>;

        using ParticleDescription = pmacc::ParticleDescription<
            T_Name,
            SuperCellSize,
            T_Attributes,
            T_Flags,
            pmacc::HandleGuardRegion<
                pmacc::particles::policies::ExchangeParticles,
                pmacc::particles::policies::DoNothing>>;


        MappingDesc* cellDescription = nullptr;

        using BufferType = ParticlesBuffer<
            ParticleDescription,
            typename MappingDesc::SuperCellSize,
            picongpu::DeviceHeap,
            MappingDesc::Dim>;

        using FrameType = typename BufferType::FrameType;
        using FrameTypeBorder = typename BufferType::FrameTypeBorder;
        using ParticlesBoxType = typename BufferType::ParticlesBoxType;

        //        const std::shared_ptr<DeviceHeap>& deviceHeap;

        BufferType* particlesBuffer = nullptr;
        //= BufferType(
        //            deviceHeap,
        //            cellDescription->getGridLayout().getDataSpace(),
        //            MappingDesc::SuperCellSize::toRT());


        auto mapperFactory = AreaMapperFactory<CORE + BORDER + GUARD>{};
        auto const mapper = mapperFactory(*cellDescription);

        auto workerCfg = lockstep::makeWorkerCfg(typename FrameType::SuperCellSize{});

        PMACC_LOCKSTEP_KERNEL(KernelFillGaps{}, workerCfg)
        (mapper.getGridDim())(particlesBuffer->getDeviceParticleBox(), mapper);
    }
    //
    //    template<class T_Method>
    //    void runTest(uint32_t numSamples)
    //    {
    //        typedef pmacc::random::RNGProvider<2, T_Method> RNGProvider;
    //
    //        const std::string rngName = RNGProvider::RNGMethod::getName();
    //        std::cout << std::endl
    //                  << "Running test for " << rngName << " with " << numSamples << " samples per cell" <<
    //                  std::endl;
    //        // Size of the detector
    //        const Space2D size(256, 256);
    //        // Size of the rng provider (= number of states used)
    //        const Space2D rngSize(256, 256);
    //
    //        pmacc::HostDeviceBuffer<uint32_t, 2> detector(size);
    //        auto rngProvider = new RNGProvider(rngSize);
    //
    //        pmacc::Environment<>::get().DataConnector().share(std::shared_ptr<pmacc::ISimulationData>(rngProvider));
    //        rngProvider->init(0x42133742);
    //
    //        generateRandomNumbers(rngSize, numSamples, detector.getDeviceBuffer(), GetRanidx<RNGProvider>());
    //
    //        detector.deviceToHost();
    //        auto box = detector.getHostBuffer().getDataBox();
    //        // Write data to file
    //        std::ofstream dataFile((rngName + "_data.txt").c_str());
    //        for(int y = 0; y < size.y(); y++)
    //        {
    //            for(int x = 0; x < size.x(); x++)
    //                dataFile << box(Space2D(x, y)) << ",";
    //        }
    //        writePGM(rngName + "_img.pgm", detector.getHostBuffer());
    //
    //        uint64_t totalNumSamples = 0;
    //        double mean = 0;
    //        uint32_t maxVal = 0;
    //        uint32_t minVal = static_cast<uint32_t>(-1);
    //        for(int y = 0; y < size.y(); y++)
    //        {
    //            for(int x = 0; x < size.x(); x++)
    //            {
    //                Space2D idx(x, y);
    //                uint32_t val = box(idx);
    //                if(val > maxVal)
    //                    maxVal = val;
    //                if(val < minVal)
    //                    minVal = val;
    //                totalNumSamples += val;
    //                mean += pmacc::math::linearize(size.shrink<1>(1), idx) * static_cast<uint64_t>(val);
    //            }
    //        }
    //        PMACC_ASSERT(totalNumSamples == uint64_t(rngSize.productOfComponents()) * uint64_t(numSamples));
    //        // Expected value: (n-1)/2
    //        double Ex = (size.productOfComponents() - 1) / 2.;
    //        // Variance: (n^2 - 1) / 12
    //        double var = (cupla::pow(static_cast<double>(size.productOfComponents()), 2.0) - 1.) / 12.;
    //        // Mean value
    //        mean /= totalNumSamples;
    //        double errSq = 0;
    //        // Calc standard derivation
    //        for(int y = 0; y < size.y(); y++)
    //        {
    //            for(int x = 0; x < size.x(); x++)
    //            {
    //                Space2D idx(x, y);
    //                uint32_t val = box(idx);
    //                errSq += val
    //                    * cupla::pow(static_cast<double>(pmacc::math::linearize(size.shrink<1>(1), idx) -
    //                    mean), 2.0);
    //            }
    //        }
    //        double stdDev = sqrt(errSq / (totalNumSamples - 1));
    //
    //        uint64_t avg = totalNumSamples / size.productOfComponents();
    //        std::cout << "  Samples: " << totalNumSamples << std::endl;
    //        std::cout << "      Min: " << minVal << std::endl;
    //        std::cout << "      Max: " << maxVal << std::endl;
    //        std::cout << " Avg/cell: " << avg << std::endl;
    //        std::cout << "     E(x): " << Ex << std::endl;
    //        std::cout << "     mean: " << mean << std::endl;
    //        std::cout << "   dev(x): " << sqrt(var) << std::endl;
    //        std::cout << " std. dev: " << stdDev << std::endl;
    //    }
} // namespace pmacc::test

int main(int argc, char** argv)
{
    //    using namespace pmacc;
    //    using namespace test::random;
    //
    //    Environment<2>::get().initDevices(Space2D::create(1), Space2D::create(0));
    //
    //    const uint32_t numSamples = (argc > 1) ? atoi(argv[1]) : 100;
    //
    //    runTest<random::methods::AlpakaRand<cupla::Acc>>(numSamples);
    //
    //    /* finalize the pmacc context */
    //    Environment<>::get().finalize();
}

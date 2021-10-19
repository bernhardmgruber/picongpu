#pragma once

#include <llama/llama.hpp>

namespace pmacc::detail
{
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
}

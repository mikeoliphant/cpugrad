#pragma once

#include <memory>
#include <memory_resource>

#include "ChannelBuffer.h"

using namespace NeuralAudio;

namespace cpugrad
{
    constexpr size_t SIMD_ALIGN = 32;

    template <typename T>
    class BatchBufferArenaT
    {
	    public:
            explicit BatchBufferArenaT() :
                stepArena(),
                scratchPool()
            {
            }

		    template <int Channels>
            ChannelBufferDynamic<T, Channels> GetBuffer(size_t numCols)
		    {
                ChannelBufferDynamic<T, Channels> buf(static_cast<T*>(stepArena.allocate(Channels * numCols * sizeof(T), SIMD_ALIGN)), numCols);

                return buf;
		    }

            template <int Channels>
            ChannelBufferDynamic<T, Channels> GetScratchBuffer(size_t numCols)
            {
                ChannelBufferDynamic<T, Channels> buf(static_cast<T*>(scratchPool.allocate(Channels * numCols * sizeof(T), SIMD_ALIGN)), numCols);

                return buf;
            }

            template <int Channels>
            void FreeScratchBuffer(ChannelBufferDynamic<T, Channels>& buf)
            {
                scratchPool.deallocate(buf.GetData(), buf.GetSize() * sizeof(T), SIMD_ALIGN);
            }

            void ReleaseScratch()
            {
                scratchPool.release();
            }

	    private:
            std::pmr::monotonic_buffer_resource stepArena; // data that persists (ie: forward pass intermediate outputs)
            std::pmr::unsynchronized_pool_resource scratchPool; // temporary data (forward pass scratch buffers and backward gradient buffers)

    };
}
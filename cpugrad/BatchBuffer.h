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
            explicit BatchBufferArenaT(size_t maxScratchSize) :
                stepArena(),
                scratchPool(),
                scratchIndex(0),
                maxScratchSize(maxScratchSize)
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
                void* data = nullptr;

                if (scratchIndex == (scratchPool.size()))
                {
                    scratchPool.push_back(stepArena.allocate(maxScratchSize * sizeof(T), SIMD_ALIGN));
                }

                data = scratchPool[scratchIndex];

                scratchIndex++;

                ChannelBufferDynamic<T, Channels> buf(static_cast<T*>(data), numCols);

                return buf;
            }

            template <int Channels>
            void FreeScratchBuffer(ChannelBufferDynamic<T, Channels>& buf)
            {
                scratchIndex--;
            }

	    private:
            std::pmr::monotonic_buffer_resource stepArena;
            std::vector<void*> scratchPool;
            size_t scratchIndex;
            size_t maxScratchSize;
    };
}
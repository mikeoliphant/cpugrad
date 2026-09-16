#pragma once

#include <memory>
#include <memory_resource>

#include "ChannelBuffer.h"

using namespace NeuralAudio;

namespace cpugrad
{
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
                ChannelBufferDynamic<T, Channels> buf(static_cast<T*>(stepArena.allocate(Channels * numCols * sizeof(T), CHANNEL_BUFFER_ALIGN)), numCols);

                return buf;
		    }

            template <int Channels>
            ChannelBufferDynamic<T, Channels> GetScratchBuffer(size_t numCols)
            {
                //ChannelBufferDynamic<T, Channels> buf(static_cast<T*>(scratchPool.allocate(Channels * numCols * sizeof(T), SIMD_ALIGN)), numCols);
                
                if ((Channels * numCols) > maxScratchSize)
                    throw std::runtime_error("Tried to allocate a buffer > maxScratchSize");

                if (scratchIndex == (scratchPool.size()))
                {
                    scratchPool.push_back(stepArena.allocate(maxScratchSize * sizeof(T), CHANNEL_BUFFER_ALIGN));
                }

                void* data = scratchPool[scratchIndex];

                scratchIndex++;

                ChannelBufferDynamic<T, Channels> buf(static_cast<T*>(data), numCols);

                return buf;
            }

            template <int Channels>
            void FreeScratchBuffer(ChannelBufferDynamic<T, Channels>& buf)
            {
                //scratchPool.deallocate(buf.GetData(), buf.GetSize() * sizeof(T), CHANNEL_BUFFER_ALIGN);
                
                if (scratchIndex == 0)
                    throw std::runtime_error("Tried to free too many scratch buffers");

                if (buf.GetDataConst() != scratchPool[scratchIndex - 1])
                    throw std::runtime_error("Freed a scratch buffer that was not top of the stack");

                scratchIndex--;
            }

	    private:
            std::pmr::monotonic_buffer_resource stepArena;
            //std::pmr::unsynchronized_pool_resource scratchPool;
            std::vector<void*> scratchPool;
            size_t scratchIndex;
            size_t maxScratchSize;
    };
}
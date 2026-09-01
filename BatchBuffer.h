#pragma once

#include <memory>
#include <memory_resource>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "ChannelBuffer.h"

using namespace NeuralAudio;

constexpr size_t SIMD_ALIGN = 32;

template <typename T>
class BatchBufferArenaT
{
	public:
        explicit BatchBufferArenaT(size_t maxCapacity) :
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

        void Release()
        {
            scratchPool.release();
            //stepArena.release();
        }

	private:
        std::pmr::monotonic_buffer_resource stepArena; // data that persists for a training step (ie: forward pass intermediate outputs)
        std::pmr::unsynchronized_pool_resource scratchPool; // temporary data (forward pass scratch buffers and backward gradient buffers)

};
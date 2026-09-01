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

// Os-specify memory allocation
class VirtualMemoryResource : public std::pmr::memory_resource
{
    public:
        explicit VirtualMemoryResource(size_t maxBytes)
#ifndef _WIN32
            : maxBytes(maxBytes)
#endif
        {
    #if defined(_WIN32)
            ptr = VirtualAlloc(nullptr, maxBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    #else
            ptr = mmap(nullptr, maxBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

            if (ptr == MAP_FAILED)
                ptr = nullptr;
    #endif
            if (!ptr) throw std::bad_alloc();
        }

        ~VirtualMemoryResource() override {
    #if defined(_WIN32)
            VirtualFree(ptr, 0, MEM_RELEASE);
    #else
            munmap(ptr, maxBytes);
    #endif
        }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override
        {
            (void)alignment;
    #if defined(_WIN32)
            return VirtualAlloc(ptr, bytes, MEM_COMMIT, PAGE_READWRITE);
    #else
            return ptr_; // Linux handles on-demand physical commitment via page faults
    #endif
        }
        void do_deallocate(void*, size_t, size_t) override {}
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

    private:
        void* ptr = nullptr;
#ifndef _WIN32
        size_t maxBytes;
#endif
};

constexpr size_t SIMD_ALIGN = 32;

template <typename T>
class BatchBufferArenaT
{
	public:
        explicit BatchBufferArenaT(size_t maxCapacity) :
            osMem(maxCapacity),
            stepArena(&osMem),
            //scratchPool(&osMem)
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
            stepArena.release();
        }

	private:
        VirtualMemoryResource osMem;
        std::pmr::monotonic_buffer_resource stepArena; // data that persists for a training step (ie: forward pass intermediate outputs)
        std::pmr::unsynchronized_pool_resource scratchPool; // temporary data (forward pass scratch buffers and backward gradient buffers)

};
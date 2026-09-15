#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <cstdint>
#include <set>
#include <fenv.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/resource.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread.h>
#include <pthread/qos.h>
#endif

/**
 * @brief Utility class to manage hardware floating-point denormal modes.
 */
class DenormalManager
{
public:
    DenormalManager() = delete;

    /**
     * @brief Disables floating-point denormal numbers (sets FTZ and DAZ modes).
     */
    static inline void DisableDenormals()
    {
#if defined(__APPLE__)
#if defined(__arm64__) || defined(__aarch64__)
        // Apple Silicon ARM64
        fesetenv(FE_DFL_DISABLE_DENORMS_ENV);
#else
        // Apple Intel x86_64
        fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
#endif
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        // Windows & Linux (Intel/AMD) via standard SSE intrinsics
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#elif defined(__aarch64__) && defined(__linux__)
        // Linux ARM64
        unsigned long long fpcrValue;
        asm volatile("mrs %0, fpcr" : "=r"(fpcrValue));
        fpcrValue |= (1ULL << 24); // Set the FZ (Flush-to-zero) bit
        asm volatile("msr fpcr, %0" : : "r"(fpcrValue));
#elif defined(_MSC_VER) && defined(_M_ARM64)
        // Windows ARM64 (MSVC)
        unsigned int currentControl;
        _controlfp_s(&currentControl, _DN_FLUSH, _MCW_DN);
#endif
    }
};

class ThreadAffinityManager
{
public:
    ThreadAffinityManager() = delete;

    static std::vector<uint32_t> GetPhysicalCores() noexcept
    {
        uint32_t numCores = 0;
        std::vector<uint32_t> coreIDs;

#if defined(_WIN32)
        DWORD length = 0;
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) && GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
            std::vector<uint8_t> buffer(length);

            auto* engines = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());

            if (GetLogicalProcessorInformationEx(RelationProcessorCore, engines, &length))
            {
                DWORD offset = 0;

                while (offset < length)
                {
                    auto* current = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);

                    if (current->Relationship == RelationProcessorCore)
                    {
                        // FIX: Added [0] to index into the GroupMask array
                        ULONG_PTR mask = current->Processor.GroupMask[0].Mask;
                        unsigned long firstLogicalId = 0;

                        if (_BitScanForward(&firstLogicalId, mask))
                        {
                            coreIDs.push_back(firstLogicalId);
                        }
                    }

                    offset += current->Size;
                }
            }
        }
#elif defined(__linux__)
        long totalLogical = sysconf(_SC_NPROCESSORS_CONF);

        std::set<uint32_t> seenPhysicalCores;

        for (long i = 0; i < totalLogical; ++i)
        {
            std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id";
            std::ifstream file(path);
            uint32_t coreID = 0;

            if (file >> coreID)
            {
                // If this is the first time we encounter this physical core,
                // capture this logical ID for pinning.
                if (seenPhysicalCores.find(coreID) == seenPhysicalCores.end())
                {
                    seenPhysicalCores.insert(coreID);
                    coreIDs.push_back(static_cast<uint32_t>(i));
                }
            }
            else
            {
                coreIDs.push_back(static_cast<uint32_t>(i));
            }
        }
#elif defined(__APPLE__)
        int count = 0;
        size_t size = sizeof(count);

        if (sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nullptr, 0) == 0)
        {
            numCores = static_cast<uint32_t>(count);
        }

        if (sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0)
        {
            numCores = static_cast<uint32_t>(count);
        }
#endif
        if (coreIDs.size() == 0)
        {
            if (numCores == 0)
                numCores = std::thread::hardware_concurrency();

            if (numCores == 0)
                numCores = 1;

            for (uint32_t c = 0; c < numCores; c++)
            {
                coreIDs.push_back(c);
            }
        }

        return coreIDs;
    }

    static void SetHighPerformancePriority()
    {
#if defined(_WIN32)
        HANDLE threadHandle = GetCurrentThread();
        if (!SetThreadPriority(threadHandle, THREAD_PRIORITY_HIGHEST))
        {
            std::cerr << "Failed to set Windows thread priority. Error: " << GetLastError() << std::endl;
        }

#elif defined(__APPLE__)
        int qosResult = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        if (qosResult != 0) {
            std::cerr << "Failed to set macOS QoS class. Error code: " << qosResult << std::endl;
        }

#elif defined(__linux__)
        pthread_t nativeThread = pthread_self();

        struct sched_param schedParam;
        schedParam.sched_priority = sched_get_priority_max(SCHED_FIFO);

        int schedResult = pthread_setschedparam(nativeThread, SCHED_FIFO, &schedParam);

        if (schedResult != 0)
        {
            std::cerr << "Failed to set Linux real-time priority. Error code: " << schedResult << std::endl;

#include <sys/resource.h>
            if (setpriority(PRIO_PROCESS, 0, -20) != 0) {
                std::cerr << "Linux fallback to nice value failed as well." << std::endl;
            }
        }
#endif
    }

    static bool PinCurrentThread(uint32_t coreIndex) noexcept
    {
#if defined(_WIN32)
        HANDLE thread = GetCurrentThread();
        DWORD_PTR mask = static_cast<DWORD_PTR>(1) << coreIndex;
        DWORD_PTR result = SetThreadAffinityMask(thread, mask);
        return result != 0;
#elif defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(coreIndex, &cpuset);

        pthread_t thread = pthread_self();
        return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(__APPLE__)
        (void)coreIndex;
        return false;
#else
        (void)coreIndex;
        return false;
#endif
    }
};
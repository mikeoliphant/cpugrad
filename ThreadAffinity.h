#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <cstdint>

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
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

class ThreadAffinityManager {
public:
    ThreadAffinityManager() = delete;

    static uint32_t GetPhysicalCoreCount() noexcept {
#if defined(_WIN32)
        DWORD length = 0;
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) &&
            GetLastError() == ERROR_INSUFFICIENT_BUFFER) {

            std::vector<uint8_t> buffer(length);
            auto* engines = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());

            if (GetLogicalProcessorInformationEx(RelationProcessorCore, engines, &length)) {
                uint32_t physicalCores = 0;
                DWORD offset = 0;
                while (offset < length) {
                    auto* current = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
                    if (current->Relationship == RelationProcessorCore) {
                        ++physicalCores;
                    }
                    offset += current->Size;
                }
                return physicalCores > 0 ? physicalCores : std::thread::hardware_concurrency();
            }
        }
#elif defined(__linux__)
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (cores > 0) {
            return static_cast<uint32_t>(cores);
        }
#elif defined(__APPLE__)
        int count = 0;
        size_t size = sizeof(count);
        if (sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0) {
            return static_cast<uint32_t>(count);
        }
#endif
        uint32_t logicalCores = std::thread::hardware_concurrency();
        return logicalCores > 0 ? logicalCores : 1;
    }

    static bool PinCurrentThread(uint32_t coreIndex) noexcept {
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
#pragma once
#include <windows.h>
#include <psapi.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class SystemMonitor {
public:
    struct MemoryInfo {
        size_t totalPhysicalMemory;
        size_t availablePhysicalMemory;
        size_t usedPhysicalMemory;
        size_t processMemoryUsage;
        double memoryUsagePercent;
    };

    struct GPUInfo {
        bool available;
        size_t totalMemory;
        size_t freeMemory;
        size_t usedMemory;
        double memoryUsagePercent;
        double gpuUtilization;
    };

    static MemoryInfo getMemoryInfo() {
        MemoryInfo info = {0};
        
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            info.totalPhysicalMemory = memStatus.ullTotalPhys;
            info.availablePhysicalMemory = memStatus.ullAvailPhys;
            info.usedPhysicalMemory = memStatus.ullTotalPhys - memStatus.ullAvailPhys;
            info.memoryUsagePercent = static_cast<double>(info.usedPhysicalMemory) / info.totalPhysicalMemory * 100.0;
        }

        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            info.processMemoryUsage = pmc.WorkingSetSize;
        }

        return info;
    }

    static GPUInfo getGPUInfo() {
        GPUInfo info = {false, 0, 0, 0, 0.0, 0.0};
        
        #ifdef USE_CUDA
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        if (error == cudaSuccess && deviceCount > 0) {
            info.available = true;
            
            size_t free_mem, total_mem;
            error = cudaMemGetInfo(&free_mem, &total_mem);
            
            if (error == cudaSuccess) {
                info.totalMemory = total_mem;
                info.freeMemory = free_mem;
                info.usedMemory = total_mem - free_mem;
                info.memoryUsagePercent = static_cast<double>(info.usedMemory) / total_mem * 100.0;
            }
            
            cudaDeviceProp prop;
            error = cudaGetDeviceProperties(&prop, 0);
            if (error == cudaSuccess) {
            }
        }
        #endif
        
        return info;
    }

    static void printResourceUsage(const std::string& stageName) {
        std::cout << "\n--- " << stageName << " Resource Usage ---" << std::endl;
        
        MemoryInfo memInfo = getMemoryInfo();
        std::cout << "Memory Usage:" << std::endl;
        std::cout << "  Total Physical Memory: " << formatBytes(memInfo.totalPhysicalMemory) << std::endl;
        std::cout << "  Used Physical Memory:  " << formatBytes(memInfo.usedPhysicalMemory) 
                  << " (" << std::fixed << std::setprecision(1) << memInfo.memoryUsagePercent << "%)" << std::endl;
        std::cout << "  Process Memory Usage:  " << formatBytes(memInfo.processMemoryUsage) << std::endl;
        
        GPUInfo gpuInfo = getGPUInfo();
        if (gpuInfo.available) {
            std::cout << "GPU Memory Usage:" << std::endl;
            std::cout << "  Total GPU Memory: " << formatBytes(gpuInfo.totalMemory) << std::endl;
            std::cout << "  Used GPU Memory:  " << formatBytes(gpuInfo.usedMemory) 
                      << " (" << std::fixed << std::setprecision(1) << gpuInfo.memoryUsagePercent << "%)" << std::endl;
        } else {
            std::cout << "GPU: Not available or not using CUDA" << std::endl;
        }
        
        std::cout << "--------------------------------" << std::endl;
    }

    static void printTimingAndResources(const std::string& stageName, double elapsedSeconds) {
        std::cout << "\n======= " << stageName << " Complete =======" << std::endl;
        std::cout << stageName << " completed in " << std::fixed << std::setprecision(3) 
                  << elapsedSeconds << " seconds" << std::endl;
        printResourceUsage(stageName);
    }
    
    static std::string formatBytes(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unit < 4) {
            size /= 1024.0;
            unit++;
        }
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return ss.str();
    }
};
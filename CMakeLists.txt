cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(cpugrad VERSION 0.0.1)

set(BUILD_NAMCORE OFF CACHE BOOL "") 
set(BUILD_RTNEURAL OFF CACHE BOOL "") 

add_subdirectory(deps/NeuralAudio NeuralAudio)

# FFT backend for MR-STFT: AUDIOFFT (AudioFFT's Ooura backend, double precision) or PFFFT (float SIMD, ~3x faster FFTs).
# AUDIOFFT is the default: with PFFFT, 100-epoch A2 models came out measurably less accurate (mean ESR ~0.000184 vs
# ~0.000127-0.000151 over 3 seeds), apparently from float FFT error in quiet bins, which the log-magnitude loss weights most.
set(CPUGRAD_FFT "AUDIOFFT" CACHE STRING "FFT backend: AUDIOFFT (double precision, default) or PFFFT (faster, less accurate training)")
set_property(CACHE CPUGRAD_FFT PROPERTY STRINGS PFFFT AUDIOFFT)

if(CPUGRAD_FFT STREQUAL "PFFFT")
	set(FFT_SOURCES deps/pffft/pffft.c deps/pffft/pffft.h)
	set(FFT_INCLUDE_DIR deps/pffft)
elseif(CPUGRAD_FFT STREQUAL "AUDIOFFT")
	set(FFT_SOURCES deps/AudioFFT/AudioFFT.cpp)
	set(FFT_INCLUDE_DIR deps/AudioFFT)
else()
	message(FATAL_ERROR "CPUGRAD_FFT must be PFFFT or AUDIOFFT (got '${CPUGRAD_FFT}')")
endif()

message(STATUS "cpugrad FFT backend: ${CPUGRAD_FFT}")

set(SOURCES
	cpugrad/ModelTrainer.h
	cpugrad/BackpropModel.h
	cpugrad/Activation.h
	cpugrad/Dense.h
	cpugrad/Conv1D.h
	cpugrad/Loss.h
	cpugrad/MrStft.h
	cpugrad/Fft.h
	cpugrad/Dataset.h
	cpugrad/Optimizer.h
	cpugrad/BatchBuffer.h
	cpugrad/Platform.h
	${FFT_SOURCES})

add_library(cpugrad ${SOURCES})
target_include_directories(cpugrad PUBLIC deps/NeuralAudio/NeuralAudio)

# Fft.h is included (via MrStft.h) by users of cpugrad, so they need the backend's header and define too
target_include_directories(cpugrad PUBLIC ${FFT_INCLUDE_DIR})
target_compile_definitions(cpugrad PUBLIC CPUGRAD_FFT_${CPUGRAD_FFT})

target_link_libraries(cpugrad PUBLIC NeuralAudio)

source_group(cpugrad ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_subdirectory(cpugrad-test cpugrad-test)
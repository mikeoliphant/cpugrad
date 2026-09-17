cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(cpugrad VERSION 0.0.1)

set(BUILD_NAMCORE OFF CACHE BOOL "") 
set(BUILD_RTNEURAL OFF CACHE BOOL "") 

add_subdirectory(deps/NeuralAudio NeuralAudio)

set(SOURCES
	cpugrad/ModelTrainer.h
	cpugrad/BackpropModel.h
	cpugrad/Dense.h
	cpugrad/Conv1D.h
	cpugrad/Loss.h
	cpugrad/MrStft.h
	cpugrad/Dataset.h
	cpugrad/Optimizer.h
	cpugrad/BatchBuffer.h
	cpugrad/Platform.h
	deps/AudioFFT/AudioFFT.cpp)

add_library(cpugrad ${SOURCES})
target_include_directories(cpugrad PUBLIC deps/NeuralAudio/NeuralAudio)
target_include_directories(cpugrad PRIVATE deps/AudioFFT)

target_link_libraries(cpugrad PUBLIC NeuralAudio)

source_group(cpugrad ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_subdirectory(cpugrad-test cpugrad-test)
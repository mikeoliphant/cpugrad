cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(cpugrad VERSION 0.0.1)

set(BUILD_NAMCORE OFF CACHE BOOL "") 

add_subdirectory(deps/NeuralAudio NeuralAudio)

set(SOURCES
	ModelTrainer.h
	BackpropModel.h
	Dense.h
	Conv1D.h
	Dataset.h
	Optimizer.h
	BatchBuffer.h
	ThreadAffinity.h)

add_library(cpugrad ${SOURCES})
target_include_directories(cpugrad PUBLIC deps/NeuralAudio/NeuralAudio)
target_link_libraries(cpugrad PUBLIC NeuralAudio)

source_group(cpugrad ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_subdirectory(cpugrad-test cpugrad-test)
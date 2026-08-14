cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(NeuralCpuTrainer VERSION 0.0.1)

add_subdirectory(deps/NeuralAudio NeuralAudio)

set(SOURCES
	NeuralCpuTrainer.cpp
	ModelTrainer.h
	WaveNetBackprop.h
	NAM.h)

add_executable(NeuralCpuTrainer ${SOURCES})
target_include_directories(NeuralCpuTrainer PUBLIC deps/NeuralAudio/NeuralAudio)
target_include_directories(NeuralCpuTrainer PUBLIC deps/dr_libs)
target_link_libraries(NeuralCpuTrainer PUBLIC NeuralAudio)

source_group(NeuralCpuTrainer ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})


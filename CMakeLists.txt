cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(NeuralCpuTrainer VERSION 0.0.1)

if (MSVC)
	add_compile_options(
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	add_compile_options(
		-Wall 
		# -Wpedantic -Wextra -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb>"
		"$<$<CONFIG:RELWITHDEBINFO>:-O3>"
		"$<$<CONFIG:RELEASE>:-O3>"
	)
endif()



if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	add_compile_options(/arch:AVX2 /fp:fast)
	message(STATUS "Enabling /arch:AVX2")		 
else()
	add_compile_options(-march=x86-64-v3)
	message(STATUS "Enabling -march=x86-64-v3")
endif()

add_subdirectory(deps/NeuralAudio NeuralAudio)

set(SOURCES
	NeuralCpuTrainer.cpp
	ModelTrainer.h
	WaveNetBackprop.h
	Dataset.h
	Optimizer.h
	NAM.h)

add_executable(NeuralCpuTrainer ${SOURCES})
target_include_directories(NeuralCpuTrainer PUBLIC deps/NeuralAudio/NeuralAudio)
target_include_directories(NeuralCpuTrainer PUBLIC deps/dr_libs)
target_link_libraries(NeuralCpuTrainer PUBLIC NeuralAudio)

source_group(NeuralCpuTrainer ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})


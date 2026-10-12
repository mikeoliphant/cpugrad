add_executable(cpugrad-test Test.cpp Torch.h Tests.h)

find_package(Torch)

if(Torch_FOUND)
	message(STATUS "Found Torch")
	add_definitions(-DHAVE_TORCH)
	target_link_libraries(cpugrad-test PUBLIC "${TORCH_LIBRARIES}")

	if(MSVC)
	  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	  add_custom_command(TARGET cpugrad-test POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${TORCH_DLLS}
		$<TARGET_FILE_DIR:cpugrad-test>
	  )
endif()

else()
	message(STATUS "Torch not found")
endif()

target_link_libraries(cpugrad-test PUBLIC cpugrad NeuralAudio)
target_include_directories(cpugrad-test PUBLIC ../cpugrad)
target_include_directories(cpugrad-test PUBLIC ../deps/NeuralAudio/NeuralAudio)
target_include_directories(cpugrad-test PUBLIC ../deps/AudioFFT)
target_include_directories(cpugrad-test SYSTEM PRIVATE ../deps/NeuralAudio/deps/RTNeural/modules/Eigen)

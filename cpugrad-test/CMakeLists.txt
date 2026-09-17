add_executable(cpugrad-test Test.cpp Tests.h)

target_link_libraries(cpugrad-test PUBLIC cpugrad NeuralAudio)
target_include_directories(cpugrad-test PUBLIC ../cpugrad)
target_include_directories(cpugrad-test PUBLIC ../deps/NeuralAudio/NeuralAudio)
target_include_directories(cpugrad-test PUBLIC ../deps/AudioFFT)
target_include_directories(cpugrad-test SYSTEM PRIVATE ../deps/NeuralAudio/deps/RTNeural/modules/Eigen)

add_executable(cpugrad-test Test.cpp Tests.h)

target_link_libraries(cpugrad-test PUBLIC NeuralAudio)
target_include_directories(cpugrad-test  PUBLIC ..)
target_include_directories(cpugrad-test PUBLIC ../deps/NeuralAudio/NeuralAudio)
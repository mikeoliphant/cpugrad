#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <xmmintrin.h>

#include "Dense.h"
#include "Conv1D.h"
#include "ModelTrainer.h"
#include "Dataset.h"
#include "Tests.h"

using namespace cpugrad;

int main()
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	DataGen dataGen(123);

	size_t numSamples = 48000 * 180;

	auto randData = dataGen.GenerateRandom(numSamples);
	auto sinData = dataGen.GenerateSin(numSamples, 8192);
	auto delayData = dataGen.GenerateDelay(256, numSamples);
	auto xorData = dataGen.GenerateXOR(1, numSamples);


	//auto denseTrainer = new ModelTrainerT<float, DenseBackpropT<float, 1, 1, false>>();

	//denseTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

	//denseTrainer->TrainIdentity(randData);


	//denseTrainter->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//auto convTrainer = new ModelTrainerT<float, Conv1DBackpropT<float, 1, 1, 2, true, 128>>();

	//convTrainer->Train(delayData);

	////convTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	//auto twoConvTrainer = new ModelTrainerT<float, TwoConvTestT<float, 1, 2, 128>>();

	//twoConvTrainer->Train(delayData);

	auto convTestTrainer = new ModelTrainerT<float, ConvTestT<float, 1, 16, 3, 1, 1>>();

	convTestTrainer->TrainIdentity(randData);

	//convTestTrainer->Train(xorData);

	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");
	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\BossSD1.wav)");

	return 0;
}


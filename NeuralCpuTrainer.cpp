#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "NeuralModel.h"
#define DR_WAV_IMPLEMENTATION
#include "WaveNet.h"
#include "WaveNetBackprop.h"
#include "ModelTrainer.h"
#include "NAM.h"

using namespace NeuralAudio;
using namespace NeuralCpuTrain;

static void TestNAM(std::filesystem::path modelPath)
{
	NeuralAudio::NeuralModelLoader loader;

	auto namModel = loader.CreateFromFile(modelPath);

	size_t numSamples = 24000; //48000 * 3;

	auto a2 = new A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>();

	auto modelTrainer = new ModelTrainerT<float, BATCH_SIZE>(*a2);

	auto input = modelTrainer->GenerateSin(numSamples);

	std::vector<float> namOutput(numSamples);

	namModel->Process(input.data(), namOutput.data(), numSamples);

	std::ifstream jsonStream(modelPath, std::ifstream::binary);

	nlohmann::json modelJson;
	jsonStream >> modelJson;

	std::vector<float> weights = modelJson.at("weights");

	auto it = weights.begin();

	a2->SetWeights(it);
	a2->SetHeadScale(*it);

	double err = modelTrainer->VerifyModel(input.data(), namOutput.data(), numSamples);

	std::cout << "Err: " << err << std::endl;
}

int main()
{
	//DenseBackpropT<float, 1, 1, false> dense;
	//auto denseTrainter = new ModelTrainerT<float, BATCH_SIZE>(dense);

	//denseTrainter->TestIdentity();

	//Conv1DBackpropT<float, 1, 1, 3, true, 1> convBackprop;
	//auto convTrainer = new ModelTrainerT<float, BATCH_SIZE>(convBackprop);

	//convTrainer->TestDelay(2);

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	//TestNAM(R"(C:\Code\NeuralCpuTrainer\BossWN-a2lite.nam)");

	auto a2 = new A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>();

	auto modelTrainer = new ModelTrainerT<float, BATCH_SIZE>(*a2);

	modelTrainer->TestDelay(0);

	modelTrainer->TestWav(R"(C:\Share\Recordings\NAM\TZ3-sweep-v3.wav)", R"(C:\Share\Recordings\NAM\BossSD1CaptureNeuralAudio.wav)");

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}


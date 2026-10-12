#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "Dense.h"
#include "Conv1D.h"
#include "ModelTrainer.h"
#include "Dataset.h"
#include "Tests.h"

#ifdef HAVE_TORCH
#include "Torch.h"
#endif

using namespace cpugrad;

#ifdef HAVE_TORCH
void SpecTestTorch()
{
	const int64_t num_samples = 8192;
	const double sample_rate = 16000.0;

	// Calculate step size (1 / 16000) and the exclusive upper boundary
	double step_size = 1.0 / sample_rate;
	double end_time = (double)num_samples / sample_rate; // 8192 / 16000 = 0.512

	// This generates exactly 8192 points from 0 to 0.5119375
	torch::Tensor t = torch::arange(0.0, end_time, step_size,
		torch::TensorOptions().dtype(torch::kFloat64));

	// 2. Generate target signal: A clean 440Hz tone with an exponential decay * 0
	auto decay = torch::exp(-3.0 * t);
	auto target_signal = torch::sin(2 * M_PI * 440.0 * t) * decay;

	// Generate output signal: Fundamental tone (442Hz) + a harsh 3000Hz noise component
	auto noise = 0.25 * torch::sin(2 * M_PI * 3000.0 * t);
	auto output_signal = (torch::sin(2 * M_PI * 442.0 * t) * decay) + noise;


	//std::vector<StftWindowConfig> configs({ { 2048, 240, 1200 }, { 1024, 120, 600 }, { 512, 50, 240 } });
	std::vector<StftWindowConfig> configs({ { 2048, 240, 1200 } });

	float specWeight = 0.0f;
	float logMagWeight = 1.0f;
	float linMagWeight = 0.0f;

	MultiResolutionSTFTLoss torchLoss(configs);

	torchLoss.SetWeights(specWeight, logMagWeight, linMagWeight);

	double loss = torchLoss.GetMeanLoss(output_signal, target_signal);

	std::cout << "torch loss: " << loss << std::endl;

	auto torchGradient = torchLoss.GetLossGradient(output_signal, target_signal);

	//for (float val : vec)
	//{
	//	std::cout << val << std::endl;
	//}

	MultiResolutionStftLossT<float> cpugradLoss(configs);

	cpugradLoss.SetWeights(specWeight, logMagWeight);

	auto outputSignalVec = TensorToVector(output_signal);
	auto targetSignalVec = TensorToVector(target_signal);

	loss = cpugradLoss.GetMeanLoss(outputSignalVec.data(), targetSignalVec.data(), outputSignalVec.size());

	std::cout << "cpugrad loss: " << loss << std::endl;

	std::vector<float> cpugradGradient(outputSignalVec.size());

	cpugradLoss.ComputeLoss(outputSignalVec.data(), targetSignalVec.data(), cpugradGradient.data(), outputSignalVec.size(), static_cast<double>(num_samples));

	MSELossT<float> mseLoss;

	double mse = mseLoss.GetMeanLoss(torchGradient.data(), cpugradGradient.data(), torchGradient.size());

	std::cout << "Gradient MSE: " << mse << std::endl;
}

#endif

int main()
{
	SpecTestTorch();

	std::mt19937 rand(123);

	DataGen dataGen(rand);

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


#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "WaveNet.h"
#include "WaveNetBackprop.h"

using namespace NeuralAudio;
using namespace NeuralCpuTrain;

int main()
{
	const size_t totalSamples = 48000;
	const size_t blockSize = 64; // model processes 64 samples at a time
	const size_t numBlocks = totalSamples / blockSize;

	// Create long input (sine) and target (identity for now)
	std::vector<float> input(totalSamples);
	std::vector<float> target(totalSamples);

	for (size_t i = 0; i < totalSamples; ++i)
	{
		input[i] = std::sin(static_cast<float>(i) * 0.01f);
		target[i] = input[i]; // identity target for now
	}

	DenseLayerT<float, 1, 1, false> model;
	DenseBackpropT<float, 1, 1, false> modelBackprop(model);

	// Initialize weights
	size_t nWeights = model.GetNumWeights();
	std::vector<float> weights(nWeights);
	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& w : weights) w = dist(rng);

	auto it = weights.begin();

	model.SetWeights(it);

	for (int iter = 0; iter < 200; ++iter)
	{
		ChannelBuffer<float, 1, blockSize> blockIn;
		ChannelBuffer<float, 1, blockSize> blockOut;
		ChannelBuffer<float, 1, blockSize> blockTarget;
		ChannelBuffer<float, 1, blockSize> blockGradient;
		ChannelBuffer<float, 1, blockSize> blockOutGradient;

		DenseBackpropWeightsT<float, 1, 1, false> weightAccumulator;

		weightAccumulator.SetZero();

		for (size_t b = 0; b < numBlocks; ++b)
		{
			std::memcpy(blockIn.GetData(), input.data() + b * blockSize, blockSize * sizeof(float));
			std::memcpy(blockTarget.GetData(), target.data() + b * blockSize, blockSize * sizeof(float));

			model.Process(blockIn, blockOut);

			auto map = blockGradient.GetEigenMap();

			map = 2.0f * (blockOut.GetEigenMapConst().array() - blockTarget.GetEigenMapConst().array());

			modelBackprop.Backward(blockIn, blockGradient, blockOutGradient, weightAccumulator);
		}

		weightAccumulator.Divide(static_cast<float>(numBlocks * blockSize));

		weightAccumulator.ApplyGradients(model);

		double totErr = 0.0;

		for (size_t b = 0; b < numBlocks; ++b)
		{
			std::memcpy(blockIn.GetData(), input.data() + b * blockSize, blockSize * sizeof(float));
			std::memcpy(blockTarget.GetData(), target.data() + b * blockSize, blockSize * sizeof(float));

			model.Process(blockIn, blockOut);

			for (int i = 0; i < blockSize; i++)
			{
				double diff = blockOut(0, i) - blockIn(0, i);

				totErr += diff * diff;
			}
		}

		float blah = model.GetWeights()(0, 0);

		std::cout << "Iter: " << iter << " MSE: " << (totErr / static_cast<float>(numBlocks * blockSize)) << std::endl;
	}

	return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "WaveNet.h"
#include "WaveNetBackprop.h"

#define MAX_SAMPLES 8192

using namespace NeuralAudio;
using namespace NeuralCpuTrain;

static void TestModel(BackpropModelT<float, 1, 1>& modelBackprop)
{
	const size_t totalSamples = MAX_SAMPLES;

	// Create long input (sine) and target (identity for now)
	ChannelBuffer<float, 1, totalSamples> input;
	ChannelBuffer<float, 1, totalSamples> target;

	for (size_t i = 0; i < totalSamples; ++i)
	{
		input(0, i) = std::sin(static_cast<float>(i) * 0.01f);
		target(0, i) = input(0, i); // identity target for now
	}

	// Initialize weights
	//size_t nWeights = modelBackprop.GetNumWeights();
	//std::vector<float> weights(nWeights);
	//std::mt19937 rng(123);
	//std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
	//for (auto& w : weights) w = dist(rng);

	//auto it = weights.begin();

	//modelBackprop.SetWeights(it);

	modelBackprop.RandomizeWeights();

	size_t receptiveField = modelBackprop.GetReceptiveField();
	size_t validSampleCount = totalSamples - receptiveField;

	ChannelBuffer<float, 1, totalSamples> forwardOutput;
	ChannelBuffer<float, 1, totalSamples> outputGradient;
	ChannelBuffer<float, 1, totalSamples> layerOutputGradient;

	for (int iter = 0; iter < 2000; ++iter)
	{
		forwardOutput.SetZero();

		modelBackprop.Reset();

		modelBackprop.Forward(input, forwardOutput);

		auto map = outputGradient.GetEigenMap();

		double totErr = 0.0;

		for (size_t i = 0; i < totalSamples; i++)
		{
			if (i < receptiveField)
			{
				map(i) = 0.0f;
			}
			else
			{
				float diff = forwardOutput(0, i) - input(0, i);

				map(i) = 2.0f * diff;

				totErr += diff * diff;
			}
		}

		std::cout << "Iter: " << iter << " MSE: " << (totErr / static_cast<float>(validSampleCount)) << std::endl;
		
		modelBackprop.Backward(input, outputGradient, layerOutputGradient);

		modelBackprop.DivideWeights(static_cast<float>((float)validSampleCount));

		modelBackprop.ApplyGradients(0.05f);
	}
}

template <typename T, int ConditionSize, int Channels, int KernelSize, int Dilation>
class WaveNetLayerBackpropT : public BackpropModelT<T, Channels, Channels>
{
public:
	void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& output, const ChannelRowSpan<T, Channels>& headOutput)
	{
		conv.Forward(input, convOut);

		conditionMixIn.Forward(condition, conditionMixInOut);

		auto convOutMap = convOut.GetEigenMap();
		convOutMap.noalias() += conditionMixInOut.GetEigenMapConst();

		relu.Forward(convOut, reluOut);

		auto headOutputMap = headOutput.GetEigenMap();
		headOutputMap.noalias() += reluOut.GetEigenMapConst();

		oneByOne.Forward(reluOut, output);
	}

	void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		oneByOne.Backward(reluOut, dOutput, dOneByOneOut);

		auto dOneByOneOutMap = dOneByOneOut.GetEigenMap();
		dOneByOneOutMap.noalias() += dHeadOutput.GetEigenMapConst();

		relu.Backward(convOut, dOneByOneOut, dReluOut);
		conditionMixIn.Backward(condition, dReluOut, dInput);	// dInput just used for temp storage, will be overwritten
		conv.Backward(input, dReluOut, dInput);
	}

	size_t GetReceptiveField() override
	{
		return conv.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		return conv.GetNumWeights() + oneByOne.GetNumWeights() + conditionMixIn.GetNumWeights();;
	}

	void RandomizeWeights() override
	{
		conv.RandomizeWeights();
		oneByOne.RandomizeWeights();
		conditionMixIn.RandomizeWeights();
	}

	void SetWeights(std::vector<float>::iterator& inWeights) override
	{
		conv.SetWeights(inWeights);
		conditionMixIn.SetWeights(inWeights);
		oneByOne.SetWeights(inWeights);
	}

	void Reset() override
	{
		conv.Reset();
		convOut.SetZero();
		conditionMixIn.Reset();
		conditionMixInOut.SetZero();
		reluOut.SetZero();
		oneByOne.Reset();
		dOneByOneOut.SetZero();
		dReluOut.SetZero();
	}

	void DivideWeights(float divisor) override
	{
		conv.DivideWeights(divisor);
		oneByOne.DivideWeights(divisor);
	}

	void ApplyGradients(float scale) override
	{
		conv.ApplyGradients(scale);
		oneByOne.ApplyGradients(scale);
	}

private:
	Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
	ChannelBuffer<T, Channels, MAX_SAMPLES> convOut;
	DenseBackpropT<T, ConditionSize, Channels, false> conditionMixIn;
	ChannelBuffer<T, Channels, MAX_SAMPLES> conditionMixInOut;
	LeakyReLUT<T, Channels> relu;
	ChannelBuffer<T, Channels, MAX_SAMPLES> reluOut;
	DenseBackpropT<T, Channels, Channels, true> oneByOne;
	ChannelBuffer<T, Channels, MAX_SAMPLES> dOneByOneOut;
	ChannelBuffer<T, Channels, MAX_SAMPLES> dReluOut;
};


template <typename T, int Channels>
class A2BackpropT : public BackpropModelT<T, Channels, Channels>
{
	public:
		void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& output) override
		{
			headOutput.SetZero();

			layer1.Forward(input, input, layer1Out, headOutput);
			layer2.Forward(layer1Out, input, output, headOutput);

			auto outputMap = output.GetEigenMap();
			outputMap.noalias() += headOutput.GetEigenMapConst();
		}

		void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput) override
		{
			layer2.Backward(layer1Out, input, dOutput, dOutput, dLayer1Out);
			layer1.Backward(input, input, dLayer1Out, dOutput, dInput);
		}

		size_t GetReceptiveField() override
		{
			return layer1.GetReceptiveField() + layer2.GetReceptiveField();
		}

		size_t GetNumWeights() override
		{
			return layer1.GetNumWeights() + layer2.GetNumWeights();
		}

		void SetWeights(std::vector<float>::iterator& inWeights) override
		{
			layer1.SetWeights(inWeights);
			layer2.SetWeights(inWeights);
		}

		void Reset() override
		{
			layer1.Reset();
			layer1Out.SetZero();
			dLayer1Out.SetZero();
			layer2.Reset();
		}

		void DivideWeights(float divisor) override
		{
			layer1.DivideWeights(divisor);
			layer2.DivideWeights(divisor);
		}

		void ApplyGradients(float scale) override
		{
			layer1.ApplyGradients(scale);
			layer2.ApplyGradients(scale);
		}

	private:
		WaveNetLayerBackpropT<T, 1, Channels, 3, 1> layer1;
		ChannelBuffer<T, Channels, MAX_SAMPLES> layer1Out;
		WaveNetLayerBackpropT<T, 1, Channels, 3, 2> layer2;
		ChannelBuffer<T, Channels, MAX_SAMPLES> dLayer1Out;
		ChannelBuffer<T, Channels, MAX_SAMPLES> headOutput;
};

int main()
{
	//DenseBackpropT<float, 1, 1, false> modelBackprop;
	//TestModel(modelBackprop);

	//Conv1DBackpropT<float, 1, 1, 3, true, 1> convBackprop;
	//TestModel(convBackprop);

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	A2BackpropT<float, 1> a2;
	TestModel(a2);

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}


#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

#include "dr_wav.h"
#include "WaveNet.h"
#include "WaveNetBackprop.h"

#define BATCH_SIZE 8192

using namespace NeuralAudio;
using namespace NeuralCpuTrain;

static void TestModel(BackpropModelT<float, 1, 1>& modelBackprop, const ChannelRowSpan<float, 1>& input, const ChannelRowSpan<float, 1>& target)
{
	assert(input.GetNumCols() == target.GetNumCols());

	size_t totalSize = input.GetNumCols();

	const size_t batchSize = BATCH_SIZE;

	modelBackprop.RandomizeWeights();

	size_t receptiveField = modelBackprop.GetReceptiveField();
	size_t validSampleCount = batchSize - receptiveField;

	ChannelBuffer<float, 1, batchSize> batchInput;
	ChannelBuffer<float, 1, batchSize> batchTarget;
	ChannelBuffer<float, 1, batchSize> forwardOutput;
	ChannelBuffer<float, 1, batchSize> outputGradient;
	ChannelBuffer<float, 1, batchSize> layerOutputGradient;

	for (int iter = 0; iter < 20000; ++iter)
	{
		size_t currentOffset = 0;
		int samplesRemaining = (int)totalSize;

		while (samplesRemaining > 0)
		{
			size_t thisBatchSize = std::min((size_t)samplesRemaining, batchSize);

			auto batchInMap = batchInput.Slice(thisBatchSize).GetEigenMap();
			batchInMap.noalias() = input.Slice(currentOffset, thisBatchSize).GetEigenMapConst();

			auto batchTargetMap = batchTarget.Slice(thisBatchSize).GetEigenMap();
			batchTargetMap.noalias() = target.Slice(currentOffset, thisBatchSize).GetEigenMapConst();

			forwardOutput.SetZero();

			modelBackprop.Reset();

			modelBackprop.Forward(batchInput, forwardOutput);

			auto map = outputGradient.GetEigenMap();

			double totErr = 0.0;

			for (size_t i = 0; i < thisBatchSize; i++)
			{
				if (i < receptiveField)
				{
					map(i) = 0.0f;
				}
				else
				{
					float diff = forwardOutput(0, i) - batchTarget(0, i);

					map(i) = 2.0f * diff;

					totErr += diff * diff;
				}
			}

			std::cout << "Iter: " << iter << " MSE: " << (totErr / static_cast<float>(validSampleCount)) << std::endl;

			modelBackprop.Backward(batchInput, outputGradient, layerOutputGradient);

			modelBackprop.DivideWeights(static_cast<float>((float)validSampleCount));

			modelBackprop.ApplyGradients(0.05f);

			currentOffset += validSampleCount;
			samplesRemaining -= (int)validSampleCount;
		}
	}
}

static void TestIdentity(BackpropModelT<float, 1, 1>& modelBackprop)
{
	const size_t totalSamples = 48000;

	ChannelBuffer<float, 1, totalSamples> input;
	ChannelBuffer<float, 1, totalSamples> target;

	for (size_t i = 0; i < totalSamples; ++i)
	{
		input(0, i) = std::sin(static_cast<float>(i) * 0.01f);
		target(0, i) = input(0, i);
	}

	TestModel(modelBackprop, input, target);
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

		conditionMixIn.Backward(condition, dReluOut, dConditionMixInOut);	// dConditionMixInOut not used - can optimize

		conv.Backward(input, dReluOut, dInput);
	}

	void BackwardNoLayerOutput(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		//oneByOne.Backward(reluOut, dOutput, dOneByOneOut);

		//auto dOneByOneOutMap = dOneByOneOut.GetEigenMap();
		//dOneByOneOutMap.noalias() += dHeadOutput.GetEigenMapConst();

		relu.Backward(convOut, dHeadOutput, dReluOut);
		conditionMixIn.Backward(condition, dReluOut, dConditionMixInOut);	// dConditionMixInOut not used - can optimize
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
		conditionMixIn.DivideWeights(divisor);
		oneByOne.DivideWeights(divisor);
	}

	void ApplyGradients(float scale) override
	{
		conv.ApplyGradients(scale);
		conditionMixIn.ApplyGradients(scale);
		oneByOne.ApplyGradients(scale);
	}

private:
	Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
	ChannelBuffer<T, Channels, BATCH_SIZE> convOut;
	DenseBackpropT<T, ConditionSize, Channels, false> conditionMixIn;
	ChannelBuffer<T, Channels, BATCH_SIZE> conditionMixInOut;
	ChannelBuffer<T, ConditionSize, BATCH_SIZE> dConditionMixInOut;
	LeakyReLUT<T, Channels> relu;
	ChannelBuffer<T, Channels, BATCH_SIZE> reluOut;
	DenseBackpropT<T, Channels, Channels, true> oneByOne;
	ChannelBuffer<T, Channels, BATCH_SIZE> dOneByOneOut;
	ChannelBuffer<T, Channels, BATCH_SIZE> dReluOut;
};


template <typename T, int InOutChannels, int Channels>
class A2BackpropT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
	public:
		void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
		{
			headOutput.SetZero();

			layerArrayRechannel.Forward(input, layerArrayRechannelOut);

			layer1.Forward(layerArrayRechannelOut, input, layer1Out, headOutput);
			layer2.Forward(layer1Out, input, layer2Out, headOutput);

			headRechannel.Forward(headOutput, output);

			auto outputMap = output.GetEigenMap();
			outputMap *= 0.1f;	// head scale
		}

		void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
		{
			auto dOutputMap = dOutput.GetEigenMap();
			dOutputMap *= 0.1f; // head scale

			headRechannel.Backward(headOutput, dOutput, dHeadRechannelOut);

			layer2.BackwardNoLayerOutput(layer1Out, input, dHeadRechannelOut, dLayer2Out);
			layer1.Backward(layerArrayRechannelOut, input, dLayer2Out, dHeadRechannelOut, dLayer1Out);

			layerArrayRechannel.Backward(input, dLayer1Out, dInput);
		}

		size_t GetReceptiveField() override
		{
			return layer1.GetReceptiveField() + layer2.GetReceptiveField() + headRechannel.GetReceptiveField();
		}

		size_t GetNumWeights() override
		{
			return layer1.GetNumWeights() + layer2.GetNumWeights() + layerArrayRechannel.GetNumWeights() + headRechannel.GetNumWeights();
		}

		void SetWeights(std::vector<float>::iterator& inWeights) override
		{
			layerArrayRechannel.SetWeights(inWeights);

			layer1.SetWeights(inWeights);
			layer2.SetWeights(inWeights);

			headRechannel.SetWeights(inWeights);
		}

		void Reset() override
		{
			layerArrayRechannel.Reset();
			layerArrayRechannelOut.SetZero();

			layer1.Reset();
			layer1Out.SetZero();
			dLayer1Out.SetZero();

			layer2.Reset();
			layer2Out.SetZero();
			dLayer2Out.SetZero();

			headRechannel.Reset();
			dHeadRechannelOut.SetZero();
		}

		void RandomizeWeights() override
		{
			layerArrayRechannel.RandomizeWeights();
			layer1.RandomizeWeights();
			layer2.RandomizeWeights();
			headRechannel.RandomizeWeights();
		}

		void DivideWeights(float divisor) override
		{
			layerArrayRechannel.DivideWeights(divisor);

			layer1.DivideWeights(divisor);
			layer2.DivideWeights(divisor);

			headRechannel.DivideWeights(divisor);
		}

		void ApplyGradients(float scale) override
		{
			layerArrayRechannel.ApplyGradients(scale);

			layer1.ApplyGradients(scale);
			layer2.ApplyGradients(scale);

			headRechannel.ApplyGradients(scale);
		}

	private:
		DenseBackpropT<T, InOutChannels, Channels, false> layerArrayRechannel;
		ChannelBuffer<T, Channels, BATCH_SIZE> layerArrayRechannelOut;
		ChannelBuffer<T, Channels, BATCH_SIZE> dLayer1Out;
		WaveNetLayerBackpropT<T, InOutChannels, Channels, 3, 1> layer1;
		ChannelBuffer<T, Channels, BATCH_SIZE> layer1Out;
		WaveNetLayerBackpropT<T, InOutChannels, Channels, 3, 2> layer2;
		ChannelBuffer<T, Channels, BATCH_SIZE> layer2Out;
		ChannelBuffer<T, Channels, BATCH_SIZE> dLayer2Out;
		ChannelBuffer<T, Channels, BATCH_SIZE> headOutput;
		Conv1DBackpropT<T, Channels, InOutChannels, 16, true, 1> headRechannel;
		ChannelBuffer<T, Channels, BATCH_SIZE> dHeadRechannelOut;
};

int main()
{
	//DenseBackpropT<float, 1, 1, false> modelBackprop;
	//TestModel(modelBackprop);

	//Conv1DBackpropT<float, 1, 1, 3, true, 1> convBackprop;
	//TestModel(convBackprop);

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	auto a2 = new A2BackpropT<float, 1, 3>();
	TestIdentity(*a2);

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}


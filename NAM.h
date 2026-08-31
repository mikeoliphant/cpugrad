#pragma once

#include "ModelTrainer.h"

using namespace NeuralCpuTrain;

template <typename T, int ConditionSize, int Channels, int KernelSize, int Dilation>
class WaveNetLayerBackpropT : public BackpropModelT<T, Channels, Channels>
{
public:
	using BackpropModelT<T, Channels, Channels>::Forward;
	using BackpropModelT<T, Channels, Channels>::Backward;
	using BackpropModelT<T, Channels, Channels>::trainingContext;

	void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& output, const ChannelRowSpan<T, Channels>& headOutput)
	{
		size_t numSamples = input.GetNumCols();

		conv.Reset();
		convOut.SetZero();
		conv.Forward(input, convOut.Slice(numSamples));

		conditionMixIn.Reset();
		//conditionMixInOut.SetZero();
		conditionMixIn.Forward(condition, conditionMixInOut.Slice(numSamples));

		auto convOutMap = convOut.Slice(numSamples).GetEigenMap();
		convOutMap.noalias() += conditionMixInOut.Slice(numSamples).GetEigenMapConst();

		relu.Reset();
		//reluOut.SetZero();
		relu.Forward(convOut.Slice(numSamples), reluOut.Slice(numSamples));

		auto headOutputMap = headOutput.GetEigenMap();
		headOutputMap.noalias() += reluOut.Slice(numSamples).GetEigenMapConst();

		oneByOne.Reset();
		//output.SetZero();
		oneByOne.Forward(reluOut.Slice(numSamples), output);	// Not needed on last layer - can optimize

		auto outputMap = output.GetEigenMap();
		outputMap.noalias() += input.GetEigenMapConst();
	}

	void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		size_t numSamples = input.GetNumCols();

		oneByOne.Backward(reluOut.Slice(numSamples), dOutput, dOneByOneOut.Slice(numSamples));

		auto dOneByOneOutMap = dOneByOneOut.Slice(numSamples).GetEigenMap();
		dOneByOneOutMap.noalias() += dHeadOutput.GetEigenMapConst();

		relu.Backward(convOut.Slice(numSamples), dOneByOneOut.Slice(numSamples), dReluOut.Slice(numSamples));

		conditionMixIn.BackwardNoDInput(condition, dReluOut.Slice(numSamples));

		conv.Backward(input, dReluOut.Slice(numSamples), dInput);

		auto dInputMap = dInput.GetEigenMap();
		dInputMap.noalias() += dOutput.GetEigenMapConst();
	}

	void BackwardNoLayerOutput(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		size_t numSamples = input.GetNumCols();

		relu.Backward(convOut.Slice(numSamples), dHeadOutput, dReluOut.Slice(numSamples));
		conditionMixIn.BackwardNoDInput(condition, dReluOut.Slice(numSamples));
		conv.Backward(input, dReluOut.Slice(numSamples), dInput);
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
		// not clearing the gradient buffer works, but I'm not sure why...

		//dConditionMixInOut.SetZero();
		//dReluOut.SetZero();
		//dOneByOneOut.SetZero();
	}

	void SetTrainingContext(TrainingContextT<T>* context) override
	{
		conv.SetTrainingContext(context);
		conditionMixIn.SetTrainingContext(context);
		oneByOne.SetTrainingContext(context);
	}

private:
	Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> convOut;
	DenseBackpropT<T, ConditionSize, Channels, false> conditionMixIn;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> conditionMixInOut;
	LeakyReLUT<T, Channels> relu;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> reluOut;
	DenseBackpropT<T, Channels, Channels, true> oneByOne;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dOneByOneOut;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dReluOut;
};

template <typename T, int InOutChannels, int Channels, typename KernelSizeSequence, typename DilationsSequence>
class A2BackpropT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
	using BackpropModelT<T, InOutChannels, InOutChannels>::trainingContext;

	template <typename, typename>
	struct LayersHelper
	{};

	template <int... dilationVals, int... kernelSizeVals>
	struct LayersHelper<KernelSizes<kernelSizeVals...>, Dilations<dilationVals...>>
	{
		using type = std::tuple<WaveNetLayerBackpropT<T, InOutChannels, Channels, kernelSizeVals, dilationVals>...>;
	};

	using Layers = typename LayersHelper<KernelSizeSequence, DilationsSequence>::type;

public:
	Layers layers;
	static constexpr auto NumLayers = std::tuple_size_v<decltype (layers)>;

	void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
	{
		size_t numSamples = input.GetNumCols();

		headOutput.SetZero();

		layerArrayRechannel.Reset();
		//layerArrayRechannelOut.SetZero();
		layerArrayRechannel.Forward(input, layerArrayRechannelOut.Slice(numSamples));

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				layerOuts[layerIndex].SetZero();

				if constexpr (layerIndex == 0)
				{
					std::get<layerIndex>(layers).Forward(layerArrayRechannelOut.Slice(numSamples), input, layerOuts[layerIndex].Slice(numSamples), headOutput.Slice(numSamples));
				}
				else
				{
					std::get<layerIndex>(layers).Forward(layerOuts[layerIndex - 1].Slice(numSamples), input, layerOuts[layerIndex].Slice(numSamples), headOutput.Slice(numSamples));
				}
			});

		//headRechannel.Reset();
		headRechannel.Forward(headOutput.Slice(numSamples), output);

		auto outputMap = output.GetEigenMap();
		outputMap *= headScale;
	}

	void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
	{
		size_t numSamples = input.GetNumCols();

		auto dOutputMap = dOutput.GetEigenMap();
		dOutputMap *= headScale;

		dHeadRechannelOut.SetZero();
		headRechannel.Backward(headOutput.Slice(numSamples), dOutput, dHeadRechannelOut.Slice(numSamples));

		ForEachIndex<NumLayers>([&](auto layerIndexForward)
			{
				constexpr auto layerIndexBackward = NumLayers - 1 - layerIndexForward;

				dLayerOuts[layerIndexForward].SetZero();

				if constexpr (layerIndexForward == 0)
				{
					std::get<layerIndexBackward>(layers).BackwardNoLayerOutput(layerOuts[layerIndexBackward - 1].Slice(numSamples), input, dHeadRechannelOut.Slice(numSamples), dLayerOuts[layerIndexBackward].Slice(numSamples));
				}
				else if constexpr (layerIndexBackward > 0)
				{
					std::get<layerIndexBackward>(layers).Backward(layerOuts[layerIndexBackward - 1].Slice(numSamples), input, dLayerOuts[layerIndexBackward + 1].Slice(numSamples), dHeadRechannelOut.Slice(numSamples), dLayerOuts[layerIndexBackward].Slice(numSamples));
				}
				else  // Last layer
				{
					std::get<layerIndexBackward>(layers).Backward(layerArrayRechannelOut.Slice(numSamples), input, dLayerOuts[layerIndexBackward + 1].Slice(numSamples), dHeadRechannelOut.Slice(numSamples), dLayerOuts[layerIndexBackward].Slice(numSamples));
				}
			});

		layerArrayRechannel.Backward(input, dLayerOuts[0].Slice(numSamples), dInput);
	}

	size_t GetReceptiveField() override
	{
		size_t fieldSize = 0;

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				fieldSize += std::get<layerIndex>(layers).GetReceptiveField();
			});

		return fieldSize + headRechannel.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		size_t numWeights = 0;

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				numWeights += std::get<layerIndex>(layers).GetNumWeights();
			});

		return numWeights + headRechannel.GetNumWeights() + layerArrayRechannel.GetNumWeights();
	}

	void SetWeights(std::vector<float>::iterator& inWeights) override
	{
		layerArrayRechannel.SetWeights(inWeights);

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).SetWeights(inWeights);
			});

		headRechannel.SetWeights(inWeights);
	}

	void SetHeadScale(float scale)
	{
		this->headScale = scale;
	}

	void Reset() override
	{
		//layerArrayRechannel.Reset();
		//layerArrayRechannelOut.SetZero();

		//ForEachIndex<NumLayers>([&](auto layerIndex)
		//	{
		//		std::get<layerIndex>(layers).Reset();
		//		layerOuts[layerIndex].SetZero();
		//		dLayerOuts[layerIndex].SetZero();
		//	});

		//headOutput.SetZero();
		//headRechannel.Reset();
		//dHeadRechannelOut.SetZero();
	}

	void SetTrainingContext(TrainingContextT<T>* context) override
	{
		layerArrayRechannel.SetTrainingContext(context);
		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).SetTrainingContext(context);
			});

		headRechannel.SetTrainingContext(context);
	}

	void RandomizeWeights() override
	{
		layerArrayRechannel.RandomizeWeights();

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).RandomizeWeights();
			});

		headRechannel.RandomizeWeights();
	}

private:
	DenseBackpropT<T, InOutChannels, Channels, false> layerArrayRechannel;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> layerArrayRechannelOut;

	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> layerOuts[NumLayers];
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dLayerOuts[NumLayers];

	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> headOutput;
	Conv1DBackpropT<T, Channels, InOutChannels, 16, true, 1> headRechannel;
	ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dHeadRechannelOut;
	float headScale = 0.1f;
};

//using A2KernelSizes = std::integer_sequence<int, 6, 6>;
//using A2Dilations = std::integer_sequence<int, 1, 3>;

using A2KernelSizes = std::integer_sequence<int, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
using A2Dilations = std::integer_sequence<int, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;


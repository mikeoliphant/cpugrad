#pragma once

#include "Dense.h"
#include "Conv1D.h"

using namespace NeuralAudio;
using namespace cpugrad;

template <typename T, int InOutChannels, int Channels, int KernelSize, int Dilation, int HeadKernelSize>
class ConvTestT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
public:
	using BackpropModelT<T, InOutChannels, InOutChannels>::trainingContext;

	void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
	{
		assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

		const size_t numSamplesIn = input.GetNumCols();
		const size_t offset = conv.GetReceptiveField();
		const size_t numSamplesOut = numSamplesIn - offset;

		trainingContext->GetBufferArena().template GetBuffer<Channels>(rechannelOut, numSamplesIn);

		rechannel.Forward(input, rechannelOut);

		trainingContext->GetBufferArena().template GetBuffer<Channels>(convOut, numSamplesOut);

		convOut.SetZero();
		conv.Forward(rechannelOut, convOut);

		trainingContext->GetBufferArena().template GetBuffer<Channels>(reluOut, numSamplesOut);

		relu.Forward(convOut, reluOut);

		oneByOne.Forward(reluOut, output);
	}

	void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
	{
		assert((input.GetNumCols() == (dOutput.GetNumCols() + GetReceptiveField())) && (input.GetNumCols() == dInput.GetNumCols()));

		size_t numSamples = dOutput.GetNumCols();

		numSamples += oneByOne.GetReceptiveField();
		auto dOneByOneOut = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(numSamples);

		dOneByOneOut.SetZero();
		oneByOne.Backward(reluOut, dOutput, dOneByOneOut);

		auto dReluOut = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(numSamples);

		relu.Backward(convOut, dOneByOneOut, dReluOut);

		trainingContext->GetBufferArena().FreeScratchBuffer(dOneByOneOut);

		numSamples += conv.GetReceptiveField();
		auto dConvOut = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(numSamples);

		dConvOut.SetZero();
		conv.Backward(rechannelOut, dReluOut, dConvOut);

		trainingContext->GetBufferArena().FreeScratchBuffer(dReluOut);

		rechannel.Backward(input, dConvOut, dInput);

		trainingContext->GetBufferArena().FreeScratchBuffer(dConvOut);

		assert(numSamples == dInput.GetNumCols());
	}

	size_t GetReceptiveField() override
	{
		return conv.GetReceptiveField() + oneByOne.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		return conv.GetNumWeights() + oneByOne.GetNumWeights() + rechannel.GetNumWeights();
	}

	void RandomizeWeights(std::mt19937& rand) override
	{
		rechannel.RandomizeWeights(rand);

		conv.RandomizeWeights(rand);

		oneByOne.RandomizeWeights(rand);
	}

	void SetWeights(std::vector<float>::iterator& inWeights) override
	{
		rechannel.SetWeights(inWeights);
		conv.SetWeights(inWeights);
		oneByOne.SetWeights(inWeights);
	}

	void SetTrainingContext(TrainingContextT<T>* context) override
	{
		BackpropModelT<T, InOutChannels, InOutChannels>::SetTrainingContext(context);

		rechannel.SetTrainingContext(context);
		conv.SetTrainingContext(context);
		oneByOne.SetTrainingContext(context);
	}

private:
	DenseBackpropT<T, InOutChannels, Channels, true> rechannel;
	ChannelBufferDynamic<T, Channels> rechannelOut;
	Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
	ChannelBufferDynamic<T, Channels> convOut;
	LeakyReLUT<T, Channels> relu;
	ChannelBufferDynamic<T, Channels> reluOut;
	Conv1DBackpropT<T, Channels, InOutChannels, HeadKernelSize, true, 1> oneByOne;
};

template <typename T, int InOutChannels, int KernelSize, int Dilation>
class TwoConvTestT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
public:
	using BackpropModelT<T, InOutChannels, InOutChannels>::trainingContext;

	void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
	{
		assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

		size_t numSamples = input.GetNumCols();

		numSamples -= conv1.GetReceptiveField();
		trainingContext->GetBufferArena().template GetBuffer<InOutChannels>(conv1Out, numSamples);

		conv1Out.SetZero();
		conv1.Forward(input, conv1Out);

		conv2.Forward(conv1Out, output);
	}

	void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
	{
		assert((input.GetNumCols() == (dOutput.GetNumCols() + GetReceptiveField())) && (input.GetNumCols() == dInput.GetNumCols()));

		size_t numSamples = dOutput.GetNumCols();

		numSamples += conv2.GetReceptiveField();
		auto dConv2Out = trainingContext->GetBufferArena().template GetScratchBuffer<InOutChannels>(numSamples);

		dConv2Out.SetZero();
		conv2.Backward(conv1Out, dOutput, dConv2Out);

		numSamples += conv1.GetReceptiveField();

		conv1.Backward(input, dConv2Out, dInput);

		trainingContext->GetBufferArena().FreeScratchBuffer(dConv2Out);

		assert(numSamples == dInput.GetNumCols());
	}

	size_t GetReceptiveField() override
	{
		return conv1.GetReceptiveField() + conv2.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		return conv1.GetNumWeights() + conv2.GetNumWeights();
	}

	void RandomizeWeights() override
	{
		conv1.RandomizeWeights();
		conv2.RandomizeWeights();
	}

	void SetWeights(std::vector<float>::iterator& inWeights) override
	{
		conv1.SetWeights(inWeights);
		conv2.SetWeights(inWeights);
	}

	void SetTrainingContext(TrainingContextT<T>* context) override
	{
		BackpropModelT<T, InOutChannels, InOutChannels>::SetTrainingContext(context);

		conv1.SetTrainingContext(context);
		conv2.SetTrainingContext(context);
	}

private:
	Conv1DBackpropT<T, InOutChannels, InOutChannels, KernelSize, true, Dilation> conv1;
	ChannelBufferDynamic<T, InOutChannels> conv1Out;
	Conv1DBackpropT<T, InOutChannels, InOutChannels, KernelSize, true, Dilation> conv2;
};

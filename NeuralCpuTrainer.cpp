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

#define BATCH_SIZE 131072


using namespace NeuralAudio;
using namespace NeuralCpuTrain;

template <typename T, int ConditionSize, int Channels, int KernelSize, int Dilation>
class WaveNetLayerBackpropT : public BackpropModelT<T, Channels, Channels>
{
public:
	using BackpropModelT<T, Channels, Channels>::Forward;
	using BackpropModelT<T, Channels, Channels>::Backward;

	void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& output, const ChannelRowSpan<T, Channels>& headOutput)
	{
		size_t numSamples = input.GetNumCols();

		conv.Forward(input, convOut.Slice(numSamples));

		conditionMixIn.Forward(condition, conditionMixInOut.Slice(numSamples));

		auto convOutMap = convOut.Slice(numSamples).GetEigenMap();
		convOutMap.noalias() += conditionMixInOut.Slice(numSamples).GetEigenMapConst();

		relu.Forward(convOut.Slice(numSamples), reluOut.Slice(numSamples));

		auto headOutputMap = headOutput.GetEigenMap();
		headOutputMap.noalias() += reluOut.Slice(numSamples).GetEigenMapConst();

		oneByOne.Forward(reluOut.Slice(numSamples), output);

		auto outputMap = output.GetEigenMap();
		outputMap.noalias() += input.GetEigenMapConst();
	}

	void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		size_t numSamples = input.GetNumCols();

		oneByOne.Backward(reluOut.Slice(numSamples), dOutput, dOneByOneOut.Slice(numSamples));

		auto dOneByOneOutMap = dOneByOneOut.Slice(numSamples).GetEigenMap();
		dOneByOneOutMap.noalias() += dHeadOutput.GetEigenMapConst();

		relu.Backward(convOut, dOneByOneOut.Slice(numSamples), dReluOut);

		conditionMixIn.Backward(condition, dReluOut.Slice(numSamples), dConditionMixInOut.Slice(numSamples));	// dConditionMixInOut not used - can optimize

		conv.Backward(input, dReluOut.Slice(numSamples), dInput);

		auto dInputMap = dInput.GetEigenMap();
		dInputMap.noalias() += dOutput.GetEigenMapConst();
	}

	void BackwardNoLayerOutput(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		size_t numSamples = input.GetNumCols();

		relu.Backward(convOut.Slice(numSamples), dHeadOutput, dReluOut.Slice(numSamples));
		conditionMixIn.Backward(condition, dReluOut.Slice(numSamples), dConditionMixInOut.Slice(numSamples));	// dConditionMixInOut not used - can optimize
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
		conv.Reset();
		convOut.SetZero();
		conditionMixIn.Reset();
		conditionMixInOut.SetZero();
		reluOut.SetZero();
		oneByOne.Reset();
		dOneByOneOut.SetZero();
		dReluOut.SetZero();
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

template <typename T, int InOutChannels, int Channels, typename KernelSizeSequence, typename DilationsSequence>
class A2BackpropT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
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

			layerArrayRechannel.Forward(input, layerArrayRechannelOut.Slice(numSamples));

			ForEachIndex<NumLayers>([&](auto layerIndex)
				{
					if constexpr (layerIndex == 0)
					{
						std::get<layerIndex>(layers).Forward(layerArrayRechannelOut.Slice(numSamples), input, layerOuts[layerIndex].Slice(numSamples), headOutput.Slice(numSamples));
					}
					else
					{
						std::get<layerIndex>(layers).Forward(layerOuts[layerIndex - 1].Slice(numSamples), input, layerOuts[layerIndex].Slice(numSamples), headOutput.Slice(numSamples));
					}
				});

			headRechannel.Forward(headOutput.Slice(numSamples), output);

			auto outputMap = output.GetEigenMap();
			outputMap *= headScale;
		}

		void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
		{
			size_t numSamples = input.GetNumCols();

			auto dOutputMap = dOutput.GetEigenMap();
			dOutputMap *= headScale;

			headRechannel.Backward(headOutput.Slice(numSamples), dOutput, dHeadRechannelOut.Slice(numSamples));

			ForEachIndex<NumLayers>([&](auto layerIndexForward)
				{
					constexpr auto layerIndexBackward = NumLayers - 1 - layerIndexForward;

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
			layerArrayRechannel.Reset();
			layerArrayRechannelOut.SetZero();

			ForEachIndex<NumLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).Reset();
					layerOuts[layerIndex].SetZero();
					dLayerOuts[layerIndex].SetZero();
				});

			headRechannel.Reset();
			dHeadRechannelOut.SetZero();
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

		void ApplyGradients(float scale) override
		{
			layerArrayRechannel.ApplyGradients(scale);

			ForEachIndex<NumLayers>([&](auto layerIndex)
				{
					std::get<layerIndex>(layers).ApplyGradients(scale);
				});

			headRechannel.ApplyGradients(scale);
		}

	private:
		DenseBackpropT<T, InOutChannels, Channels, false> layerArrayRechannel;
		ChannelBuffer<T, Channels, BATCH_SIZE> layerArrayRechannelOut;

		ChannelBuffer<T, Channels, BATCH_SIZE> layerOuts[NumLayers];
		ChannelBuffer<T, Channels, BATCH_SIZE> dLayerOuts[NumLayers];

		ChannelBuffer<T, Channels, BATCH_SIZE> headOutput;
		Conv1DBackpropT<T, Channels, InOutChannels, 16, true, 1> headRechannel;
		ChannelBuffer<T, Channels, BATCH_SIZE> dHeadRechannelOut;
		float headScale = 0.1f;
};

//using A2KernelSizes = std::integer_sequence<int, 6, 6>;
//using A2Dilations = std::integer_sequence<int, 1, 3>;

using A2KernelSizes = std::integer_sequence<int, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
using A2Dilations = std::integer_sequence<int, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

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

	//convTrainer->TestWav(R"(C:\Share\Recordings\NAM\TZ3-sweep-v3.wav)", R"(C:\Share\Recordings\NAM\BossSD1CaptureNeuralAudio.wav)");

	//convTrainer->TestDelay(2);

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	TestNAM(R"(C:\Code\NeuralCpuTrainer\BossWN-a2lite.nam)");

	//auto a2 = new A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>();

	//auto modelTrainer = new ModelTrainerT<float, BATCH_SIZE>(*a2);

	//modelTrainer->TestDelay(0);

	//modelTrainer->TestWav(R"(C:\Share\Recordings\NAM\TZ3-sweep-v3.wav)", R"(C:\Share\Recordings\NAM\BossSD1CaptureNeuralAudio.wav)");

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}


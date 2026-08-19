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

	auto modelTrainer = new ModelTrainerT<float>(*a2);

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

	std::vector<float> verifyOutput(numSamples);

	modelTrainer->VerifyModel(input.data(), verifyOutput.data(), numSamples);

	MSELossT<float> mseLoss;

	double err = mseLoss.GetTotSquared(verifyOutput.data(), namOutput.data(), numSamples, a2->GetReceptiveField()) / (float)(numSamples - a2->GetReceptiveField());

	std::cout << "Err: " << err << std::endl;
}

template <typename T, int InOutChannels, int Channels, int KernelSize, int Dilation, int HeadKernelSize>
class ConvTestT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
	public:
		void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
		{
			size_t numSamples = input.GetNumCols();

			rechannel.Forward(input, rechannelOut.Slice(numSamples));

			conv.Forward(rechannelOut.Slice(numSamples), convOut.Slice(numSamples));

			relu.Forward(convOut.Slice(numSamples), reluOut.Slice(numSamples));

			oneByOne.Forward(reluOut.Slice(numSamples), output);
		}

		void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
		{
			size_t numSamples = input.GetNumCols();

			oneByOne.Backward(reluOut.Slice(numSamples), dOutput, dOneByOneOut.Slice(numSamples));

			relu.Backward(convOut.Slice(numSamples), dOneByOneOut.Slice(numSamples), dReluOut.Slice(numSamples));

			conv.Backward(rechannelOut, dReluOut.Slice(numSamples), dConvOut.Slice(numSamples));

			rechannel.Backward(input, dConvOut.Slice(numSamples), dInput);
		}

		size_t GetReceptiveField() override
		{
			return conv.GetReceptiveField() + oneByOne.GetReceptiveField();
		}

		size_t GetNumWeights() override
		{
			return conv.GetNumWeights() + oneByOne.GetNumWeights() + rechannel.GetNumWeights();
		}

		void RandomizeWeights() override
		{
			rechannel.RandomizeWeights();

			conv.RandomizeWeights();

			oneByOne.RandomizeWeights();
		}

		void SetWeights(std::vector<float>::iterator& inWeights) override
		{
			rechannel.SetWeights(inWeights);
			conv.SetWeights(inWeights);
			oneByOne.SetWeights(inWeights);
		}

		void Reset() override
		{
			rechannel.Reset();
			rechannelOut.SetZero();
			conv.Reset();
			convOut.SetZero();
			dConvOut.SetZero();
			reluOut.SetZero();
			dReluOut.SetZero();
			oneByOne.Reset();
			dOneByOneOut.SetZero();
		}

		void ResetGradients() override
		{
			rechannel.ResetGradients();
			conv.ResetGradients();
			oneByOne.ResetGradients();
		}


		void ApplyGradients(float scale) override
		{
			rechannel.ApplyGradients(scale);
			conv.ApplyGradients(scale);
			oneByOne.ApplyGradients(scale);
		}

	private:
		DenseBackpropT<T, InOutChannels, Channels, true> rechannel;
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> rechannelOut;
		Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> convOut;
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dConvOut;
		LeakyReLUT<T, Channels> relu;
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dReluOut;
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> reluOut;
		Conv1DBackpropT<T, Channels, InOutChannels, HeadKernelSize, true, 1> oneByOne;
		//DenseBackpropT<T, Channels, InOutChannels, true> oneByOne;
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dOneByOneOut;
};

int main()
{
	//TestNAM(R"(C:\Code\NeuralCpuTrainer\BossWN-a2lite.nam)");


	//DenseBackpropT<float, 1, 1, false> dense;
	//auto denseTrainer = new ModelTrainerT<float>(dense);

	//denseTrainer->TestIdentity();
	//
	//auto denseData = denseTrainter->GenerateSin(48000 * 180);
	//denseTrainter->TestIdentity(denseData);


	//denseTrainter->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//Conv1DBackpropT<float, 1, 1, 3, true, 1> convBackprop;
	//auto convTrainer = new ModelTrainerT<float>(convBackprop);

	//convTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//convTrainer->TestDelay(1);

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	//auto convTest = new ConvTestT<float, 1, 3, 6, 1, 16>();

	//auto convTestTrainer = new ModelTrainerT<float>(*convTest);

	//auto data = convTestTrainer->GenerateSin(48000 * 180);
	//convTestTrainer->TestIdentity(data);

	//convTestTrainer->TestXOR(1);
	//convTestTrainer->TestIdentity();

	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");
	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\BossSD1.wav)");

	//auto a2 = new A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>();

	using TestKernelSizes = std::integer_sequence<int, 6>; //, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
	using TestDilations = std::integer_sequence<int, 1>; //, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

	auto a2 = new A2BackpropT<float, 1, 3, TestKernelSizes, TestDilations>();

	auto modelTrainer = new ModelTrainerT<float>(*a2);

	auto a2data = modelTrainer->GenerateSin(48000 * 180);
	modelTrainer->TestIdentity(a2data);

	modelTrainer->TestIdentity();

	//modelTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\BossSD1.wav)");

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}


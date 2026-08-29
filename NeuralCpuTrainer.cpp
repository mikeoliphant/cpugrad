#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <xmmintrin.h>

#include "NeuralModel.h"
#define DR_WAV_IMPLEMENTATION
#include "WaveNet.h"
#include "WaveNetBackprop.h"
#include "ModelTrainer.h"
#include "NAM.h"
#include "Dataset.h"

using namespace NeuralAudio;
using namespace NeuralCpuTrain;

static void TestNAM(std::filesystem::path modelPath)
{
	NeuralAudio::NeuralModelLoader loader;

	auto namModel = loader.CreateFromFile(modelPath);

	size_t numSamples = 24000; //48000 * 3;

	auto modelTrainer = new ModelTrainerT<float, A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>>();

	DataGen dataGen;

	auto input = dataGen.GenerateSin(numSamples, numSamples);

	std::vector<float> namOutput(numSamples);

	namModel->Process(input.data(), namOutput.data(), numSamples);

	std::ifstream jsonStream(modelPath, std::ifstream::binary);

	nlohmann::json modelJson;
	jsonStream >> modelJson;

	std::vector<float> weights = modelJson.at("weights");

	auto it = weights.begin();

	modelTrainer->GetModel()->SetWeights(it);
	modelTrainer->GetModel()->SetHeadScale(*it);

	std::vector<float> verifyOutput(numSamples);

	modelTrainer->VerifyModel(input.data(), verifyOutput.data(), numSamples);

	MSELossT<float> mseLoss;

	double err = mseLoss.GetTotSquared(verifyOutput.data(), namOutput.data(), numSamples, modelTrainer->GetReceptiveField()) / (float)(numSamples - modelTrainer->GetReceptiveField());

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

		void AddWeightGradients(OptimizerT<T>& optimizer) override
		{
			rechannel.AddWeightGradients(optimizer);
			conv.AddWeightGradients(optimizer);
			oneByOne.AddWeightGradients(optimizer);
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
		ChannelBuffer<T, Channels, MAX_BATCH_SIZE> dOneByOneOut;
};

int main()
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	DataGen dataGen(123);

	size_t numSamples = 48000 * 10;

	auto randData = dataGen.GenerateRandom(numSamples);
	auto sinData = dataGen.GenerateSin(numSamples, 8192);
	auto delayData = dataGen.GenerateDelay(1, numSamples);
	auto xorData = dataGen.GenerateXOR(1, numSamples);

	//TestNAM(R"(C:\Code\NeuralCpuTrainer\BossWN-a2lite.nam)");


	//DenseBackpropT<float, 1, 1, false> dense;
	//auto denseTrainer = new ModelTrainerT<float>(dense);

	//denseTrainer->TestBackprop(0, randData.data(), randData.data(), 16000);

	//denseTrainer->TrainIdentity(randData);


	//denseTrainter->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//Conv1DBackpropT<float, 1, 1, 3, true, 1> convBackprop;
	//auto convTrainer = new ModelTrainerT<float>(convBackprop);

	//convTrainer->TrainIdentity(randData);

	////convTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	//auto convTest = new ConvTestT<float, 1, 16, 3, 1, 1>();

	//auto convTestTrainer = new ModelTrainerT<float>(*convTest);

	//convTestTrainer->TrainIdentity(randData);

	//convTestTrainer->Train(xorData);
	//convTestTrainer->TestIdentity();

	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");
	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\BossSD1.wav)");

	//auto a2 = new A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>();

	using TestKernelSizes = std::integer_sequence<int, 6>;//, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
	using TestDilations = std::integer_sequence<int, 1>;//, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

	//auto a2 = new A2BackpropT<float, 1, 3, TestKernelSizes, TestDilations>();
	//auto a2 = new ();

	//std::cout << sizeof(A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>) << std::endl;

	auto modelTrainer = new ModelTrainerT<float, A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>>();
	
	//modelTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

	//modelTrainer->TrainIdentity(sinData);

	modelTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\BossSD1.wav)");

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}


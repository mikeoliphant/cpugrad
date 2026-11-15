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
void STFTTestTorch()
{
	const int64_t num_samples = 8192;
	const double sample_rate = 16000.0;

	// Calculate step size (1 / 16000) and the exclusive upper boundary
	double step_size = 1.0 / sample_rate;
	double end_time = (double)num_samples / sample_rate;

	// This generates exactly 8192 points from 0 to 0.5119375
	torch::Tensor t = torch::arange(0.0, end_time, step_size,
		torch::TensorOptions().dtype(torch::kFloat64));

	// 2. Generate target signal: A clean 440Hz tone with an exponential decay
	auto decay = torch::exp(-3.0 * t);
	auto targetSignal = torch::sin(2 * M_PI * 440.0 * t) * decay;
	auto targetTensor = targetSignal.view({ 1, 1, -1 });

	// 1. Maintain standard batch/channel spatial matrix tracking [Batch * Channel, Time]
	auto x_flat = targetTensor.view({ targetTensor.size(0) * targetTensor.size(1), -1 });

	int64_t fft_size = 2048;
	int64_t hop_size = 240;
	int64_t win_length = 1200;

	auto n = torch::arange(win_length, torch::TensorOptions().dtype(torch::kFloat));
	torch::Tensor window = 0.5 - 0.5 * torch::cos(2.0 * M_PI * n / static_cast<double>(win_length));

	// 2. MANUAL PADDING CORRECTION: Replicate Python's center=True (n_fft // 2 on both sides)
	// using explicit reflection padding to ensure identical frame counts in C++
	int64_t pad_amount = fft_size / 2;
	auto x_padded = torch::reflection_pad1d(x_flat.unsqueeze(1), { pad_amount, pad_amount }).squeeze(1);

	// 3. Execute STFT with center=false since padding is handled manually
	auto stft_output = torch::stft(x_padded, fft_size, hop_size, win_length, window,
		/*center=*/false, std::string("reflect"),
		/*normalized=*/false, /*onesided=*/true, /*return_complex=*/true);

	torch::Tensor real_view = torch::view_as_real(stft_output).to(torch::kCPU).to(torch::kFloat32).contiguous();

	auto sizes = real_view.sizes();
	auto strides = real_view.strides();

	// 3. Fetch data pointer via void* cast to cleanly avoid the internal template bug
	float* raw_ptr = static_cast<float*>(real_view.data_ptr());

	// 4. Manually construct a 4D TensorAccessor using 2 template arguments <Type, Dimensions>
	auto stft_acc = torch::TensorAccessor<float, 4>(
		raw_ptr,
		sizes.data(),
		strides.data()
	);

	// 4. Safely iterate using multi-dimensional brackets
	int64_t batch_size = sizes[0];
	int64_t num_bins = sizes[1];
	int64_t num_frames = sizes[2];

	StftFrameProcessor<float> cpugradStft(fft_size, hop_size, win_length);
	cpugradStft.SetNumSamples(num_samples);

	std::cout << "torch frames: " << num_frames << std::endl;
	std::cout << "cpugrad frames: " << cpugradStft.GetOutputFrames() << std::endl;

	double mse = 0;

	std::vector<float> targetVec = TensorToVector(targetSignal);

	std::vector<float> frameData(fft_size);
	std::vector<float> realData(num_bins);
	std::vector<float> imagData(num_bins);

	for (int64_t f = 0; f < num_frames; ++f)
	{
		cpugradStft.ProcessForwardFrame(targetVec.data(), f, frameData, realData, imagData);

		for (int64_t k = 0; k < num_bins; ++k)
		{
			float real = stft_acc[0][k][f][0];
			float imag = stft_acc[0][k][f][1];

			mse += ((real - realData[k]) * (real - realData[k])) + ((imag - imagData[k]) * (imag - imagData[k]));
		}
	}

	mse /= static_cast<double>(num_frames * num_bins);

	std::cout << "mse: " << mse << std::endl;
}

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

	// 2. Generate target signal: A clean 440Hz tone with an exponential decay
	auto decay = torch::exp(-3.0 * t);
	auto target_signal = (torch::sin(2 * M_PI * 440.0 * t) * decay).to(torch::kFloat);

	// Generate output signal: Fundamental tone (442Hz) + a harsh 3000Hz noise component
	auto noise = 0.25 * torch::sin(2 * M_PI * 3000.0 * t);
	auto output_signal = ((torch::sin(2 * M_PI * 442.0 * t) * decay) + noise).to(torch::kFloat);

	// Both signals are float32 so torch and cpugrad see bit-identical inputs at the same precision

	//std::vector<StftWindowConfig> configs({ { 2048, 240, 1200 }, { 1024, 120, 600 }, { 512, 50, 240 } });
	std::vector<StftWindowConfig> configs({ { 2048, 240, 1200 } });

	float specWeight = 0.0f;
	float logMagWeight = 1.0f;
	float linMagWeight = 0.0f;

	MultiResolutionSTFTLoss torchLoss(configs);

	torchLoss.SetWeights(specWeight, logMagWeight, linMagWeight);

	double loss = torchLoss.GetMeanLoss(output_signal, target_signal);

	std::cout << "torch loss: " << std::format("{:.8f}", loss) << std::endl;

	auto torchGradient = torchLoss.GetLossGradient(output_signal, target_signal);

	MultiResolutionStftLossT<float> cpugradLoss(configs);

	cpugradLoss.SetWeights(specWeight, logMagWeight);

	auto outputSignalVec = TensorToVector(output_signal);
	auto targetSignalVec = TensorToVector(target_signal);

	loss = cpugradLoss.GetMeanLoss(outputSignalVec.data(), targetSignalVec.data(), outputSignalVec.size());

	std::cout << "cpugrad loss: " << std::format("{:.8f}", loss) << std::endl;

	std::vector<float> cpugradGradient(outputSignalVec.size());

	cpugradLoss.ComputeLoss(outputSignalVec.data(), targetSignalVec.data(), cpugradGradient.data(), outputSignalVec.size(), 1.0);

	MSELossT<float> mseLoss;

	double mse = mseLoss.GetMeanLoss(torchGradient.data(), cpugradGradient.data(), torchGradient.size());

	std::cout << "Gradient MSE: " << mse << std::endl;

	for (size_t i = 0; i < torchGradient.size(); i++)
	{
		std::cout << std::format("{:.8f}", torchGradient[i]) << " " << std::format("{:.8f}", cpugradGradient[i]) << std::endl;
	}
}

#endif

int main()
{
#ifdef HAVE_TORCH
	//STFTTestTorch();
	SpecTestTorch();
#endif

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


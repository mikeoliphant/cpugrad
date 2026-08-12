#pragma once

#include <cmath>
#include <filesystem>
#include "dr_wav.h"

#include "WaveNetBackprop.h"

namespace NeuralCpuTrain
{
	template <typename T>
	class LossT
	{
		public:
			virtual ~LossT() = default;
			virtual void ComputeLoss(const T* output, const T *target, T* outGradient, size_t numSamples, size_t receptiveFieldSize) = 0;
			virtual T GetTotSquared(const T* output, const T* target, size_t numSamples, size_t receptiveFieldSize) = 0;
	};

	template <typename T>
	class MSELossT : public LossT<T>
	{
		public:
			void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, size_t receptiveFieldSize) override
			{
				for (size_t t = 0; t < numSamples; t++)
				{
					if (t < receptiveFieldSize)
						outGradient[t] = 0;
					else
						outGradient[t] = TCONST(2) * (output[t] - target[t]);
				}
			}

			T GetTotSquared(const T* output, const T* target, size_t numSamples, size_t receptiveFieldSize) override
			{
				T tot = TCONST(0);

				for (size_t t = 0; t < numSamples; t++)
				{
					if (t >= receptiveFieldSize)
					{
						T diff = output[t] - target[t];
						tot += (diff * diff);
					}
				}

				return tot;
			}
	};

	template <typename T, int BatchSize>
	class ModelTrainerT
	{
		public:
			ModelTrainerT(BackpropModelT<T, 1, 1>& modelBackprop) :
				modelBackprop(modelBackprop),
				lossFunction(std::make_unique<MSELossT<T>>())
			{
			}

			float VerifyModel(const T* input, const T* target, const size_t totalSamples)
			{
				size_t receptiveField = modelBackprop.GetReceptiveField();
				size_t validSampleCount = BatchSize - receptiveField;

				size_t currentOffset = 0;
				int samplesRemaining = (int)totalSamples;

				double totErr = 0.0;

				while (samplesRemaining > 0)
				{
					size_t thisBatchSize = (size_t)std::min(samplesRemaining, BatchSize);

					T* batchInPtr = batchInput.GetData();
					std::copy(input + currentOffset, input + currentOffset + thisBatchSize, batchInPtr);

					auto batchTargetPtr = batchTarget.GetData();
					std::copy(target + currentOffset, target + currentOffset + thisBatchSize, batchTargetPtr);

					forwardOutput.SetZero();

					modelBackprop.Reset();

					modelBackprop.Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

					totErr += lossFunction->GetTotSquared(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), thisBatchSize, receptiveField);

					currentOffset += validSampleCount;
					samplesRemaining -= (int)validSampleCount;
				}

				return (totErr / static_cast<float>(totalSamples));
			}

			void TrainModel(const float* input, const float* target, const size_t totalSamples, const float* verifyInput, const float* verifyTarget, const size_t verifySamples)
			{
				modelBackprop.RandomizeWeights();

				size_t receptiveField = modelBackprop.GetReceptiveField();
				size_t validSampleCount = BatchSize - receptiveField;

				for (int iter = 0; iter < 20000; ++iter)
				{
					size_t currentOffset = 0;
					int samplesRemaining = (int)totalSamples;

					while (samplesRemaining > 0)
					{
						size_t thisBatchSize = (size_t)std::min(samplesRemaining, BatchSize);

						float* batchInPtr = batchInput.GetData();
						std::copy(input + currentOffset, input + currentOffset + thisBatchSize, batchInPtr);

						auto batchTargetPtr = batchTarget.GetData();
						std::copy(target + currentOffset, target + currentOffset + thisBatchSize, batchTargetPtr);

						forwardOutput.SetZero();

						modelBackprop.Reset();

						modelBackprop.Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

						lossFunction->ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), thisBatchSize, receptiveField);

						modelBackprop.Backward(batchInput.Slice(thisBatchSize), outputGradient.Slice(thisBatchSize), layerOutputGradient.Slice(thisBatchSize));

						modelBackprop.ApplyGradients(0.005f / (float)validSampleCount);

						currentOffset += validSampleCount;
						samplesRemaining -= (int)validSampleCount;
					}

					float mse = VerifyModel(verifyInput, verifyTarget, verifySamples);

					std::cout << "Iter: " << iter << " MSE: " << mse << std::endl;
				}
			}

			std::vector<float> GenerateRandom(size_t numSamples)
			{
				std::vector<float> rand(numSamples);

				std::random_device rd;
				std::mt19937 gen(rd());

				std::uniform_real_distribution<float> dis(0.0f, 1.0f);

				for (size_t i = 0; i < numSamples; i++)
					rand[i] = dis(gen);

				return rand;
			}

			void TestIdentity()
			{
				const size_t totalSamples = 48000 * 180;

				auto rand = GenerateRandom(totalSamples);

				TrainModel(rand.data(), rand.data(), totalSamples, rand.data(), rand.data(), totalSamples);
			}

			void TestDelay(size_t delay)
			{
				const size_t totalSamples = 48000 * 10;

				auto rand = GenerateRandom(totalSamples);

				std::vector<float> target(totalSamples);

				for (size_t i = 0; i < totalSamples; i++)
				{
					if (i < delay)
						target[i] = 0;
					else
						target[i] = rand[i - delay];
				}

				TrainModel(rand.data(), target.data(), totalSamples, rand.data(), target.data(), totalSamples);
			}

			void TestWav(const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath)
			{
				unsigned int channels;
				unsigned int sampleRate;
				drwav_uint64 numFrames;

				float* inData = drwav_open_file_and_read_pcm_frames_f32(inWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);
				float* targetData = drwav_open_file_and_read_pcm_frames_f32(targetWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);

				size_t verifyFrames = 48000 * 9;
				size_t verifyOffset = (size_t)numFrames - verifyFrames;

				TrainModel(inData, targetData, (size_t)numFrames - verifyFrames, inData + verifyOffset, targetData + verifyOffset, verifyFrames);
			}

		private:
			BackpropModelT<float, 1, 1>& modelBackprop;
			ChannelBuffer<float, 1, BatchSize> batchInput;
			ChannelBuffer<float, 1, BatchSize> batchTarget;
			ChannelBuffer<float, 1, BatchSize> forwardOutput;
			ChannelBuffer<float, 1, BatchSize> outputGradient;
			ChannelBuffer<float, 1, BatchSize> layerOutputGradient;
			std::unique_ptr<LossT<T>> lossFunction;
	};
}

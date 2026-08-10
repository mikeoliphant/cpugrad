#pragma once

#include <cmath>
#include <filesystem>
#include "dr_wav.h"

#include "WaveNetBackprop.h"

namespace NeuralCpuTrain
{

	template <int BatchSize>
	class ModelTrainerT
	{
		public:
			ModelTrainerT(BackpropModelT<float, 1, 1>& modelBackprop) :
				modelBackprop(modelBackprop)
			{
			}

			float VerifyModel(const float* input, const float* target, const size_t totalSamples)
			{
				size_t receptiveField = modelBackprop.GetReceptiveField();
				size_t validSampleCount = BatchSize - receptiveField;

				size_t currentOffset = 0;
				int samplesRemaining = (int)totalSamples;

				double totErr = 0.0;

				while (samplesRemaining > 0)
				{
					size_t thisBatchSize = (size_t)std::min(samplesRemaining, BatchSize);

					float* batchInPtr = batchInput.GetData();
					std::copy(input + currentOffset, input + currentOffset + thisBatchSize, batchInPtr);

					auto batchTargetPtr = batchTarget.GetData();
					std::copy(target + currentOffset, target + currentOffset + thisBatchSize, batchTargetPtr);

					forwardOutput.SetZero();

					modelBackprop.Reset();

					modelBackprop.Forward(batchInput, forwardOutput);

					auto map = outputGradient.GetEigenMap();

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

					currentOffset += validSampleCount;
					samplesRemaining -= (int)validSampleCount;
				}

				return (totErr / static_cast<float>(totalSamples));
			}

			void TestModel(const float* input, const float* target, const size_t totalSamples, const float* verifyInput, const float* verifyTarget, const size_t verifySamples)
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

						modelBackprop.Forward(batchInput, forwardOutput);

						auto map = outputGradient.GetEigenMap();

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
							}
						}

						modelBackprop.Backward(batchInput, outputGradient, layerOutputGradient);

						modelBackprop.DivideWeights(static_cast<float>((float)validSampleCount));

						modelBackprop.ApplyGradients(0.001f);

						currentOffset += validSampleCount;
						samplesRemaining -= (int)validSampleCount;
					}

					float mse = VerifyModel(verifyInput, verifyTarget, verifySamples);

					std::cout << "Iter: " << iter << " MSE: " << mse << std::endl;
				}
			}

			void TestIdentity()
			{
				const size_t totalSamples = 48000;

				std::vector<float> input(totalSamples);
				std::vector<float> target(totalSamples);

				for (size_t i = 0; i < totalSamples; ++i)
				{
					input[i] = std::sin(static_cast<float>(i) * 0.01f);
					target[i] = input[i];
				}

				TestModel(input.data(), target.data(), totalSamples, input.data(), target.data(), totalSamples);
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

				TestModel(inData, targetData, (size_t)numFrames - verifyFrames, inData + verifyOffset, targetData + verifyOffset, verifyFrames);
			}

		private:
			BackpropModelT<float, 1, 1>& modelBackprop;
			ChannelBuffer<float, 1, BatchSize> batchInput;
			ChannelBuffer<float, 1, BatchSize> batchTarget;
			ChannelBuffer<float, 1, BatchSize> forwardOutput;
			ChannelBuffer<float, 1, BatchSize> outputGradient;
			ChannelBuffer<float, 1, BatchSize> layerOutputGradient;

	};
}

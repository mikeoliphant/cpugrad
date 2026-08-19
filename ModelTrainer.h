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
			virtual std::string& GetName() { return name; }
			virtual void ComputeLoss(const T* output, const T *target, T* outGradient, size_t numSamples, size_t receptiveFieldSize, T scaleFactor) = 0;
			virtual double GetTotSquared(const T* output, const T* target, size_t numSamples, size_t receptiveFieldSize) = 0;

		protected:
			std::string name;
	};

	template <typename T>
	class MSELossT : public LossT<T>
	{
		public:
			MSELossT()
			{
				this->name = "MSE";
			}

			void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, size_t receptiveFieldSize, T scaleFactor) override
			{
				for (size_t t = 0; t < numSamples; t++)
				{
					if (t < receptiveFieldSize)
						outGradient[t] = 0;
					else
						outGradient[t] = ((TCONST(2) * (output[t] - target[t])) / static_cast<T>(numSamples - receptiveFieldSize)) * scaleFactor;
				}
			}

			double GetTotSquared(const T* output, const T* target, size_t numSamples, size_t receptiveFieldSize) override
			{
				T tot = TCONST(0);

				for (size_t t = receptiveFieldSize; t < numSamples; t++)
				{
					T diff = output[t] - target[t];
					tot += (diff * diff);
				}

				return tot;
			}
	};

	template <typename T>
	class ESRLossT : public LossT<T>
	{
		static constexpr T epsilon = TCONST(1e-8);

	public:
		ESRLossT()
		{
			this->name = "ESR";
		}

		void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, size_t receptiveFieldSize, T scaleFactor) override
		{
			double totEnergy = 0;

			for (size_t t = receptiveFieldSize; t < numSamples; t++)
			{
				totEnergy += (target[t] * target[t]);
			}

			totEnergy += epsilon;

			for (size_t t = 0; t < numSamples; t++)
			{
				if (t < receptiveFieldSize)
					outGradient[t] = 0;
				else
					outGradient[t] = ((TCONST(2) * (output[t] - target[t])) / static_cast<T>(totEnergy)) * scaleFactor;
			}
		}

		double GetTotSquared(const T* output, const T* target, size_t numSamples, size_t receptiveFieldSize) override
		{
			double totEnergy = 0;

			for (size_t t = receptiveFieldSize; t < numSamples; t++)
			{
				totEnergy += (target[t] * target[t]);
			}

			totEnergy += epsilon;

			T tot = TCONST(0);

			for (size_t t = 0; t < numSamples; t++)
			{
				if (t >= receptiveFieldSize)
				{
					T diff = output[t] - target[t];
					tot += (diff * diff);
				}
			}

			return (tot / totEnergy) * static_cast<T>(numSamples - receptiveFieldSize);
		}
	};

	struct TrainingDataBatch
	{
		size_t Offset;
		size_t Size;
	};

	class TrainingData
	{
		public:
			TrainingData(size_t trainingOffset, size_t trainingSamples, size_t batchSize, size_t batchSkip) :
				gen(123)
			{
				batches.reserve((trainingSamples / batchSize) + 1);

				int samplesRemaining = (int)trainingSamples;
				size_t currentOffset = trainingOffset;

				while (samplesRemaining > 0)
				{
					batches.emplace_back(currentOffset, std::min((size_t)samplesRemaining, batchSize) );

					currentOffset += batchSkip;
					samplesRemaining -= (int)batchSkip;
				}
			}

			std::vector<TrainingDataBatch>& Batches()
			{
				return batches;
			}

			void ShuffleBatches()
			{
				std::shuffle(batches.begin(), batches.end(), gen);

			}

		private:
			std::vector<TrainingDataBatch> batches;
			std::mt19937 gen;
	};

	template <typename T>
	class ModelTrainerT
	{
		public:
			ModelTrainerT(BackpropModelT<T, 1, 1>& modelBackprop) :
				modelBackprop(modelBackprop),
				lossFunction(std::make_unique<MSELossT<T>>()),
				lossEvalFunction(std::make_unique<MSELossT<T>>())
			{
			}

			void VerifyModel(const T* input, T* output, const size_t totalSamples)
			{
				size_t receptiveField = modelBackprop.GetReceptiveField();
				//size_t batchSize = receptiveField + 8192;
				size_t batchSize = MAX_BATCH_SIZE;
				size_t validSampleCount = batchSize - receptiveField;

				size_t currentOffset = 0;	// ** NOTE - we will have invalid data for the initial receptive field
				int samplesRemaining = (int)totalSamples;

				std::vector<float> verifyOutput(totalSamples);

				while (samplesRemaining > 0)
				{					
					size_t thisBatchSize = (size_t)std::min(samplesRemaining, (int)batchSize);

					T* batchInPtr = batchInput.GetData();
					std::copy(input + currentOffset, input + currentOffset + thisBatchSize, batchInPtr);

					forwardOutput.SetZero();

					modelBackprop.Reset();

					modelBackprop.Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

					T* forwardOutputPtr = forwardOutput.GetData();
					std::copy(forwardOutputPtr + receptiveField, forwardOutputPtr + thisBatchSize, output + currentOffset + receptiveField);

					currentOffset += validSampleCount;
					samplesRemaining -= (int)validSampleCount;
				}
			}

			void TrainModel(const T* input, T* target, const size_t totalSamples, const T* verifyInput, const T* verifyTarget, const size_t verifySamples)
			{
				float learningRate = 0.005f;

				//ApplyHPF(target, totalSamples);

				modelBackprop.RandomizeWeights();

				size_t receptiveField = modelBackprop.GetReceptiveField();
				size_t batchSize = receptiveField + 8192;
				size_t validSampleCount = batchSize - receptiveField;


				if (receptiveField > batchSize)
					throw std::runtime_error("Model receptive field exceeds batch size");

				TrainingData trainingData(receptiveField, totalSamples, batchSize, batchSize - receptiveField);

				size_t numBatches = 16;

				std::cout << "Training " << trainingData.Batches().size() << " batches of size " << (batchSize - receptiveField) << " (+" << receptiveField << ")" << std::endl;

				std::vector<T> verifyOutput(verifySamples);

				for (int iter = 0; iter < 20000; ++iter)
				{
					trainingData.ShuffleBatches();

					modelBackprop.ResetGradients();

					size_t currentBatchNum = 0;
					size_t startBatchNum = 0;
					size_t totBatches = trainingData.Batches().size();

					for (TrainingDataBatch& batch : trainingData.Batches())
					{
						size_t thisBatchSize = batch.Size;
						size_t thisBatchStart = batch.Offset - receptiveField;

						float* batchInPtr = batchInput.GetData();
						std::copy(input + thisBatchStart, input + thisBatchStart + thisBatchSize, batchInPtr);

						auto batchTargetPtr = batchTarget.GetData();
						std::copy(target + thisBatchStart, target + thisBatchStart + thisBatchSize, batchTargetPtr);

						forwardOutput.SetZero();

						modelBackprop.Reset();

						modelBackprop.Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

						size_t endBatch = std::min(startBatchNum + numBatches, totBatches - 1);

						float lossScale = 1.0f / (float)(endBatch - startBatchNum);

						//ApplyHPF(forwardOutput.GetData() + receptiveField, thisBatchSize - receptiveField);

						lossFunction->ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), thisBatchSize, receptiveField, lossScale);

						//std::cout << "Batch loss: " << lossFunction->GetTotSquared(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), thisBatchSize, receptiveField) / (float)validSampleCount << std::endl;

						modelBackprop.Backward(batchInput.Slice(thisBatchSize), outputGradient.Slice(thisBatchSize), layerOutputGradient.Slice(thisBatchSize));

						currentBatchNum++;

						if (((currentBatchNum  % numBatches) == 0) || (currentBatchNum == (totBatches - 1)))
						{
							modelBackprop.ApplyGradients(learningRate);
							modelBackprop.ResetGradients();

							startBatchNum = currentBatchNum - 1;
						}
					}

					VerifyModel(verifyInput, verifyOutput.data(), verifySamples);

					double err = lossEvalFunction->GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);

					std::cout << "Epoch: " << iter << " " << lossEvalFunction->GetName() << ": " << std::format("{:.10f}", err) << std::endl;
				}
			}

			void ApplyHPF(T* data, size_t numSamples)
			{
				constexpr T coefficient = T(0.95);

				for (size_t n = numSamples - 1; n > 0; --n)
				{
					data[n] = data[n] - (coefficient * data[n - 1]);
				}

				data[0] = data[0] * (1.0f - coefficient);
			}

			std::vector<float> GenerateSin(size_t numSamples)
			{
				std::vector<float> data(numSamples);

				size_t sweep = numSamples; // 8192;

				for (size_t i = 0; i < numSamples; i++)
					data[i] = (float)std::sin(i * 0.01) * ((float)(i % sweep) / (float)sweep);

				return data;
			}

			std::vector<float> GenerateRandom(size_t numSamples)
			{
				std::vector<float> rand(numSamples);

				//std::mt19937 gen(123);
				std::random_device rd;
				std::mt19937 gen(rd());

				std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

				for (size_t i = 0; i < numSamples; i++)
					rand[i] = dis(gen);

				return rand;
			}

			void TestIdentity()
			{
				const size_t totalSamples = 48000 * 10;

				auto rand = GenerateRandom(totalSamples);

				TestIdentity(rand);
			}

			void TestIdentity(std::vector<float>& data)
			{
				TrainModel(data.data(), data.data(), data.size(), data.data(), data.data(), data.size());
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

			void TestXOR(size_t delay)
			{
				const size_t totalSamples = 48000 * 10;

				auto rand = GenerateRandom(totalSamples);

				for (int i = 0; i < totalSamples; i++)
					rand[i] = std::copysign(1.0f, rand[i]);

				std::vector<float> target(totalSamples);

				for (size_t i = 0; i < totalSamples; i++)
				{
					if (i < delay)
						target[i] = 0;
					else
						target[i] = -1 * rand[i] * rand[i - delay];
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

				size_t startOffset = 48000 * 1;

				size_t verifyFrames = 48000 * 9;
				size_t verifyOffset = (size_t)numFrames - verifyFrames;

				size_t frameDelay = 4;

				TrainModel(inData + startOffset - frameDelay, targetData + startOffset, (size_t)numFrames - verifyFrames - startOffset - frameDelay, inData + verifyOffset - frameDelay, targetData + verifyOffset, verifyFrames - frameDelay);
			}

		private:
			BackpropModelT<float, 1, 1>& modelBackprop;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchInput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchTarget;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> forwardOutput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> outputGradient;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> layerOutputGradient;
			std::unique_ptr<LossT<T>> lossFunction;
			std::unique_ptr<LossT<T>> lossEvalFunction;
	};
}

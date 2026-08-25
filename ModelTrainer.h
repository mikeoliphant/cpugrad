#pragma once

#include <cmath>
#include <filesystem>
#include "dr_wav.h"

#include "WaveNetBackprop.h"
#include "Optimizer.h"

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
			TrainingData(size_t totalSamples, size_t trainingSize, size_t warmupSize) :
				gen(123)
			{
				int samplesRemaining = (int)totalSamples;
				size_t currentOffset = 0;

				while (samplesRemaining >= (int)(trainingSize + warmupSize))	// skip last uneven batch
				{
					batches.emplace_back(currentOffset, trainingSize + warmupSize);

					currentOffset += trainingSize;
					samplesRemaining -= (int)trainingSize;
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
				modelBackprop(&modelBackprop),
				lossFunction(std::make_unique<MSELossT<T>>()),
				lossEvalFunction(std::make_unique<ESRLossT<T>>()),
				optimizer(std::make_unique<AdamOptimizerT<T>>())
			{
				this->modelBackprop->AddWeightGradients(*optimizer);
			}

			void VerifyModel(const T* input, T* output, const size_t totalSamples)
			{
				size_t receptiveField = modelBackprop->GetReceptiveField();
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

					modelBackprop->Reset();

					modelBackprop->Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

					T* forwardOutputPtr = forwardOutput.GetData();
					std::copy(forwardOutputPtr + receptiveField, forwardOutputPtr + thisBatchSize, output + currentOffset + receptiveField);

					currentOffset += validSampleCount;
					samplesRemaining -= (int)validSampleCount;
				}
			}

			void TrainModel(const T* input, T* target, const size_t trainingSamples, const T* verifyInput, const T* verifyTarget, const size_t verifySamples)
			{
				float learningRate = 0.004f;

				//ApplyHPF(target, totalSamples);

				modelBackprop->RandomizeWeights();

				size_t trainingSize = 8192;
				size_t receptiveField = modelBackprop->GetReceptiveField();

				TrainingData trainingData(trainingSamples, trainingSize, receptiveField);

				size_t miniBatchSize = 16;
				size_t totBatches = trainingData.Batches().size();
				size_t numMiniBatches = (size_t)std::ceil((float)totBatches / (float)miniBatchSize);

				std::cout << "Training " << trainingData.Batches().size() << " batches of size " << trainingSize << " (+" << receptiveField << ")" << std::endl;

				std::vector<T> verifyOutput(verifySamples);

				for (int iter = 0; iter < 20000; ++iter)
				{
					trainingData.ShuffleBatches();

					optimizer->ResetGradients();
					//modelBackprop->ResetGradients();

					size_t currentBatchNum = 0;

					for (size_t currentMiniBatchNum = 0; currentMiniBatchNum < numMiniBatches; currentMiniBatchNum++)
					{
						size_t thisMiniBatchSize = std::min(miniBatchSize, (totBatches - currentBatchNum));

						float lossScale = 1.0f / (float)thisMiniBatchSize;

						//std::cout << lossScale << " " << thisMiniBatchSize << std::endl;

						for (size_t b = 0; b < thisMiniBatchSize; b++, currentBatchNum++)
						{
							size_t thisBatchSize = trainingData.Batches()[currentBatchNum].Size;
							size_t thisBatchStart = trainingData.Batches()[currentBatchNum].Offset;

							float* batchInPtr = batchInput.GetData();
							std::copy(input + thisBatchStart, input + thisBatchStart + thisBatchSize, batchInPtr);

							auto batchTargetPtr = batchTarget.GetData();
							std::copy(target + thisBatchStart, target + thisBatchStart + thisBatchSize, batchTargetPtr);

							forwardOutput.SetZero();

							modelBackprop->Reset();

							modelBackprop->Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

							//ApplyHPF(forwardOutput.GetData() + receptiveField, thisBatchSize - receptiveField);

							lossFunction->ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), thisBatchSize, receptiveField, lossScale);

							//std::cout << "Batch loss: " << lossFunction->GetTotSquared(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), thisBatchSize, receptiveField) / (float)thisBatchSize << std::endl;

							modelBackprop->Backward(batchInput.Slice(thisBatchSize), outputGradient.Slice(thisBatchSize), layerOutputGradient.Slice(thisBatchSize));
						}

						optimizer->ApplyGradients();
						optimizer->ResetGradients();

						//modelBackprop->ApplyGradients(learningRate);
						//modelBackprop->ResetGradients();
					}

					VerifyModel(verifyInput, verifyOutput.data(), verifySamples);

					double err = lossFunction->GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);

					std::cout << "Epoch: " << iter << " " << lossFunction->GetName() << ": " << std::format("{:.10f}", err);
					
					if (lossEvalFunction->GetName() != lossFunction->GetName())
					{
						err = lossEvalFunction->GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);
						std::cout << " " << lossEvalFunction->GetName() << ": " << std::format("{:.10f}", err);
					}

					std::cout << std::endl;
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

			void TrainIdentity(std::vector<float>& data)
			{
				size_t verifySamples = (size_t)(data.size() * .1f);

				TrainModel(data.data(), data.data(), data.size() - verifySamples, data.data() + verifySamples, data.data() + verifySamples, verifySamples);
			}

			void Train(std::pair<std::vector<float>, std::vector<float>>& dataPair)
			{
				size_t numSamples = dataPair.first.size();
				size_t verifySamples = (size_t)(numSamples * .1f);

				TrainModel(dataPair.first.data(), dataPair.second.data(), numSamples - verifySamples, dataPair.first.data() + verifySamples, dataPair.second.data() + verifySamples, verifySamples);
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

				size_t frameDelay = 0;

				//startOffset = 0;
				//verifyFrames = (size_t)(numFrames * 0.1f);

				size_t verifyOffset = (size_t)numFrames - verifyFrames;

				TrainModel(inData + startOffset - frameDelay, targetData + startOffset, (size_t)numFrames - verifyFrames - startOffset - frameDelay, inData + verifyOffset - frameDelay, targetData + verifyOffset, verifyFrames - frameDelay);
			}

		private:
			BackpropModelT<float, 1, 1>* modelBackprop;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchInput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchTarget;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> forwardOutput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> outputGradient;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> layerOutputGradient;
			std::unique_ptr<LossT<T>> lossFunction;
			std::unique_ptr<LossT<T>> lossEvalFunction;
			std::unique_ptr<OptimizerT<T>> optimizer;
	};
}

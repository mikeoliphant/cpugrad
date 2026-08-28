#pragma once

#include <cmath>
#include <chrono>
#include <filesystem>
#include <thread>

#include "dr_wav.h"

#include "WaveNetBackprop.h"
#include "Optimizer.h"

namespace NeuralCpuTrain
{
	using Clock = std::chrono::steady_clock;

	template <typename T>
	class LossT
	{
		public:
			virtual ~LossT() = default;
			virtual std::string& GetName() { return name; }
			virtual void ComputeLoss(const T* output, const T *target, T* outGradient, size_t numSamples, size_t receptiveFieldSize, double scaleFactor) = 0;
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

			void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, size_t receptiveFieldSize, double scaleFactor) override
			{
				double scale = scaleFactor / static_cast<double>(numSamples - receptiveFieldSize);

				for (size_t t = 0; t < numSamples; t++)
				{
					if (t < receptiveFieldSize)
						outGradient[t] = 0;
					else
						outGradient[t] = static_cast<T>(TCONST(2) * (static_cast<double>(output[t]) - static_cast<double>(target[t])) * scale);
				}
			}

			double GetTotSquared(const T* output, const T* target, size_t numSamples, size_t receptiveFieldSize) override
			{
				double tot = 0;

				for (size_t t = receptiveFieldSize; t < numSamples; t++)
				{
					double diff = static_cast<double>(output[t]) - static_cast<double>(target[t]);
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

		void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, size_t receptiveFieldSize, double scaleFactor) override
		{
			double totEnergy = 0;

			for (size_t t = receptiveFieldSize; t < numSamples; t++)
			{
				totEnergy += (target[t] * target[t]);
			}

			totEnergy += epsilon;

			double scale = scaleFactor / totEnergy;

			for (size_t t = 0; t < numSamples; t++)
			{
				if (t < receptiveFieldSize)
					outGradient[t] = 0;
				else
					outGradient[t] = static_cast<T>(TCONST(2) * (static_cast<double>(output[t]) - static_cast<double>(target[t])) * scale);
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

			double tot = 0;

			for (size_t t = 0; t < numSamples; t++)
			{
				if (t >= receptiveFieldSize)
				{
					double diff = static_cast<double>(output[t]) - static_cast<double>(target[t]);
					tot += (diff * diff);
				}
			}

			return (tot / totEnergy) * static_cast<double>(numSamples - receptiveFieldSize);
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

	template <typename T, typename ModelType, typename LossType = MSELossT<T>>
	class TrainerWorkerT;

	template <typename T, typename ModelType, typename LossType = MSELossT<T>, typename LossEvalType = ESRLossT<T>>
	class ModelTrainerT
	{
		public:
			ModelTrainerT() :
				lossFunction(),
				lossEvalFunction(),
				optimizer()
			{
				for (int w = 0; w < 8; w++)
				{
					modelTrainerWorkers.emplace_back(std::make_unique<TrainerWorkerT<T, ModelType>>());
				}
				
				mainWorker = modelTrainerWorkers[0].get();

				this->modelBackprop = mainWorker->GetModel();

				this->modelBackprop->AddWeightGradients(optimizer);
			}
			
			size_t GetReceptiveField()
			{
				return modelBackprop->GetReceptiveField();
			}

			ModelType* GetModel()
			{
				return modelBackprop;
			}

			void VerifyModel(const T* input, T* output, const size_t totalSamples)
			{
				size_t receptiveField = modelBackprop->GetReceptiveField();
				size_t maxForwardSize = MAX_BATCH_SIZE - receptiveField;
				size_t currentOffset = receptiveField;	// ** NOTE - we will have invalid data for the initial receptive field

				while (currentOffset < totalSamples)
				{
					size_t samplesRemaining = totalSamples - currentOffset;

					size_t forwardSize = std::min(maxForwardSize, samplesRemaining);
					size_t thisBatchSize = forwardSize + receptiveField;
					size_t trainingStart = currentOffset - receptiveField;

					T* batchInPtr = batchInput.GetData();
					std::copy(input + trainingStart, input + trainingStart + thisBatchSize, batchInPtr);

					forwardOutput.SetZero();

					modelBackprop->Reset();

					modelBackprop->Forward(batchInput.Slice(thisBatchSize), forwardOutput.Slice(thisBatchSize));

					T* forwardOutputPtr = forwardOutput.GetData();
					std::copy(forwardOutputPtr + receptiveField, forwardOutputPtr + thisBatchSize, output + currentOffset);

					currentOffset += forwardSize;
				}
			}

			void TestBackprop(size_t weightIndex, const T* input, T* target, const size_t numSamples)
			{
				T* weightPtr = optimizer.GetWeightPtr(weightIndex);
				T* dWeightPtr = optimizer.GetDWeightPtr(weightIndex);

				modelBackprop->RandomizeWeights();

				size_t receptiveField = modelBackprop->GetReceptiveField();

				float* batchInPtr = batchInput.GetData();
				std::copy(input, input + numSamples, batchInPtr);

				auto batchTargetPtr = batchTarget.GetData();
				std::copy(target, target + numSamples, batchTargetPtr);

				double delta = 0.001;

				T originalWeight = *weightPtr;

				*weightPtr = originalWeight + (T)delta;

				forwardOutput.SetZero();

				modelBackprop->Reset();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput.Slice(numSamples));

				double upErr = lossFunction.GetTotSquared(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), numSamples, receptiveField) / static_cast<double>(numSamples - receptiveField);

				*weightPtr = originalWeight - (T)delta;

				forwardOutput.SetZero();

				modelBackprop->Reset();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput.Slice(numSamples));

				double downErr = lossFunction.GetTotSquared(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), numSamples, receptiveField) / static_cast<double>(numSamples - receptiveField);

				*weightPtr = originalWeight;

				forwardOutput.SetZero();

				modelBackprop->Reset();
				optimizer.ResetGradients();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput.Slice(numSamples));

				lossFunction.ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), numSamples, receptiveField, 1.0f);

				modelBackprop->Backward(batchInput.Slice(numSamples), outputGradient.Slice(numSamples), layerOutputGradient.Slice(numSamples));

				double numGrad = (upErr - downErr) / (2.0 * delta); 

				double relErr = (*dWeightPtr - numGrad) / std::max({ std::abs((double)*dWeightPtr), std::abs(numGrad), 1e-8 });

				std::cout << "DWeight: " << *dWeightPtr << " NumGrad: " << numGrad << " RelErr: " << relErr << std::endl;
			}

			void TrainModel(const T* input, T* target, const size_t trainingSamples, const T* verifyInput, const T* verifyTarget, const size_t verifySamples)
			{
				float learningRate = 0.004f;
				float learningRateDecay = 0.993f;

				modelBackprop->RandomizeWeights();

				size_t trainingSize = 8192;
				size_t receptiveField = modelBackprop->GetReceptiveField();

				if ((trainingSize + receptiveField) > MAX_BATCH_SIZE)
					throw std::runtime_error("MAX_BATCH_SIZE is too small");

				TrainingData trainingData(trainingSamples, trainingSize, receptiveField);

				size_t miniBatchSize = 16;
				size_t totBatches = trainingData.Batches().size();
				size_t numMiniBatches = (size_t)std::ceil((float)totBatches / (float)miniBatchSize);

				std::cout << "Training " << trainingData.Batches().size() << " batches of size " << trainingSize << " (+" << receptiveField << ")" << std::endl;

				std::vector<T> verifyOutput(verifySamples);

				for (int epoch = 0; epoch < 20000; ++epoch)
				{
					auto epochStart = Clock::now();
					auto forwardDuration = Clock::duration::zero();
					auto backDuration = Clock::duration::zero();

					trainingData.ShuffleBatches();

					optimizer.SetLearningRate(learningRate);
					optimizer.ResetGradients();

					size_t currentBatchNum = 0;

					for (size_t currentMiniBatchNum = 0; currentMiniBatchNum < numMiniBatches; currentMiniBatchNum++)
					{
						std::vector<std::jthread> threads;
						
						size_t thisMiniBatchSize = std::min(miniBatchSize, (totBatches - currentBatchNum));

						float lossScale = 1.0f / (float)thisMiniBatchSize;

						for (auto& worker : modelTrainerWorkers)
						{
							worker->Reset();
						}

						for (size_t w = 1; w < modelTrainerWorkers.size(); w++)
						{
							modelTrainerWorkers[w]->CopyWeightsFrom(optimizer);
						}

						for (size_t b = 0; b < thisMiniBatchSize; b++, currentBatchNum++)
						{
							modelTrainerWorkers[b % modelTrainerWorkers.size()]->AddBatch(trainingData.Batches()[currentBatchNum]);
						}

						for (auto& worker : modelTrainerWorkers)
						{
							threads.emplace_back(
								&TrainerWorkerT<T, ModelType>::TrainBatches,
								worker.get(),
								std::ref(input),
								std::ref(target),
								lossScale
							);
							
							//worker->TrainBatches(input, target, lossScale);
						}

						threads.clear();

						for (size_t w = 1; w < std::min(modelTrainerWorkers.size(), thisMiniBatchSize); w++)
						{
							modelTrainerWorkers[w]->AddDWeightsTo(optimizer);
						}

						optimizer.ApplyGradients();
						optimizer.ResetGradients();
					}

					VerifyModel(verifyInput, verifyOutput.data(), verifySamples);

					double err = lossFunction.GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);

					auto epochEnd = Clock::now();

					double epochTime = std::chrono::duration<double>(epochEnd - epochStart).count();

					std::cout << "Epoch: " << epoch << " " << std::format("{:.2f}", epochTime) << "s LR: " << learningRate << " " << lossFunction.GetName() << ": " << std::format("{:.10f}", err);
					
					if (lossEvalFunction.GetName() != lossFunction.GetName())
					{
						err = lossEvalFunction.GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);
						std::cout << " " << lossEvalFunction.GetName() << ": " << std::format("{:.10f}", err);
					}

					std::cout << std::endl;

					if (epoch == 0)
					{
						double forwardTime = std::chrono::duration<double>(forwardDuration).count();
						double backTime = std::chrono::duration<double>(backDuration).count();

						std::cout << "Forward: " << forwardTime << " Back: " << backTime << " Other: " << (epochTime - forwardTime - backTime) << std::endl;
					}

					learningRate *= learningRateDecay;
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
			std::vector<std::unique_ptr<TrainerWorkerT<T, ModelType>>> modelTrainerWorkers;
			TrainerWorkerT<T, ModelType>* mainWorker;
			ModelType* modelBackprop;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchInput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchTarget;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> forwardOutput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> outputGradient;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> layerOutputGradient;
			LossType lossFunction;
			LossEvalType lossEvalFunction;
			AdamOptimizerT<T> optimizer;
	};

	template <typename T, typename ModelType, typename LossType>
	class TrainerWorkerT
	{
		public:
			TrainerWorkerT() :
				modelBackprop(std::make_unique<ModelType>()),
				lossFunction(),
				optimizer()
			{
				this->modelBackprop->AddWeightGradients(optimizer);
			}

			ModelType* GetModel()
			{
				return modelBackprop.get();
			}

			void CopyWeightsFrom(AdamOptimizerT<T>& optimizer)
			{
				this->optimizer.CopyWeightsFrom(optimizer);
			}

			void AddDWeightsTo(AdamOptimizerT<T>& optimizer)
			{
				this->optimizer.AddDWeightsTo(optimizer);
			}

			void Reset()
			{
				batches.clear();
				optimizer.ResetGradients();
			}

			void AddBatch(const TrainingDataBatch& batch)
			{
				batches.push_back(batch);
			}

			void TrainBatches(const T* input, const T* target, double lossScale)
			{				
				for (auto& b : batches)
				{
					size_t numSamples = b.Size;

					size_t receptiveField = modelBackprop->GetReceptiveField();

					float* batchInPtr = batchInput.GetData();
					std::copy(input + b.Offset, input + b.Offset + numSamples, batchInPtr);

					auto batchTargetPtr = batchTarget.GetData();
					std::copy(target + b.Offset, target + b.Offset + numSamples, batchTargetPtr);

					forwardOutput.SetZero();

					modelBackprop->Reset();

					//auto forwardStart = Clock::now();
					modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput.Slice(numSamples));
					//forwardDuration += (Clock::now() - forwardStart);

					lossFunction.ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), numSamples, receptiveField, lossScale);

					//std::cout << "Batch loss: " << lossFunction.GetTotSquared(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), numSamples, receptiveField) / (float)(numSamples - receptiveField) << std::endl;

					//auto backStart = Clock::now();
					modelBackprop->Backward(batchInput.Slice(numSamples), outputGradient.Slice(numSamples), layerOutputGradient.Slice(numSamples));
					//backDuration += (Clock::now() - backStart);
				}
			}

		private:
			std::vector<TrainingDataBatch> batches;
			std::unique_ptr<ModelType> modelBackprop;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchInput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> batchTarget;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> forwardOutput;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> outputGradient;
			ChannelBuffer<float, 1, MAX_BATCH_SIZE> layerOutputGradient;
			LossType lossFunction;
			AdamOptimizerT<T> optimizer;
	};
}

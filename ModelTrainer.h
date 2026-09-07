#pragma once

#include <cmath>
#include <chrono>
#include <filesystem>
#include <thread>

#include "dr_wav.h"

#include "WaveNetBackprop.h"
#include "BatchBuffer.h"
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
				lossEvalFunction()
			{
				for (int w = 0; w < 8; w++)
				{
					modelTrainerWorkers.emplace_back(std::make_unique<TrainerWorkerT<T, ModelType, LossType>>());
				}
				
				mainWorker = modelTrainerWorkers[0].get();

				this->modelBackprop = mainWorker->GetModel();
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
				mainWorker->VerifyModel(input, output, totalSamples);
			}

			void TestBackprop(size_t weightIndex, const T* input, T* target, const size_t numSamples)
			{
				mainWorker->TestBackprop(weightIndex, input, target, numSamples);
			}

			void TrainModel(const T* input, T* target, const size_t trainingSamples, const T* verifyInput, const T* verifyTarget, const size_t verifySamples)
			{
				float learningRate = 0.004f;
				float learningRateDecay = 0.993f;

				modelBackprop->RandomizeWeights();

				size_t trainingSize = 8192;
				size_t receptiveField = modelBackprop->GetReceptiveField();

				TrainingData trainingData(trainingSamples, trainingSize, receptiveField);

				size_t miniBatchSize = 16;
				size_t totBatches = trainingData.Batches().size();
				size_t numMiniBatches = (size_t)std::ceil((float)totBatches / (float)miniBatchSize);

				std::cout << "Training " << trainingData.Batches().size() << " batches of size " << trainingSize << " (+" << receptiveField << ")" << std::endl;

				std::vector<T> verifyOutput(verifySamples);

				for (int epoch = 0; epoch < 20000; ++epoch)
				{
					auto epochStart = Clock::now();
					auto trainDuration = Clock::duration::zero();

					auto threadTotalDuration = Clock::duration::zero();
					auto threadForwardDuration = Clock::duration::zero();
					auto threadBackDuration = Clock::duration::zero();

					trainingData.ShuffleBatches();

					mainWorker->GetOptimizer().SetLearningRate(learningRate);
					mainWorker->GetOptimizer().ResetGradients();

					size_t currentBatchNum = 0;

					for (auto& worker : modelTrainerWorkers)
					{
						worker->ResetDurations();
					}

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
							modelTrainerWorkers[w]->CopyWeightsFrom(mainWorker->GetOptimizer());
						}

						for (size_t b = 0; b < thisMiniBatchSize; b++, currentBatchNum++)
						{
							modelTrainerWorkers[b % modelTrainerWorkers.size()]->AddBatch(trainingData.Batches()[currentBatchNum]);
						}

						auto trainStart = Clock::now();

						for (auto& worker : modelTrainerWorkers)
						{
							threads.emplace_back(
								&TrainerWorkerT<T, ModelType>::TrainBatches,
								worker.get(),
								std::ref(input),
								std::ref(target),
								lossScale
							);							
						}

						threads.clear();

						trainDuration += (Clock::now() - trainStart);

						for (size_t w = 1; w < std::min(modelTrainerWorkers.size(), thisMiniBatchSize); w++)
						{
							modelTrainerWorkers[w]->AddDWeightsTo(mainWorker->GetOptimizer());
						}

						mainWorker->GetOptimizer().ApplyGradients();
						mainWorker->GetOptimizer().ResetGradients();
					}

					for (auto& worker : modelTrainerWorkers)
					{
						worker->AddDurations(threadTotalDuration, threadForwardDuration, threadBackDuration);
					}

					auto verifyStart = Clock::now();
					VerifyModel(verifyInput, verifyOutput.data(), verifySamples);
					double verifyTime = std::chrono::duration<double>(Clock::now() - verifyStart).count();

					double err = lossFunction.GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);

					double epochTime = std::chrono::duration<double>(Clock::now() - epochStart).count();

					if (epoch == 0)
					{
						double trainTime = std::chrono::duration<double>(trainDuration).count();

						std::cout << "Train: " << trainTime << " Verify: " << verifyTime << " Other: " << (epochTime - trainTime - verifyTime) << std::endl;

						double threadTotalTime = std::chrono::duration<double>(threadTotalDuration).count();
						double threadForwardTime = std::chrono::duration<double>(threadForwardDuration).count();
						double threadBackTime = std::chrono::duration<double>(threadBackDuration).count();

						std::cout << "Thread - Forward: " << threadForwardTime << " Back: " << threadBackTime << " Other: " << (threadTotalTime - threadForwardTime - threadBackTime) << std::endl;
					}

					std::cout << "Epoch: " << epoch << " " << std::format("{:.2f}", epochTime) << "s LR: " << std::format("{:.5f}", learningRate) << " " << lossFunction.GetName() << ": " << std::format("{:.8f}", err);
					
					if (lossEvalFunction.GetName() != lossFunction.GetName())
					{
						err = lossEvalFunction.GetTotSquared(verifyOutput.data(), verifyTarget, verifySamples, receptiveField) / static_cast<double>(verifySamples - receptiveField);
						std::cout << " " << lossEvalFunction.GetName() << ": " << std::format("{:.8f}", err);
					}

					std::cout << std::endl;

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

			void TrainWav(const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath)
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
			LossType lossFunction;
			LossEvalType lossEvalFunction;
	};

	template <typename T, typename ModelType, typename LossType>
	class TrainerWorkerT : public TrainingContextT<T>
	{
		public:
			TrainerWorkerT() :
				modelBackprop(std::make_unique<ModelType>()),
				lossFunction(),
				optimizer(),
				bufferArena()
			{
				this->modelBackprop->SetTrainingContext(this);
			}

			ModelType* GetModel()
			{
				return modelBackprop.get();
			}

			AdamOptimizerT<T>& GetOptimizer()
			{
				return optimizer;
			}

			BatchBufferArenaT<T>& GetBufferArena() override
			{
				return bufferArena;
			}

			void CopyWeightsFrom(AdamOptimizerT<T>& optimizer)
			{
				this->optimizer.CopyWeightsFrom(optimizer);
			}

			void AddDWeightsTo(AdamOptimizerT<T>& optimizer)
			{
				this->optimizer.AddDWeightsTo(optimizer);
			}

			void AddWeightGradient(T* weights, T* dWeights, size_t numWeights) override
			{
				optimizer.AddWeightGradient(weights, dWeights, numWeights);
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

			void ResetDurations()
			{
				totalDuration = Clock::duration::zero();
				forwardDuration = Clock::duration::zero();
				backDuration = Clock::duration::zero();
			}

			void AddDurations(Clock::duration& total, Clock::duration& forward, Clock::duration& back)
			{
				total += totalDuration;
				forward += forwardDuration;
				back += backDuration;
			}

			void TrainBatches(const T* input, const T* target, double lossScale)
			{	
				auto totalStart = Clock::now();

				for (auto& b : batches)
				{
					size_t numSamples = b.Size;

					size_t receptiveField = modelBackprop->GetReceptiveField();
					size_t outputSize = numSamples - receptiveField;

					if (batchInput.GetNumCols() == 0)
					{
						batchInput = bufferArena.template GetBuffer<1>(numSamples);
					}

					float* batchInPtr = batchInput.GetData();
					std::copy(input + b.Offset, input + b.Offset + numSamples, batchInPtr);

					if (batchTarget.GetNumCols() == 0)
					{
						batchTarget = bufferArena.template GetBuffer<1>(outputSize);
					}

					auto batchTargetPtr = batchTarget.GetData();
					std::copy(target + b.Offset + receptiveField, target + b.Offset + numSamples, batchTargetPtr);

					if (forwardOutput.GetNumCols() == 0)
					{
						forwardOutput = bufferArena.template GetBuffer<1>(outputSize);
					}

					forwardOutput.SetZero();

					auto forwardStart = Clock::now();
					modelBackprop->Forward(batchInput, forwardOutput);
					forwardDuration += (Clock::now() - forwardStart);

					if (outputGradient.GetNumCols() == 0)
					{
						outputGradient = bufferArena.template GetBuffer<1>(outputSize);
					}

					lossFunction.ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), outputSize, 0, lossScale);

					//std::cout << "Batch loss: " << lossFunction.GetTotSquared(forwardSlice.GetDataConst(), batchTargetSlice.GetDataConst(), outputSize, 0) / (float)outputSize << std::endl;

					// Really shouldn't need this
					auto layerOutputGradient = bufferArena.template GetScratchBuffer<1>(numSamples);

					auto backStart = Clock::now();
					modelBackprop->Backward(batchInput, outputGradient, layerOutputGradient);
					backDuration += (Clock::now() - backStart);

					bufferArena.FreeScratchBuffer(layerOutputGradient);

					bufferArena.ReleaseScratch();
				}

				totalDuration += (Clock::now() - totalStart);
			}

			void VerifyModel(const T* input, T* output, const size_t totalSamples)
			{
				size_t outputSize = 8192;
				size_t receptiveField = modelBackprop->GetReceptiveField();
				size_t batchSize = outputSize + receptiveField;
				size_t currentOffset = receptiveField;	// ** NOTE - we will have invalid data for the initial receptive field

				if (forwardOutput.GetNumCols() == 0)
				{
					forwardOutput = bufferArena.template GetBuffer<1>(outputSize);
				}

				if (batchInput.GetNumCols() == 0)
				{
					batchInput = bufferArena.template GetBuffer<1>(batchSize);
				}

				while (currentOffset < totalSamples)
				{
					size_t samplesRemaining = totalSamples - currentOffset;

					size_t trainingStart = currentOffset - receptiveField;

					T* batchInPtr = batchInput.GetData();

					std::copy(input + trainingStart, input + trainingStart + receptiveField + std::min(outputSize, samplesRemaining), batchInPtr);

					forwardOutput.SetZero();

					modelBackprop->Forward(batchInput, forwardOutput);

					size_t toCopy = std::min(outputSize, samplesRemaining);
					T* forwardOutputPtr = forwardOutput.GetData();
					std::copy(forwardOutputPtr, forwardOutputPtr + toCopy, output + currentOffset);

					currentOffset += outputSize;
				}
			}

			void TestBackprop(size_t weightIndex, const T* input, T* target, const size_t numSamples)
			{
				T* weightPtr = optimizer.GetWeightPtr(weightIndex);
				T* dWeightPtr = optimizer.GetDWeightPtr(weightIndex);

				modelBackprop->RandomizeWeights();

				size_t receptiveField = modelBackprop->GetReceptiveField();
				size_t outputSize = numSamples - receptiveField;

				if (forwardOutput.GetNumCols() == 0)
				{
					forwardOutput = bufferArena.template GetBuffer<1>(outputSize);
				}

				if (batchInput.GetNumCols() == 0)
				{
					batchInput = bufferArena.template GetBuffer<1>(numSamples);
				}

				if (batchTarget.GetNumCols() == 0)
				{
					batchTarget = bufferArena.template GetBuffer<1>(outputSize);
				}

				float* batchInPtr = batchInput.GetData();
				std::copy(input, input + numSamples, batchInPtr);

				auto batchTargetPtr = batchTarget.GetData();
				std::copy(target, target + numSamples, batchTargetPtr);

				double delta = 0.001;

				T originalWeight = *weightPtr;

				*weightPtr = originalWeight + (T)delta;

				forwardOutput.SetZero();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput);

				double upErr = lossFunction.GetTotSquared(forwardOutput.GetDataConst(), batchTarget.Slice(receptiveField, outputSize).GetDataConst(), outputSize, 0) / static_cast<double>(outputSize);

				*weightPtr = originalWeight - (T)delta;

				forwardOutput.SetZero();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput.Slice(outputSize));

				double downErr = lossFunction.GetTotSquared(forwardOutput.GetDataConst(), batchTarget.Slice(receptiveField, outputSize).GetDataConst(), outputSize, 0) / static_cast<double>(outputSize);

				*weightPtr = originalWeight;

				forwardOutput.SetZero();

				optimizer.ResetGradients();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput);

				if (outputGradient.GetNumCols() == 0)
				{
					outputGradient = bufferArena.template GetBuffer<1>(outputSize);
				}

				lossFunction.ComputeLoss(forwardOutput.GetDataConst(), batchTarget.Slice(receptiveField, outputSize).GetDataConst(), outputGradient.GetData(), outputSize, 0, 1.0f);

				modelBackprop->Backward(batchInput.Slice(numSamples), outputGradient.Slice(outputSize), forwardOutput.Slice(numSamples));

				double numGrad = (upErr - downErr) / (2.0 * delta);

				double relErr = (*dWeightPtr - numGrad) / std::max({ std::abs((double)*dWeightPtr), std::abs(numGrad), 1e-8 });

				std::cout << "DWeight: " << *dWeightPtr << " NumGrad: " << numGrad << " RelErr: " << relErr << std::endl;
			}


		private:
			std::vector<TrainingDataBatch> batches;
			std::unique_ptr<ModelType> modelBackprop;
			LossType lossFunction;
			AdamOptimizerT<T> optimizer;
			BatchBufferArenaT<T> bufferArena;
			ChannelBufferDynamic<float, 1> batchInput;
			ChannelBufferDynamic<float, 1> batchTarget;
			ChannelBufferDynamic<float, 1> forwardOutput;
			ChannelBufferDynamic<float, 1> outputGradient;
			Clock::duration forwardDuration = Clock::duration::zero();
			Clock::duration backDuration = Clock::duration::zero();
			Clock::duration totalDuration = Clock::duration::zero();

	};
}

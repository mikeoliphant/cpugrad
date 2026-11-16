#pragma once

#include <cmath>
#include <chrono>
#include <filesystem>
#include <functional>
#include <thread>
#include <barrier>
#include <atomic>
#include <random>

#include "BatchBuffer.h"
#include "Platform.h"
#include "Optimizer.h"
#include "Loss.h"
#include "MrStft.h"
#include "LossSpec.h"

namespace cpugrad
{
	using Clock = std::chrono::steady_clock;

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

	template <typename T, typename ModelType>
	class TrainerWorkerT;

	// The training loss is set at runtime with SetLoss() (MSE by default). LossEvalType is the metric used to pick the best checkpoint.
	template <typename T, typename ModelType, typename LossEvalType = ESRLossT<T>>
	class ModelTrainerT
	{
		public:
			ModelTrainerT(size_t numThreads = 0)
				: ModelTrainerT(numThreads, std::mt19937(std::random_device{}()))
			{
			}

			ModelTrainerT(size_t numThreads, const std::mt19937& rand) :
				rand(rand),
				lossEvalFunction()
			{
				DenormalManager::DisableDenormals();

				auto coreIds = ThreadAffinityManager::GetPhysicalCores();

				if (numThreads == 0)
				{
					numThreads = coreIds.size();
				}

				for (size_t w = 0; w < numThreads; w++)
				{
					modelTrainerWorkers.emplace_back(std::make_unique<TrainerWorkerT<T, ModelType>>(coreIds[w], trainingSize));
				}

				mainWorker = modelTrainerWorkers[0].get();

				this->modelBackprop = mainWorker->GetModel();

				bestWeights.resize(modelBackprop->GetNumWeights());

				SetLoss(LossSpec{ LossTerm{ "mse", 1.0, {} } });
			}

			// Each worker gets its own loss instance, since losses keep per-call scratch state
			void SetLoss(const LossSpec& spec)
			{
				lossSpec = spec;
				lossFunction = CreateLoss<T>(spec);

				for (auto& worker : modelTrainerWorkers)
				{
					worker->SetLoss(CreateLoss<T>(spec));
				}
			}

			const LossSpec& GetLossSpec() const
			{
				return lossSpec;
			}

			size_t GetReceptiveField()
			{
				return modelBackprop->GetReceptiveField();
			}

			ModelType& GetModel()
			{
				return *modelBackprop;
			}

			void SetMaxEpochs(size_t maxEpochs)
			{
				this->maxEpochs = maxEpochs;
			}

			void SetEpochCallback(std::function<bool(size_t epoch, double loss)> callback)
			{
				this->epochCallback = std::move(callback);
			}

			std::vector<float>& GetBestWeights()
			{
				return bestWeights;
			}

			double GetBestLoss()
			{
				return minLoss;
			}

			size_t GetMaxEpochs() const { return maxEpochs; }
			size_t GetEpochsCompleted() const { return epochsCompleted; }
			size_t GetBestEpoch() const { return bestEpoch; }	// 1-based, 0 if no epoch has completed
			size_t GetNumThreads() const { return modelTrainerWorkers.size(); }
			float GetLearningRate() const { return initialLearningRate; }
			float GetLearningRateDecay() const { return learningRateDecay; }
			size_t GetTrainingSize() const { return trainingSize; }
			size_t GetMiniBatchSize() const { return miniBatchSize; }

			void VerifyModel(const T* input, T* output, const size_t totalSamples)
			{
				mainWorker->VerifyModel(input, output, totalSamples);
			}

			void TestBackprop(size_t weightIndex, const T* input, T* target, const size_t numSamples)
			{
				mainWorker->TestBackprop(weightIndex, input, target, numSamples);
			}

			void TrainModel(const T* input, T* target, const size_t trainingSamples, const T* verifyInput, T* verifyTarget, const size_t verifySamples)
			{
				float learningRate = initialLearningRate;

				epochsCompleted = 0;
				bestEpoch = 0;

				modelBackprop->RandomizeWeights(rand);

				size_t receptiveField = modelBackprop->GetReceptiveField();

				TrainingData trainingData(trainingSamples, trainingSize, receptiveField);

				size_t totBatches = trainingData.Batches().size();
				size_t numMiniBatches = (size_t)std::ceil((float)totBatches / (float)miniBatchSize);

				std::cout << "Training " << trainingData.Batches().size() << " batches of size " << trainingSize << " (+" << receptiveField << ") using " << modelTrainerWorkers.size() << " threads" << std::endl;

				std::vector<T> verifyOutput(verifySamples);

				minLoss = std::numeric_limits<double>::max();

				double lossScale = 1.0f;

				// Each worker has a thread that runs one task per round. runOnWorkers() sets the task and runs a round
				// on all workers, returning when they have all finished.
				const size_t numWorkers = modelTrainerWorkers.size();

				std::function<void(Worker& worker, size_t workerIndex)> workerTask;

				std::barrier startBarrier(numWorkers + 1);
				std::barrier finishBarrier(numWorkers + 1);

				auto runOnWorkers = [&](std::function<void(Worker&, size_t)> task)
				{
					workerTask = std::move(task);

					startBarrier.arrive_and_wait();
					finishBarrier.arrive_and_wait();
				};

				std::vector<std::jthread> threads;

				threads.reserve(numWorkers);

				for (size_t w = 0; w < numWorkers; w++)
				{
					threads.emplace_back([&, w](std::stop_token stop)
					{
						Worker& worker = *modelTrainerWorkers[w];

						worker.PinThread();

						while (true)
						{
							startBarrier.arrive_and_wait();

							if (stop.stop_requested())
								break;

							workerTask(worker, w);

							finishBarrier.arrive_and_wait();
						}
					});
				}

				for (size_t epoch = 0; epoch < maxEpochs; epoch++)
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
						size_t thisMiniBatchSize = std::min(miniBatchSize, (totBatches - currentBatchNum));

						lossScale = 1.0f / (float)thisMiniBatchSize;

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

						runOnWorkers([&](Worker& worker, size_t) { worker.TrainBatches(input, target, lossScale); });

						trainDuration += (Clock::now() - trainStart);

						for (size_t w = 1; w < std::min(modelTrainerWorkers.size(), thisMiniBatchSize); w++)
						{
							modelTrainerWorkers[w]->AddDWeightsTo(mainWorker->GetOptimizer());
						}

						mainWorker->GetOptimizer().ApplyGradients(1.0f);
						mainWorker->GetOptimizer().ResetGradients();
					}

					for (auto& worker : modelTrainerWorkers)
					{
						worker->AddDurations(threadTotalDuration, threadForwardDuration, threadBackDuration);
					}

					auto verifyStart = Clock::now();

					// Validate on all workers, each taking every numWorkers'th chunk. The other workers last got the main
					// weights before the epoch's final update, so refresh them first.
					for (size_t w = 1; w < numWorkers; w++)
					{
						modelTrainerWorkers[w]->CopyWeightsFrom(mainWorker->GetOptimizer());
					}

					runOnWorkers([&](Worker& worker, size_t w) { worker.VerifyModel(verifyInput, verifyOutput.data(), verifySamples, w, numWorkers); });

					double verifyTime = std::chrono::duration<double>(Clock::now() - verifyStart).count();

					// The training loss on the validation data, for display (MR-STFT over 9 s takes a while, so split it across workers)
					double trainingLoss = MeanLossOnWorkers(runOnWorkers, verifyOutput.data() + receptiveField, verifyTarget + receptiveField, verifySamples - receptiveField);

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

					std::cout << "Epoch " << std::format("{:3d}", (epoch + 1)) << ": " << std::format("{:.2f}", epochTime) << "s LR: " << std::format("{:.5f}", learningRate) << " " << lossFunction->GetName() << ": " << std::format("{:.8f}", trainingLoss);

					// The best epoch is picked by the evaluation loss, computed on one thread so the choice doesn't depend on the thread count
					double err = lossEvalFunction.GetMeanLoss(verifyOutput.data() + receptiveField, verifyTarget + receptiveField, verifySamples - receptiveField);

					if (lossEvalFunction.GetName() != lossFunction->GetName())
					{
						std::cout << " " << lossEvalFunction.GetName() << ": " << std::format("{:.8f}", err);
					}

					if (err < minLoss)
					{
						// save weights
						auto it = bestWeights.begin();

						modelBackprop->GetWeights(it);

						std::cout << " *";

						minLoss = err;
						bestEpoch = epoch + 1;
					}

					std::cout << std::endl;

					epochsCompleted = epoch + 1;

					if (epochCallback)
					{
						if (!epochCallback(epoch, err))
							break;
					}

					learningRate *= learningRateDecay;
				}

				// Release the threads from their start barrier with the stop flag set, so they exit
				for (auto& t : threads)
				{
					t.request_stop();
				}

				startBarrier.arrive_and_wait();
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

		private:
			using Worker = TrainerWorkerT<T, ModelType>;

			// Mean training loss, split across the workers (each with its own loss instance) if the loss supports it
			template <typename RunOnWorkers>
			double MeanLossOnWorkers(RunOnWorkers& runOnWorkers, const T* output, const T* target, size_t numSamples)
			{
				const size_t numStats = lossFunction->GetNumMeanStatistics();

				if (numStats == 0)
					return lossFunction->GetMeanLoss(output, target, numSamples);

				const size_t numWorkers = modelTrainerWorkers.size();

				std::vector<std::vector<double>> workerStats(numWorkers, std::vector<double>(numStats, 0.0));

				runOnWorkers([&](Worker& worker, size_t w) { worker.GetLoss().AccumulateMeanStatistics(output, target, numSamples, w, numWorkers, workerStats[w].data()); });

				std::vector<double> stats(numStats, 0.0);

				for (auto& partStats : workerStats)
				{
					for (size_t s = 0; s < numStats; s++)
						stats[s] += partStats[s];
				}

				return lossFunction->MeanLossFromStatistics(stats.data(), numSamples);
			}

			size_t trainingSize = 8192;
			size_t miniBatchSize = 16;
			float initialLearningRate = 0.004f;
			float learningRateDecay = 0.993f;
			size_t epochsCompleted = 0;
			size_t bestEpoch = 0;
			std::mt19937 rand;
			std::vector<std::unique_ptr<Worker>> modelTrainerWorkers;
			Worker* mainWorker;
			ModelType* modelBackprop;
			LossSpec lossSpec;
			std::unique_ptr<LossT<T>> lossFunction;	// for reporting the training loss on the validation data
			LossEvalType lossEvalFunction;
			size_t maxEpochs = 1000;
			std::vector<float> bestWeights;
			double minLoss;
			std::function<bool(size_t epoch, double loss)> epochCallback;
	};

	template <typename T, typename ModelType>
	class TrainerWorkerT : public TrainingContextT<T>
	{
		public:
			TrainerWorkerT(uint32_t coreID, size_t trainingSize) :
				coreID(coreID),
				modelBackprop(std::make_unique<ModelType>()),
				optimizer(),
				bufferArena(modelBackprop->GetMaxScratchBufferSize(trainingSize))
			{
				this->modelBackprop->SetTrainingContext(static_cast<TrainingContextT<T>*>(this));
			}

			void SetLoss(std::unique_ptr<LossT<T>> loss)
			{
				lossFunction = std::move(loss);
			}

			LossT<T>& GetLoss()
			{
				return *lossFunction;
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
	
			void ApplyHPF(T* data, size_t numSamples)
			{
				constexpr T coefficient = T(0.85);

				for (size_t n = numSamples - 1; n > 0; --n)
				{
					data[n] = data[n] - (coefficient * data[n - 1]);
				}

				data[0] = data[0] * (1.0f - coefficient);
			}

			void PinThread()
			{
				ThreadAffinityManager::PinCurrentThread(coreID);
				//ThreadAffinityManager::SetHighPerformancePriority();
			}

			// Forward, loss and backward for the batches added since the last Reset(), accumulating weight gradients
			void TrainBatches(const T* input, const T* target, double lossScale)
			{
				auto totalStart = Clock::now();

				for (auto& b : batches)
				{
					size_t numSamples = b.Size;

					size_t receptiveField = modelBackprop->GetReceptiveField();
					size_t outputSize = numSamples - receptiveField;

					bufferArena.template GetBuffer<1>(batchInput, numSamples);

					float* batchInPtr = batchInput.GetData();
					std::copy(input + b.Offset, input + b.Offset + numSamples, batchInPtr);

					bufferArena.template GetBuffer<1>(forwardOutput, outputSize);

					forwardOutput.SetZero();

					auto forwardStart = Clock::now();
					modelBackprop->Forward(batchInput, forwardOutput);
					forwardDuration += (Clock::now() - forwardStart);

					auto outputGradient = bufferArena.template GetScratchBuffer<1>(outputSize);
					outputGradient.SetZero();

					auto batchTarget = bufferArena.template GetScratchBuffer<1>(outputSize);

					auto batchTargetPtr = batchTarget.GetData();
					std::copy(target + b.Offset + receptiveField, target + b.Offset + numSamples, batchTargetPtr);

					//ApplyHPF(batchTargetPtr, outputSize);
					//ApplyHPF(forwardOutput.GetData(), outputSize);

					lossFunction->ComputeLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputGradient.GetData(), outputSize, lossScale);

					//std::cout << "Batch loss: " << lossFunction->GetMeanLoss(forwardOutput.GetDataConst(), batchTarget.GetDataConst(), outputSize) / (float)outputSize << std::endl;

					bufferArena.FreeScratchBuffer(batchTarget);

					// Really shouldn't need this
					auto layerOutputGradient = bufferArena.template GetScratchBuffer<1>(numSamples);

					auto backStart = Clock::now();
					modelBackprop->Backward(batchInput, outputGradient, layerOutputGradient);
					backDuration += (Clock::now() - backStart);

					bufferArena.FreeScratchBuffer(layerOutputGradient);
					bufferArena.FreeScratchBuffer(outputGradient);
				}

				totalDuration += (Clock::now() - totalStart);
			}

			// Model output, computed in chunks of VerifyChunkSize output samples. firstChunk and chunkStride let several
			// workers split the work: each computes every chunkStride'th chunk, starting at firstChunk.
			void VerifyModel(const T* input, T* output, const size_t totalSamples, size_t firstChunk = 0, size_t chunkStride = 1)
			{
				size_t outputSize = VerifyChunkSize;
				size_t receptiveField = modelBackprop->GetReceptiveField();
				size_t batchSize = outputSize + receptiveField;
				size_t currentOffset = receptiveField + (firstChunk * outputSize);	// ** NOTE - we will have invalid data for the initial receptive field

				bufferArena.template GetBuffer<1>(forwardOutput, outputSize);
				bufferArena.template GetBuffer<1>(batchInput, batchSize);

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

					currentOffset += outputSize * chunkStride;
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

				float* batchInPtr = batchInput.GetData();
				std::copy(input, input + numSamples, batchInPtr);

				auto batchTarget = bufferArena.template GetScratchBuffer<1>(outputSize);
				auto batchTargetPtr = batchTarget.GetData();
				std::copy(target, target + numSamples, batchTargetPtr);

				double delta = 0.001;

				T originalWeight = *weightPtr;

				*weightPtr = originalWeight + (T)delta;

				forwardOutput.SetZero();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput);

				double upErr = lossFunction->GetMeanLoss(forwardOutput.GetDataConst(), batchTarget.Slice(receptiveField, outputSize).GetDataConst(), outputSize);

				*weightPtr = originalWeight - (T)delta;

				forwardOutput.SetZero();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput.Slice(outputSize));

				double downErr = lossFunction->GetMeanLoss(forwardOutput.GetDataConst(), batchTarget.Slice(receptiveField, outputSize).GetDataConst(), outputSize);

				*weightPtr = originalWeight;

				forwardOutput.SetZero();

				optimizer.ResetGradients();

				modelBackprop->Forward(batchInput.Slice(numSamples), forwardOutput);

				auto outputGradient = bufferArena.template GetScratchBuffer<1>(outputSize);
				outputGradient.SetZero();

				lossFunction->ComputeLoss(forwardOutput.GetDataConst(), batchTarget.Slice(receptiveField, outputSize).GetDataConst(), outputGradient.GetData(), outputSize, 1.0f);

				modelBackprop->Backward(batchInput.Slice(numSamples), outputGradient.Slice(outputSize), forwardOutput.Slice(numSamples));

				bufferArena.FreeScratchBuffer(outputGradient);

				bufferArena.FreeScratchBuffer(batchTarget);

				double numGrad = (upErr - downErr) / (2.0 * delta);

				double relErr = (*dWeightPtr - numGrad) / std::max({ std::abs((double)*dWeightPtr), std::abs(numGrad), 1e-8 });

				std::cout << "DWeight: " << *dWeightPtr << " NumGrad: " << numGrad << " RelErr: " << relErr << std::endl;
			}


		private:
			static constexpr size_t VerifyChunkSize = 8192;

			uint32_t coreID;
			std::vector<TrainingDataBatch> batches;
			std::unique_ptr<ModelType> modelBackprop;
			std::unique_ptr<LossT<T>> lossFunction;
			AdamOptimizerT<T> optimizer;
			BatchBufferArenaT<T> bufferArena;
			ChannelBufferDynamic<float, 1> batchInput;
			ChannelBufferDynamic<float, 1> forwardOutput;
			Clock::duration forwardDuration = Clock::duration::zero();
			Clock::duration backDuration = Clock::duration::zero();
			Clock::duration totalDuration = Clock::duration::zero();

	};
}

#pragma once

#include <array>
#include <random>
#include <vector>
#include <format>
#include "ChannelBuffer.h"
#include "MatMul.h"
#include "Optimizer.h"

#define MATMUL(output, input, mult, numSamples) \
		if constexpr (MatMul<T, InChannels, OutChannels>::HasKernel()) \
		{ \
			const T* inputPtr = input.GetDataConst(); \
			const T* multPtr = mult.GetDataConst(); \
			\
			T* outputPtr = output.GetData(); \
			\
			MatMul<T, InChannels, OutChannels>::MultiplyInitZero(inputPtr, outputPtr, multPtr, numSamples); \
		} \
		else \
		{ \
			auto outBlock = output.GetEigenMap(); \
			\
			outBlock.noalias() = mult.GetEigenMapConst() * input.GetEigenMapConst(); \
		}

#define MATMUL_BIAS(output, input, mult, bias, numSamples) \
		if constexpr (MatMul<T, InChannels, OutChannels>::HasKernel()) \
		{ \
			const T* inputPtr = input.GetDataConst(); \
			const T* multPtr = mult.GetDataConst(); \
			const T* biasPtr = bias.data(); \
			\
			T* outputPtr = output.GetData(); \
			\
			MatMul<T, InChannels, OutChannels>::MultiplyInitColwise(inputPtr, outputPtr, multPtr, biasPtr, numSamples); \
		} \
		else \
		{ \
			auto outBlock = output.GetEigenMap(); \
			\
			outBlock.noalias() = (mult.GetEigenMapConst() * input.GetEigenMapConst()).colwise() + bias; \
		}

#define MATMUL_ACC(output, input, mult, numSamples) \
		if constexpr (MatMul<T, InChannels, OutChannels>::HasKernel()) \
		{ \
			const T* inputPtr = input.GetDataConst(); \
			const T* multPtr = mult.GetDataConst(); \
			\
			T* outputPtr = output.GetData(); \
			\
			MatMul<T, InChannels, OutChannels>::MultiplyAccumlulate(inputPtr, outputPtr, multPtr, numSamples); \
		} \
		else \
		{ \
			auto outBlock = output.GetEigenMap(); \
			\
			outBlock.noalias() += mult.GetEigenMapConst() * input.GetEigenMapConst(); \
		}


using namespace NeuralAudio;

namespace cpugrad
{
	template <typename T>
	class BackpropModelBaseT
	{
		public:
			BackpropModelBaseT() {}

			virtual ~BackpropModelBaseT() {}

			virtual void Forward(void* input, void* output)
			{
				(void)input;
				(void)output;
			}

			virtual size_t GetReceptiveField()
			{
				return 0;
			}

			virtual size_t GetMaxScratchBufferSize(size_t inputBufferSize)
			{
				(void)inputBufferSize;

				return 0;
			}

			virtual size_t GetInChannels()
			{
				return 0;
			}

			virtual size_t GetOutChannels()
			{
				return 0;
			}

			virtual size_t GetNumWeights()
			{
				return 0;
			}

			virtual size_t GetNumFeatures()
			{
				return 1;
			}

			virtual void RandomizeWeights(std::mt19937& rand)
			{
				RandomizeWeights(rand, GetNumFeatures());
			}

			void RandomizeWeights(std::mt19937& rand, size_t numFeatures)
			{
				//double stddev = std::sqrt(2.0 / (double)numFeatures);
				//std::normal_distribution<double> dist(0.0, stddev);

				double bound = 1.0 / std::sqrt((double)numFeatures);
				std::uniform_real_distribution<double> dist(-bound, bound);

				size_t numWeights = GetNumWeights();

				std::vector<float> weights(numWeights);

				for (size_t i = 0; i < numWeights; i++)
				{
					weights[i] = (float)dist(rand);
				}

				auto it = weights.begin();

				SetWeights(it);
			}

			virtual void SetWeights(std::vector<float>::iterator& inWeights)
			{
				(void)inWeights;
			}

			virtual void GetWeights(std::vector<float>::iterator& outWeights)
			{
				(void)outWeights;
			}

			virtual void AddWeightGradients()
			{
			}

			virtual void SetTrainingContext(TrainingContextT<T>* context)
			{
				this->trainingContext = context;

				AddWeightGradients();
			}

		protected:
			TrainingContextT<T>* trainingContext = nullptr;
	};

	template <typename T, int InChannels, int OutChannels>
	class BackpropModelT : public BackpropModelBaseT<T>
	{
		public:
			void Forward(void* input, void* output) override
			{
				Forward(*static_cast<const ChannelRowSpan<T, InChannels>*>(input), *static_cast<const ChannelRowSpan<T, OutChannels>*>(output));
			}

			virtual void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output)
			{
				(void)input;
				(void)output;
			}

			virtual void Backward(const ChannelRowSpan<T, InChannels>& input,	// forward input data
				const ChannelRowSpan<T, OutChannels>& dOutput,	// incoming backprop gradient
				const ChannelRowSpan<T, InChannels>& dInput)	// outgoing backprop gradient
			{
				(void)input;
				(void)dOutput;
				(void)dInput;
			}

			virtual void BackwardNoDInput(const ChannelRowSpan<T, InChannels>& input,	// forward input data
				const ChannelRowSpan<T, OutChannels>& dOutput)	// incoming backprop gradient
			{
				(void)input;
				(void)dOutput;
			}

			size_t GetNumFeatures() override
			{
				return InChannels;
			}

			size_t GetInChannels() override
			{
				return InChannels;
			}

			size_t GetOutChannels() override
			{
				return OutChannels;
			}

		protected:
			// Tried 4 interleaved accumulator sets for small (<=16 weight) matrices: only ~6% faster 3ch backward (1 thread), ~2% per epoch at 8 threads, so not kept
			void ComputeDW(const float* outPtr, const float* inPtr, float* dWPtr, size_t N)
			{
				float dWLocal[OutChannels * InChannels] = { 0.0f };

				for (size_t t = 0; t < N; t++)
				{
					const float* outT = outPtr + (t * OutChannels);
					const float* inT = inPtr + (t * InChannels);

					for (int j = 0; j < InChannels; j++)
					{
						const float xVal = inT[j];

						for (int i = 0; i < OutChannels; i++)
						{
							dWLocal[i + j * OutChannels] += outT[i] * xVal;
						}
					}
				}

				for (int k = 0; k < OutChannels * InChannels; k++)
				{
					dWPtr[k] += dWLocal[k];
				}
			}
	};

} // namespace cpugrad

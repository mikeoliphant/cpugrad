#pragma once

#include <array>
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
		static std::mt19937& getRand() {
			static std::mt19937 engine(123);
			//static std::random_device rd;
			//static std::mt19937 engine(rd());

			return engine;
		}

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

			virtual void RandomizeWeights()
			{
				RandomizeWeights(GetNumFeatures());
			}

			void RandomizeWeights(size_t numFeatures)
			{
				//double stddev = std::sqrt(2.0 / (double)numFeatures);
				//std::normal_distribution<double> dist(0.0, stddev);

				double bound = 1.0 / std::sqrt((double)numFeatures);
				std::uniform_real_distribution<double> dist(-bound, bound);

				size_t numWeights = GetNumWeights();

				std::vector<float> weights(numWeights);

				for (size_t i = 0; i < numWeights; i++)
				{
					weights[i] = (float)dist(getRand());
				}

				auto it = weights.begin();

				SetWeights(it);
			}

			virtual void SetWeights(std::vector<float>::iterator& inWeights)
			{
				(void)inWeights;
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
			void ComputeDW(const float* outPtr, const float* inPtr, float* dWPtr, size_t N)
			{
				float dWLocal[OutChannels * InChannels] = { 0.0f };

				for (size_t t = 0; t < N; t++)
				{
					const float* outT = outPtr + (t * OutChannels);
					const float* inT = inPtr + (t * InChannels);

#pragma unroll
					for (int j = 0; j < InChannels; j++)
					{
						const float xVal = inT[j];

#pragma unroll
						for (int i = 0; i < OutChannels; i++)
						{
							dWLocal[i + j * OutChannels] += outT[i] * xVal;
						}
					}
				}

#pragma unroll
				for (int k = 0; k < OutChannels * InChannels; k++)
				{
					dWPtr[k] += dWLocal[k];
				}
			}
	};
	
	template <typename T, int InChannels, int OutChannels, bool DoBias>
	class DenseBackpropT : public BackpropModelT<T, InChannels, OutChannels>
	{
		using BackpropModelT<T, InChannels, OutChannels>::ComputeDW;
		using BackpropModelT<T, InChannels, OutChannels>::trainingContext;

		public:
			DenseBackpropT()
			{
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				assert(input.GetNumCols() == output.GetNumCols());

				if constexpr (DoBias)
				{
					MATMUL_BIAS(output, input, weights, bias, input.GetNumCols())

					//output.GetEigenMap().noalias() = (weights.GetEigenMapConst() * input.GetEigenMapConst()).colwise() + bias;
				}
				else
				{
					MATMUL(output, input, weights, input.GetNumCols())

					//output.GetEigenMap().noalias() = weights.GetEigenMapConst() * input.GetEigenMapConst();
				}
			}

			void Backward(const ChannelRowSpan<T, InChannels>& input,
				const ChannelRowSpan<T, OutChannels>& dOutput,
				const ChannelRowSpan<T, InChannels>& dInput) override
			{
				assert((input.GetNumCols() == dOutput.GetNumCols()) && (input.GetNumCols() == dInput.GetNumCols()));

				dInput.GetEigenMap().noalias() = weights.GetEigenMapConst().transpose() * dOutput.GetEigenMapConst();

				ComputeDW(dOutput.GetDataConst(), input.GetDataConst(), dWeights.GetData(), dOutput.GetNumCols());

				if constexpr (DoBias)
				{
					dBias.noalias() += dOutput.GetEigenMapConst().rowwise().sum();
				}
			}

			void BackwardNoDInput(const ChannelRowSpan<T, InChannels>& input,
				const ChannelRowSpan<T, OutChannels>& dOutput) override
			{
				assert(input.GetNumCols() == dOutput.GetNumCols());

				ComputeDW(dOutput.GetDataConst(), input.GetDataConst(), dWeights.GetData(), dOutput.GetNumCols());

				if constexpr (DoBias)
				{
					dBias.noalias() += dOutput.GetEigenMapConst().rowwise().sum();
				}
			}

			size_t GetNumWeights() override
			{
				return OutChannels * InChannels + (DoBias ? OutChannels : 0);
			}

			void RandomizeWeights() override
			{
				BackpropModelT<T, InChannels, OutChannels>::RandomizeWeights();
			}

			void SetWeights(std::vector<float>::iterator& inWeights) override
			{
				for (size_t i = 0; i < OutChannels; i++)
					for (size_t j = 0; j < InChannels; j++)
						weights(i, j) = *(inWeights++);

				if constexpr (DoBias)
				{
					for (size_t i = 0; i < OutChannels; i++)
						bias(i) = *(inWeights++);
				}
			}

			void AddWeightGradients() override
			{
				trainingContext->AddWeightGradient(weights.GetData(), dWeights.GetData(), InChannels * OutChannels);

				if constexpr (DoBias)
				{
					trainingContext->AddWeightGradient(bias.data(), dBias.data(), OutChannels);
				}
			}

		private:
			ChannelBuffer<T, OutChannels, InChannels> weights;
			Eigen::Vector<T, OutChannels> bias;
			ChannelBuffer<T, OutChannels, InChannels> dWeights;
			Eigen::Vector<T, OutChannels> dBias;
	};

	template <typename T, int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DBackpropT : public BackpropModelT<T, InChannels, OutChannels>
	{
		using BackpropModelT<T, InChannels, OutChannels>::ComputeDW;
		using BackpropModelT<T, InChannels, OutChannels>::trainingContext;

		public:
			Conv1DBackpropT()
			{
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

				const size_t numSamplesOut = output.GetNumCols();

				for (size_t k = 0; k < KernelSize; k++)
				{
					const int inputOffset = Dilation * (int)k;

					MATMUL_ACC(output, input.Slice(inputOffset, numSamplesOut), this->weights[k], numSamplesOut)
				}

				if constexpr (DoBias)
					output.GetEigenMap().colwise() += bias;
			}

			void Backward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& dOutput, const ChannelRowSpan<T, InChannels>& dInput) override
			{
				assert(input.GetNumCols() == (dOutput.GetNumCols() + GetReceptiveField()));
				assert(input.GetNumCols() == dInput.GetNumCols());

				const size_t numSamplesOut = dOutput.GetNumCols();
				const auto doutMap = dOutput.GetEigenMapConst();
				
				if constexpr (DoBias)
				{
					dBias.noalias() += doutMap.rowwise().sum();
				}

				for (size_t k = 0; k < KernelSize; k++)
				{
					const int inputOffset = Dilation * (int)k;

					const auto inBlock = input.Slice(inputOffset, numSamplesOut);

					ComputeDW(dOutput.GetDataConst(), inBlock.GetDataConst(), dWeights[k].GetData(), numSamplesOut);
				}

				// It is faster to do two loops, since the memory regions are separate
				for (size_t k = 0; k < KernelSize; k++)
				{
					const int inputOffset = Dilation * (int)k;

					auto wMap = weights[k].GetEigenMapConst();
					auto dInputMap = dInput.Slice(inputOffset, numSamplesOut).GetEigenMap();
					dInputMap.noalias() += wMap.transpose() * dOutput.GetEigenMapConst();
				}
			}

			size_t GetNumFeatures() override
			{
				return InChannels * KernelSize;
			}

			size_t GetReceptiveField() override
			{
				return (KernelSize - 1) * Dilation;
			}

			size_t GetNumWeights() override
			{
				return OutChannels * InChannels * KernelSize + (DoBias ? OutChannels : 0);
			}

			void RandomizeWeights() override
			{
				BackpropModelT<T, InChannels, OutChannels>::RandomizeWeights();

				//if constexpr (DoBias)
				//{
				//	bias.setZero();
				//}
			}

			void SetWeights(std::vector<float>::iterator& inWeights) override
			{
				for (size_t i = 0; i < OutChannels; i++)
					for (size_t j = 0; j < InChannels; j++)
						for (size_t k = 0; k < KernelSize; k++)
							weights[k](i, j) = *(inWeights++);

				if constexpr (DoBias)
				{
					for (size_t i = 0; i < OutChannels; i++)
						bias(i) = *(inWeights++);
				}
			}

			void AddWeightGradients() override
			{
				for (int k = 0; k < KernelSize; k++)
				{
					trainingContext->AddWeightGradient(weights[k].GetData(), dWeights[k].GetData(), OutChannels * InChannels);
				}

				if constexpr (DoBias)
				{
					trainingContext->AddWeightGradient(bias.data(), dBias.data(), OutChannels);
				}
			}

		private:
			alignas(32) std::array<ChannelBuffer<T, OutChannels, InChannels>, KernelSize> weights;
			Eigen::Vector<T, OutChannels> bias;
			alignas(32) std::array<ChannelBuffer<T, OutChannels, InChannels>, KernelSize> dWeights;
			Eigen::Vector<T, OutChannels> dBias;
	};

	template <typename T, int Channels>
	class LeakyReLUT : public BackpropModelT<T, Channels, Channels>
	{
		public:
			void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& output) override
			{
				assert(input.GetNumCols() == output.GetNumCols());

				auto outputMap = output.GetEigenMap();
				auto inputMap = input.GetEigenMapConst();

				outputMap = (inputMap.array() < T(0.0)).select(inputMap.array() * T(0.01), inputMap.array());
			}

			void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput) override
			{
				assert(input.GetNumCols() == dOutput.GetNumCols());
				assert(input.GetNumCols() == dInput.GetNumCols());

				auto inputMap = input.GetEigenMapConst();
				auto dOutputMap = dOutput.GetEigenMapConst();
				auto dInputMap = dInput.GetEigenMap();

				dInputMap = (inputMap.array() < T(0.0)).select(dOutputMap.array() * T(0.01), dOutputMap.array());
			}
	};

} // namespace cpugrad

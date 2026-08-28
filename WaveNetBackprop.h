#pragma once

#include <array>
#include <vector>
#include <format>
#include "WaveNet.h"
#include "Activation.h"
#include "Optimizer.h"

#define MAX_BATCH_SIZE 14538

using namespace NeuralAudio;

namespace NeuralCpuTrain
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

			virtual void Reset()
			{}

			virtual void AddWeightGradients(OptimizerT<T>& optimizer)
			{
				(void)optimizer;
			}
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

		public:
			DenseBackpropT()
			{
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				if constexpr (DoBias)
				{
					output.GetEigenMap().noalias() += (weights.GetEigenMapConst() * input.GetEigenMapConst()).colwise() + bias;
				}
				else
				{
					output.GetEigenMap().noalias() += weights.GetEigenMapConst() * input.GetEigenMapConst();
				}
			}

			void Backward(const ChannelRowSpan<T, InChannels>& input,
				const ChannelRowSpan<T, OutChannels>& dOutput,
				const ChannelRowSpan<T, InChannels>& dInput) override
			{
				dInput.GetEigenMap().noalias() = weights.GetEigenMapConst().transpose() * dOutput.GetEigenMapConst();

				//auto map = dWeights.GetEigenMap();

				//map.noalias() += dOutput.GetEigenMapConst() * input.GetEigenMapConst().transpose();

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

			void AddWeightGradients(OptimizerT<T>& optimizer) override
			{
				optimizer.AddWeightGradient(weights.GetData(), dWeights.GetData(), InChannels * OutChannels);

				if constexpr (DoBias)
				{
					optimizer.AddWeightGradient(bias.data(), dBias.data(), OutChannels);
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
	
		public:
			Conv1DBackpropT()
			{
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				const size_t numFrames = input.GetNumCols();

				for (size_t k = 0; k < KernelSize; k++)
				{
					const int offset = Dilation * (int)(KernelSize - k - 1);

					const size_t validSize = numFrames - offset;

					const auto inBlock = input.Slice(0, validSize);
					const auto outBlock = output.Slice(offset, validSize);

					outBlock.GetEigenMap().noalias() += weights[k].GetEigenMapConst() * inBlock.GetEigenMapConst();
				}

				if constexpr (DoBias)
					output.GetEigenMap().colwise() += bias;
			}	

			void Backward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& dOutput, const ChannelRowSpan<T, InChannels>& dInput) override
			{
				const size_t numFrames = dOutput.GetNumCols();
				const auto doutMap = dOutput.GetEigenMapConst();
				
				if constexpr (DoBias)
				{
					dBias.noalias() += doutMap.rowwise().sum();
				}

				for (size_t k = 0; k < KernelSize; ++k)
				{
					const int offset = Dilation * (int)(KernelSize - k - 1);

					const size_t validSize = numFrames - offset;

					const auto inBlock = input.Slice(0, validSize);
					const auto dOutputBlock = dOutput.Slice(offset, validSize);

					//auto dwMap = dWeights[k].GetEigenMap();

					//dwMap.noalias() += dOutputBlock.GetEigenMapConst() * inBlock.GetEigenMapConst().transpose();

					ComputeDW(dOutputBlock.GetDataConst(), inBlock.GetDataConst(), dWeights[k].GetData(), validSize);

					auto wMap = weights[k].GetEigenMapConst();
					auto dInputMap = dInput.Slice(0, validSize).GetEigenMap();
					dInputMap.noalias() += wMap.transpose() * dOutputBlock.GetEigenMapConst();
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

			void AddWeightGradients(OptimizerT<T>& optimizer) override
			{
				for (int k = 0; k < KernelSize; k++)
				{
					optimizer.AddWeightGradient(weights[k].GetData(), dWeights[k].GetData(), OutChannels * InChannels);
				}

				if constexpr (DoBias)
				{
					optimizer.AddWeightGradient(bias.data(), dBias.data(), OutChannels);
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
				auto outputMap = output.GetEigenMap();
				auto inputMap = input.GetEigenMapConst();

				outputMap = (inputMap.array() < TCONST(0.0)).select(inputMap.array() * TCONST(0.01), inputMap.array());
			}

			void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput) override
			{
				auto inputMap = input.GetEigenMapConst();
				auto dOutputMap = dOutput.GetEigenMapConst();
				auto dInputMap = dInput.GetEigenMap();

				dInputMap = (inputMap.array() < TCONST(0.0)).select(dOutputMap.array() * TCONST(0.01), dOutputMap.array());
			}
	};

} // namespace NeuralCpuTrain

#pragma once

#include "WaveNet.h"
#include "Activation.h"
#include <array>
#include <vector>
#include <format>

#define MAX_BATCH_SIZE 131072

using namespace NeuralAudio;

namespace NeuralCpuTrain
{
	template <typename T, typename WeightType, int NumWeights>
	class WeightGradT
	{
		public:
			WeightGradT() :
				weightPtr(nullptr),
				t(0)
			{
				std::fill(m, m + NumWeights, T(0));
				std::fill(v, v + NumWeights, T(0));
			}

			WeightGradT(WeightType& weights) :
				weightPtr(&weights),
				t(0)
			{
				std::fill(m, m + NumWeights, T(0));
				std::fill(v, v + NumWeights, T(0));
			}

			~WeightGradT() = default;

			void SetWeights(WeightType& weights)
			{
				weightPtr = &weights;
			}

			WeightType& GetDWeights()
			{
				return dWeights;
			}

			virtual T* GetData(WeightType& w) = 0;
			virtual const T* GetDataConst(WeightType& w) const = 0;

			void ApplyGradients(float learningRate,
				float maxNorm = 1.0f,
				float weightDecay = 0.01f,
				float beta1 = 0.9f,
				float beta2 = 0.999f,
				float epsilon = 1e-8f)
			{
				if (!weightPtr) return;

				T* wp = GetData(*weightPtr);
				const T* dwp = GetDataConst(dWeights);

				// 1. Calculate the L2 Norm (Euclidean length) of the entire gradient block
				double totalSumSq = 0.0;
				for (size_t w = 0; w < NumWeights; w++)
				{
					totalSumSq += static_cast<double>(dwp[w] * dwp[w]);
				}
				float gradNorm = std::sqrt(static_cast<float>(totalSumSq));

				// 2. Determine scaling factor if norm exceeds our max allowed threshold
				float scaleFactor = 1.0f;
				if (gradNorm > maxNorm && gradNorm > 0.0f)
				{
					scaleFactor = maxNorm / gradNorm;
				}

				t++;
				const float biasCorrection1 = 1.0f - (float)std::pow(beta1, t);
				const float biasCorrection2 = 1.0f - (float)std::pow(beta2, t);

				for (size_t w = 0; w < NumWeights; w++)
				{
					// Apply scaling factor to the gradient uniformly
					float clipped_dw = dwp[w] * scaleFactor;

					// AdamW Parameter Updates
					wp[w] -= learningRate * weightDecay * wp[w];

					m[w] = beta1 * m[w] + (1.0f - beta1) * clipped_dw;
					v[w] = beta2 * v[w] + (1.0f - beta2) * (clipped_dw * clipped_dw);

					float m_hat = m[w] / biasCorrection1;
					float v_hat = v[w] / biasCorrection2;

					wp[w] -= (learningRate * m_hat) / ((float)std::sqrt(v_hat + epsilon));
				}
			}

		protected:
			WeightType* weightPtr;
			WeightType dWeights;

			T m[NumWeights]; // First moment vector (moving average of gradients)
			T v[NumWeights]; // Second moment vector (moving average of squared gradients)
			int t;           // Timestep counter
	};

	template <typename T, typename WeightType, int NumWeights>
	class EigenWeightGradT : public WeightGradT<T, WeightType, NumWeights>
	{
		public:
			using WeightGradT<T, WeightType, NumWeights>::WeightGradT;

			T* GetData(WeightType& w) override
			{
				return w.data();
			}

			const T* GetDataConst(WeightType& w) const override
			{
				return w.data();
			}
	};

	template <typename T, typename WeightType, int NumWeights>
	class ChannelBufferWeightGradT : public WeightGradT<T, WeightType, NumWeights>
	{
		public:
			using WeightGradT<T, WeightType, NumWeights>::WeightGradT;

			T* GetData(WeightType& w) override
			{
				return w.GetData();
			}

			const T* GetDataConst(WeightType& w) const override
			{
				return w.GetDataConst();
			}
	};

	template <typename T>
	class BackpropModelBaseT
	{
		static std::mt19937& getRand() {
			static std::mt19937 engine(123);
			return engine;
		}

		public:
			BackpropModelBaseT() {}

			virtual ~BackpropModelBaseT() {}

			virtual void Forward(void* input, void* output)
			{
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
				double stddev = std::sqrt(2.0 / (double)numFeatures);
				std::normal_distribution<double> dist(0.0, stddev);

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
			{}

			virtual void Reset()
			{}

			virtual void ResetGradients()
			{}

			virtual void ApplyGradients(float scale)
			{}

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
			}

			virtual void Backward(const ChannelRowSpan<T, InChannels>& input,	// forward input data
				const ChannelRowSpan<T, OutChannels>& dOutput,	// incoming backprop gradient
				const ChannelRowSpan<T, InChannels>& dInput)	// outgoing backprop gradient
			{
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
	};

	template <typename T, int InChannels, int OutChannels>
	class ChainBackpropModelT : public BackpropModelT<T, InChannels, OutChannels>
	{
		public:
			void AddLayer(std::unique_ptr<BackpropModelBaseT<T>> layer)
			{
				if (layers.size() == 0)
				{
					if (layer->GetInChannels() != InChannels)
					{
						throw std::runtime_error(std::format("First layer should have {} input channels, but has {}.", InChannels, layer->GetInChannels()));
					}
				}
				else
				{
					auto& lastLayer = layers.back();

					if (lastLayer->GetOutChannels() != layer->GetInChannels())
					{
						throw std::runtime_error(std::format("Layer input channels ({}) does not mach last layer output channesl ({}).", layer->GetInChannels(), lastLayer->GetOutChannels()));
					}
				}

				layers.push_back(std::move(layer));
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				auto& lastLayer = layers.back();

				if (lastLayer->GetOutChannels() != OutChannels)
				{
					throw std::runtime_error(std::format("Last layer should have {} output channels, but has {}.", OutChannels, lastLayer->GetOutChannels()));
				}

				for (auto& layer : layers)
				{
					layer->Forward((void *)&output, (void*)&output);
				}
			}

			//void Backward(const ChannelRowSpan<T, InSize>& input,
			//	const ChannelRowSpan<T, OutSize>& dOutput,
			//	const ChannelRowSpan<T, InSize>& dInput) override
			//{
			//}

		private:
			std::vector<std::unique_ptr<BackpropModelBaseT<T>>> layers;
	};
	
	template <typename T, int InSize, int OutSize, bool DoBias>
	class DenseBackpropT : public BackpropModelT<T, InSize, OutSize>
	{
		public:
			DenseBackpropT() :
				dWeights(weights),
				dBias(bias)
			{
			}

			void Forward(const ChannelRowSpan<T, InSize>& input, const ChannelRowSpan<T, OutSize>& output) override
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

			void Backward(const ChannelRowSpan<T, InSize>& input,
				const ChannelRowSpan<T, OutSize>& dOutput,
				const ChannelRowSpan<T, InSize>& dInput) override
			{
				dInput.GetEigenMap().noalias() = weights.GetEigenMapConst().transpose() * dOutput.GetEigenMapConst();

				auto map = dWeights.GetDWeights().GetEigenMap();

				map.noalias() += dOutput.GetEigenMapConst() * input.GetEigenMapConst().transpose();

				if constexpr (DoBias)
				{
					dBias.GetDWeights().noalias() += dOutput.GetEigenMapConst().rowwise().sum();
				}
			}

			size_t GetNumWeights() override
			{
				return OutSize * InSize + (DoBias ? OutSize : 0);
			}

			void RandomizeWeights() override
			{
				BackpropModelT<T, InSize, OutSize>::RandomizeWeights();

				if constexpr (DoBias)
				{
					bias.setZero();
				}
			}

			void SetWeights(std::vector<float>::iterator& inWeights) override
			{
				for (size_t i = 0; i < OutSize; i++)
					for (size_t j = 0; j < InSize; j++)
						weights(i, j) = *(inWeights++);

				if constexpr (DoBias)
				{
					for (size_t i = 0; i < OutSize; i++)
						bias(i) = *(inWeights++);
				}
			}

			void ResetGradients() override
			{
				dWeights.GetDWeights().SetZero();

				if constexpr (DoBias)
				{
					dBias.GetDWeights().setZero();
				}
			}

			void ApplyGradients(float scale) override
			{
				dWeights.ApplyGradients(scale);

				if constexpr (DoBias)
				{
					dBias.ApplyGradients(scale);
				}
			}

		private:
			ChannelBuffer<T, OutSize, InSize> weights;
			Eigen::Vector<T, OutSize> bias;
			ChannelBufferWeightGradT<T, ChannelBuffer<T, OutSize, InSize>, InSize * OutSize> dWeights;
			EigenWeightGradT<T, Eigen::Vector<T, OutSize>, OutSize> dBias;
	};

	template <typename T, int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DBackpropT : public BackpropModelT<T, InChannels, OutChannels>
	{
		public:
			Conv1DBackpropT() :
				dWeights(weights),
				dBias(bias)
			{
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				const size_t numFrames = input.GetNumCols();

				for (size_t k = 0; k < KernelSize; k++)
				{
					const auto offset = Dilation * ((int)k + 1 - KernelSize);

					const size_t validSize = numFrames + offset;

					const auto inBlock = input.Slice(0, validSize);
					const auto outBlock = output.Slice(-offset, validSize);

					outBlock.GetEigenMap().noalias() += weights.Slice(InChannels * k, InChannels).GetEigenMapConst() * inBlock.GetEigenMapConst();
				}

				if constexpr (DoBias)
					output.GetEigenMap().colwise() += bias;
			}	

			void Backward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& dOutput, const ChannelRowSpan<T, InChannels>& dInput) override
			{
				const size_t numFrames = dOutput.GetNumCols();
				const auto doutMap = dOutput.GetEigenMap();
				
				if constexpr (DoBias)
				{
					dBias.GetDWeights().noalias() += doutMap.rowwise().sum();
				}

				for (size_t k = 0; k < KernelSize; ++k)
				{
					const auto offset = Dilation * ((int)k + 1 - KernelSize);

					const size_t validSize = numFrames + offset;

					const auto inBlock = input.Slice(0, validSize);
					const auto dOutputBlock = dOutput.Slice(-offset, validSize);
			
					auto dwMap = dWeights.GetDWeights().Slice(InChannels * k, InChannels).GetEigenMap();
					dwMap.noalias() += dOutputBlock.GetEigenMapConst() * inBlock.GetEigenMapConst().transpose();

					auto wMap = weights.Slice(InChannels * k, InChannels).GetEigenMapConst();
					auto dInputMap = dInput.Slice(-offset, validSize).GetEigenMap();
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

				if constexpr (DoBias)
				{
					bias.setZero();
				}
			}

			void SetWeights(std::vector<float>::iterator& inWeights) override
			{
				for (size_t i = 0; i < OutChannels; i++)
					for (size_t j = 0; j < InChannels; j++)
						for (size_t k = 0; k < KernelSize; k++)
							weights(i, (k * InChannels) + j) = *(inWeights++);

				if constexpr (DoBias)
				{
					for (size_t i = 0; i < OutChannels; i++)
						bias(i) = *(inWeights++);
				}
			}

			void ResetGradients() override
			{
				for (int k = 0; k < KernelSize; k++)
				{
					dWeights.GetDWeights().SetZero();
				}

				if constexpr (DoBias)
				{
					dBias.GetDWeights().setZero();
				}
			}

			void ApplyGradients(float scale) override
			{
				dWeights.ApplyGradients(scale);

				if constexpr (DoBias)
				{
					dBias.ApplyGradients(scale);
				}
			}

		private:
			ChannelBuffer<T, OutChannels, InChannels * KernelSize> weights;
			Eigen::Vector<T, OutChannels> bias;
			ChannelBufferWeightGradT<T, ChannelBuffer<T, OutChannels, InChannels * KernelSize>, OutChannels * InChannels * KernelSize> dWeights;
			EigenWeightGradT<T, Eigen::Vector<T, OutChannels>, OutChannels> dBias;
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

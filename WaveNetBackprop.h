#pragma once

#include "WaveNet.h"
#include "Activation.h"
#include <array>
#include <vector>
#include <format>

using namespace NeuralAudio;

namespace NeuralCpuTrain
{
	template <typename T>
	class BackpropModelBaseT
	{
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
				std::mt19937 rng(123);

				size_t numWeights = GetNumWeights();

				std::vector<float> weights(numWeights);

				for (size_t i = 0; i < numWeights; i++)
				{
					weights[i] = (float)dist(rng);
				}

				auto it = weights.begin();

				SetWeights(it);
			}

			virtual void SetWeights(std::vector<float>::iterator& inWeights)
			{}

			virtual void Reset()
			{}

			virtual void DivideWeights(float divisor)
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
				forwardLayer(),
				dWeights(),
				dBias()
			{
			}

			void Forward(const ChannelRowSpan<T, InSize>& input, const ChannelRowSpan<T, OutSize>& output) override
			{
				forwardLayer.Process(input, output);
			}

			void Backward(const ChannelRowSpan<T, InSize>& input,
				const ChannelRowSpan<T, OutSize>& dOutput,
				const ChannelRowSpan<T, InSize>& dInput) override
			{
				dInput.GetEigenMap().noalias() = forwardLayer.GetWeights().GetEigenMapConst().transpose() * dOutput.GetEigenMapConst();

				auto map = dWeights.GetEigenMap();

				map.noalias() += dOutput.GetEigenMapConst() * input.GetEigenMapConst().transpose();

				if constexpr (DoBias)
				{
					dBias.noalias() += dOutput.GetEigenMapConst().rowwise().sum();
				}
			}

			size_t GetNumWeights() override
			{
				return forwardLayer.GetNumWeights();
			}

			void SetWeights(std::vector<float>::iterator& inWeights) override
			{
				forwardLayer.SetWeights(inWeights);
			}

			void Reset() override
			{
				dWeights.SetZero();

				if constexpr (DoBias)
				{
					dBias.setZero();
				}
			}

			void DivideWeights(float divisor) override
			{
				auto map = dWeights.GetEigenMap();

				map /= divisor;

				if constexpr (DoBias)
				{
					dBias /= divisor;
				}
			}

			void ApplyGradients(float scale) override
			{
				auto map = forwardLayer.GetWeights().GetEigenMap();

				map.noalias() -= dWeights.GetEigenMapConst() * scale;

				if constexpr (DoBias)
				{
					forwardLayer.GetBias().noalias() -= dBias * scale;
				}
			}

		private:
			DenseLayerT<T, InSize, OutSize, DoBias> forwardLayer;
			ChannelBuffer<T, OutSize, InSize> dWeights;
			DenseLayerT<T, InSize, OutSize, DoBias>::BiasType dBias;
	};

	template <typename T, int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DBackpropT : public BackpropModelT<T, InChannels, OutChannels>
	{
		public:
			Conv1DBackpropT()
			{
			}

			void Forward(const ChannelRowSpan<T, InChannels>& input, const ChannelRowSpan<T, OutChannels>& output) override
			{
				const size_t numFrames = input.GetNumCols();

				for (size_t k = 0; k < KernelSize; k++)
				{
					const T* weightPtr = this->weights[k].GetDataConst();

					const auto offset = Dilation * ((int)k + 1 - KernelSize);

					const size_t validSize = numFrames + offset;

					const auto inBlock = input.Slice(0, validSize);
					const auto outBlock = output.Slice(-offset, validSize);

					outBlock.GetEigenMap().noalias() += weights[k].GetEigenMapConst() * inBlock.GetEigenMapConst();
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
					dBias.noalias() += doutMap.rowwise().sum();
				}

				for (size_t k = 0; k < KernelSize; ++k)
				{
					const T* weightPtr = this->weights[k].GetDataConst();

					const auto offset = Dilation * ((int)k + 1 - KernelSize);

					const size_t validSize = numFrames + offset;

					const auto inBlock = input.Slice(0, validSize);
					const auto dOutputBlock = dOutput.Slice(-offset, validSize);
			
					auto dwMap = dWeights[k].GetEigenMap();
					dwMap.noalias() += dOutputBlock.GetEigenMapConst() * inBlock.GetEigenMapConst().transpose();

					auto wMap = weights[k].GetEigenMapConst();
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

			void Reset() override
			{
				for (int k = 0; k < KernelSize; k++)
				{
					dWeights[k].SetZero();
				}

				if constexpr (DoBias)
				{
					dBias.setZero();
				}
			}

			void DivideWeights(float divisor) override
			{
				for (int k = 0; k < KernelSize; k++)
				{
					auto map = dWeights[k].GetEigenMap();

					map /= divisor;
				}

				if constexpr (DoBias)
				{
					dBias /= divisor;
				}
			}

			void ApplyGradients(float scale) override
			{
				for (int k = 0; k < KernelSize; k++)
				{
					auto map = weights[k].GetEigenMap();

					map.noalias() -= dWeights[k].GetEigenMapConst() * scale;
				}

				if constexpr (DoBias)
				{
					bias.noalias() -= dBias * scale;
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

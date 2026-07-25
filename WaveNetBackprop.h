#pragma once

#include "WaveNet.h"
#include "Activation.h"
#include <array>
#include <vector>

using namespace NeuralAudio;

namespace NeuralCpuTrain
{
	template <typename T, int InSize, int OutSize, bool DoBias>
	struct DenseBackpropWeightsT
	{
		ChannelBuffer<T, OutSize, InSize> Weights;
		DenseLayerT<T, InSize, OutSize, DoBias>::BiasType Bias;

		void SetZero()
		{
			Weights.SetZero();

			if constexpr (DoBias)
			{
				Bias.setZero();
			}
		}

		void Divide(float divisor)
		{
			auto map = Weights.GetEigenMap();

			map /= divisor;

			if constexpr (DoBias)
			{
				Bias /= divisor;
			}
		}

		void ApplyGradients(DenseLayerT<T, InSize, OutSize, DoBias>& forwardLayer)
		{
			auto map = forwardLayer.GetWeights().GetEigenMap();
			
			map.noalias() -= Weights.GetEigenMapConst();

			if constexpr (DoBias)
			{
				forwardLayer.GetBias().noalias() -= Bias;
			}
		}
	};

	template <typename T, int InSize, int OutSize, bool DoBias>
	class DenseBackpropT
	{
		public:
			DenseBackpropT(DenseLayerT<T, InSize, OutSize, DoBias>& forwardLayer)
			{
				this->forwardLayer = &forwardLayer;
			}

			void Backward(const ChannelRowSpan<T, InSize>& input,	// forward input data
				const ChannelRowSpan<T, OutSize>& dOutput,	// incoming backprop gradient
				const ChannelRowSpan<T, InSize>& dInput,	// outgoing backprop gradient
				DenseBackpropWeightsT<T, InSize, OutSize, DoBias>& dWeights)	// accumulator for weight gradients
			{
				dInput.GetEigenMap().noalias() = forwardLayer->GetWeights().GetEigenMapConst().transpose() * dOutput.GetEigenMapConst();

				auto map = dWeights.Weights.GetEigenMap();

				map.noalias() += dOutput.GetEigenMapConst() * input.GetEigenMapConst().transpose();

				if constexpr (DoBias)
				{
					dWeights.Bias.noalias() += dOutput.GetEigenMapConst().rowwise().sum();
				}
			}

		private:
			DenseLayerT<T, InSize, OutSize, DoBias>* forwardLayer;
	};
} // namespace NeuralCpuTrain

#pragma once

#include "BackpropModel.h"

namespace cpugrad
{
	template <typename T, int InChannels, int OutChannels, bool DoBias>
	class DenseBackpropT : public BackpropModelT<T, InChannels, OutChannels>
	{
		using BackpropModelT<T, InChannels, OutChannels>::ComputeDW;
		using BackpropModelT<T, InChannels, OutChannels>::trainingContext;

	public:
		DenseBackpropT()
		{}

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

		void RandomizeWeights(std::mt19937& rand) override
		{
			BackpropModelT<T, InChannels, OutChannels>::RandomizeWeights(rand);
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

		void GetWeights(std::vector<float>::iterator& outWeights) override
		{
			for (size_t i = 0; i < OutChannels; i++)
				for (size_t j = 0; j < InChannels; j++)
					*(outWeights++) = weights(i, j);

			if constexpr (DoBias)
			{
				for (size_t i = 0; i < OutChannels; i++)
					*(outWeights++) = bias(i);
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
}
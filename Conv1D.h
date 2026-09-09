#pragma once

#include "BackpropModel.h"

namespace cpugrad
{

	template <typename T, int InChannels, int OutChannels, int KernelSize, bool DoBias, int Dilation>
	class Conv1DBackpropT : public BackpropModelT<T, InChannels, OutChannels>
	{
		using BackpropModelT<T, InChannels, OutChannels>::ComputeDW;
		using BackpropModelT<T, InChannels, OutChannels>::trainingContext;

	public:
		Conv1DBackpropT()
		{}

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

		void GetWeights(std::vector<float>::iterator& outWeights) override
		{
			for (size_t i = 0; i < OutChannels; i++)
				for (size_t j = 0; j < InChannels; j++)
					for (size_t k = 0; k < KernelSize; k++)
						*(outWeights++) = weights[k](i, j);

			if constexpr (DoBias)
			{
				for (size_t i = 0; i < OutChannels; i++)
					*(outWeights++) = bias(i);
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
}

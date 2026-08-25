#pragma once

#include <vector>

template <typename T>
class AdamWeightGradientT
{
	public:
		AdamWeightGradientT(T* weights, T* dWeights, size_t numWeights) :
			weights(weights),
			dWeights(dWeights),
			v(numWeights),
			m(numWeights),
			numWeights(numWeights)
		{
		}

		void ResetGradients()
		{
			std::fill(dWeights, dWeights + numWeights, 0);
		}

		double GetSumSquareDWeights()
		{
			double totalSumSq = 0.0;

			for (size_t w = 0; w < numWeights; w++)
			{
				totalSumSq += static_cast<double>(dWeights[w] * dWeights[w]);
			}

			return totalSumSq;
		}

		void ApplyGradients(float learningRate, float scaleFactor = 1.0f,
			float weightDecay = 0, // 0.01f,
			float beta1 = 0.9f,
			float beta2 = 0.999f,
			float epsilon = 1e-8f)
		{
			t++;

			const float biasCorrection1 = 1.0f - (float)std::pow(beta1, t);
			const float biasCorrection2 = 1.0f - (float)std::pow(beta2, t);

			for (size_t w = 0; w < numWeights; w++)
			{
				// Apply scaling factor to the gradient uniformly
				float clipped_dw = dWeights[w] * scaleFactor;

				// AdamW Parameter Updates
				weights[w] -= learningRate * weightDecay * weights[w];

				m[w] = beta1 * m[w] + (1.0f - beta1) * clipped_dw;
				v[w] = beta2 * v[w] + (1.0f - beta2) * (clipped_dw * clipped_dw);

				float m_hat = m[w] / biasCorrection1;
				float v_hat = v[w] / biasCorrection2;

				weights[w] -= (learningRate * m_hat) / ((float)std::sqrt(v_hat + epsilon));
			}
		}

	private:
		T* weights;
		T* dWeights;
		std::vector<T> v;	// First moment vector (moving average of gradients)
		std::vector<T> m;	// Second moment vector (moving average of squared gradients)
		size_t t = 0;	// Timestep counter
		size_t numWeights;
};

template <typename T>
class OptimizerT
{
	public:
		virtual ~OptimizerT() = default;

		virtual void AddWeightGradient(T* weights, T* dWeights, size_t numWeights)
		{
			(void)weights;
			(void)dWeights;
			(void)numWeights;
		}

		virtual void ResetGradients()
		{
		}

		virtual void ApplyGradients()
		{
		}

	private:
};

template <typename T>
class AdamOptimizerT : public OptimizerT<T>
{
	public:
		void AddWeightGradient(T* weights, T* dWeights, size_t numWeights) override
		{
			weightGrads.emplace_back(weights, dWeights, numWeights);
		}

		void ResetGradients() override
		{
			for (AdamWeightGradientT<T>& grad : weightGrads)
			{
				grad.ResetGradients();
			}
		}

		void ApplyGradients() override
		{
			double totalSumSq = 0;

			for (AdamWeightGradientT<T>& grad : weightGrads)
			{
				totalSumSq += grad.GetSumSquareDWeights();
			}

			float gradNorm = std::sqrt(static_cast<float>(totalSumSq));

			float scaleFactor = 1.0f;

			if (gradNorm > maxNorm && gradNorm > 0.0f)
			{
				scaleFactor = maxNorm / gradNorm;
			}

			for (AdamWeightGradientT<T>& grad : weightGrads)
			{
				grad.ApplyGradients(learningRate, scaleFactor);
			}
		}

	private:
		std::vector<AdamWeightGradientT<T>> weightGrads;
		float learningRate = 0.004;
		float maxNorm = 1.0f;
};

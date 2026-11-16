#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

template <typename T>
class LossT
{
public:
	virtual ~LossT() = default;
	virtual std::string& GetName() { return name; }
	virtual void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, double scaleFactor) = 0;
	virtual double GetMeanLoss(const T* output, const T* target, size_t numSamples) = 0;

protected:
	std::string name;
};

template <typename T>
class MSELossT : public LossT<T>
{
public:
	MSELossT()
	{
		this->name = "MSE";
	}

	void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, double scaleFactor) override
	{
		double scale = 2.0 * scaleFactor / static_cast<double>(numSamples);

		for (size_t t = 0; t < numSamples; t++)
		{
			outGradient[t] += (static_cast<double>(output[t]) - static_cast<double>(target[t])) * scale;
		}
	}

	double GetMeanLoss(const T* output, const T* target, size_t numSamples) override
	{
		double tot = 0;

		for (size_t t = 0; t < numSamples; t++)
		{
			double diff = static_cast<double>(output[t]) - static_cast<double>(target[t]);
			tot += (diff * diff);
		}

		return tot / static_cast<double>(numSamples);
	}
};

template <typename T>
class ESRLossT : public LossT<T>
{
	static constexpr T epsilon = T(1e-4);

public:
	ESRLossT()
	{
		this->name = "ESR";
	}

	void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, double scaleFactor) override
	{
		double totEnergy = 0;

		for (size_t t = 0; t < numSamples; t++)
		{
			 if (std::abs(target[t]) < epsilon)
			 {
				totEnergy += epsilon;
			 }

			totEnergy += (target[t] * target[t]);
		}

		double scale = 2.0 * scaleFactor / totEnergy;

		for (size_t t = 0; t < numSamples; t++)
		{
			outGradient[t] += (static_cast<double>(output[t]) - static_cast<double>(target[t])) * scale;
		}
	}

	double GetMeanLoss(const T* output, const T* target, size_t numSamples) override
	{
		double totEnergy = 0;

		for (size_t t = 0; t < numSamples; t++)
		{
			totEnergy += (target[t] * target[t]);
		}

		totEnergy += epsilon;

		double tot = 0;

		for (size_t t = 0; t < numSamples; t++)
		{
			double diff = static_cast<double>(output[t]) - static_cast<double>(target[t]);
			tot += (diff * diff);
		}

		return (tot / totEnergy);
	}
};

// Weighted sum of any number of losses, built at runtime
template <typename T>
class JointLossT : public LossT<T>
{
	public:
		void AddLoss(std::unique_ptr<LossT<T>> loss, double weight)
		{
			if (!this->name.empty())
				this->name += "/";

			this->name += loss->GetName();

			terms.push_back({ std::move(loss), weight });
		}

		void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, double scaleFactor) override
		{
			for (auto& term : terms)
			{
				term.loss->ComputeLoss(output, target, outGradient, numSamples, scaleFactor * term.weight);
			}
		}

		double GetMeanLoss(const T* output, const T* target, size_t numSamples) override
		{
			double total = 0;

			for (auto& term : terms)
			{
				total += term.loss->GetMeanLoss(output, target, numSamples) * term.weight;
			}

			return total;
		}

	private:
		struct Term
		{
			std::unique_ptr<LossT<T>> loss;
			double weight;
		};

		std::vector<Term> terms;
};

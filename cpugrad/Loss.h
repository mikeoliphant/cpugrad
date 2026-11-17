#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

template <typename T>
class LossT
{
public:
	virtual ~LossT() = default;
	virtual std::string& GetName() { return name; }
	virtual void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, double scaleFactor) = 0;
	// Mean loss over the data, computed on one thread. The default uses the statistics functions below, so a loss either
	// implements those (and then also supports parallel computation) or overrides this.
	virtual double GetMeanLoss(const T* output, const T* target, size_t numSamples)
	{
		std::vector<double> stats(GetNumMeanStatistics(), 0.0);

		if (stats.empty())
			throw std::logic_error("Loss \"" + GetName() + "\" must override GetMeanLoss() or implement the mean statistics functions");

		AccumulateMeanStatistics(output, target, numSamples, 0, 1, stats.data());

		return MeanLossFromStatistics(stats.data(), numSamples);
	}

	// Support for computing GetMeanLoss() in parallel. The mean is a function of GetNumMeanStatistics() sums, each of
	// which can be accumulated over one part of the data independently (AccumulateMeanStatistics() adds part "part" of
	// "numParts" into stats). Once the parts' sums are added together, MeanLossFromStatistics() gives the mean.
	// A loss that returns 0 from GetNumMeanStatistics() doesn't support this, and must override GetMeanLoss().
	virtual size_t GetNumMeanStatistics() { return 0; }

	virtual void AccumulateMeanStatistics(const T* output, const T* target, size_t numSamples, size_t part, size_t numParts, double* stats)
	{
		(void)output; (void)target; (void)numSamples; (void)part; (void)numParts; (void)stats;
	}

	virtual double MeanLossFromStatistics(const double* stats, size_t numSamples)
	{
		(void)stats; (void)numSamples;

		return 0;
	}

protected:
	// Contiguous range [begin, end) of part "part" when splitting count items into numParts
	static std::pair<size_t, size_t> PartRange(size_t count, size_t part, size_t numParts)
	{
		return { (count * part) / numParts, (count * (part + 1)) / numParts };
	}

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

	// Statistic: sum of squared differences
	size_t GetNumMeanStatistics() override { return 1; }

	void AccumulateMeanStatistics(const T* output, const T* target, size_t numSamples, size_t part, size_t numParts, double* stats) override
	{
		auto [begin, end] = this->PartRange(numSamples, part, numParts);

		double tot = 0;

		for (size_t t = begin; t < end; t++)
		{
			double diff = static_cast<double>(output[t]) - static_cast<double>(target[t]);
			tot += (diff * diff);
		}

		stats[0] += tot;
	}

	double MeanLossFromStatistics(const double* stats, size_t numSamples) override
	{
		return stats[0] / static_cast<double>(numSamples);
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

	// Statistics: target energy, sum of squared differences
	size_t GetNumMeanStatistics() override { return 2; }

	void AccumulateMeanStatistics(const T* output, const T* target, size_t numSamples, size_t part, size_t numParts, double* stats) override
	{
		auto [begin, end] = this->PartRange(numSamples, part, numParts);

		double totEnergy = 0;
		double tot = 0;

		for (size_t t = begin; t < end; t++)
		{
			totEnergy += (target[t] * target[t]);

			double diff = static_cast<double>(output[t]) - static_cast<double>(target[t]);
			tot += (diff * diff);
		}

		stats[0] += totEnergy;
		stats[1] += tot;
	}

	double MeanLossFromStatistics(const double* stats, size_t numSamples) override
	{
		(void)numSamples;

		return stats[1] / (stats[0] + epsilon);
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
			if (GetNumMeanStatistics() > 0)
				return LossT<T>::GetMeanLoss(output, target, numSamples);

			// Some term doesn't support the statistics functions, so combine the terms' own means
			double total = 0;

			for (auto& term : terms)
			{
				total += term.loss->GetMeanLoss(output, target, numSamples) * term.weight;
			}

			return total;
		}

		// Statistics: those of each term in turn (only supported if every term supports it)
		size_t GetNumMeanStatistics() override
		{
			size_t num = 0;

			for (auto& term : terms)
			{
				size_t termNum = term.loss->GetNumMeanStatistics();

				if (termNum == 0)
					return 0;

				num += termNum;
			}

			return num;
		}

		void AccumulateMeanStatistics(const T* output, const T* target, size_t numSamples, size_t part, size_t numParts, double* stats) override
		{
			for (auto& term : terms)
			{
				term.loss->AccumulateMeanStatistics(output, target, numSamples, part, numParts, stats);

				stats += term.loss->GetNumMeanStatistics();
			}
		}

		double MeanLossFromStatistics(const double* stats, size_t numSamples) override
		{
			double total = 0;

			for (auto& term : terms)
			{
				total += term.loss->MeanLossFromStatistics(stats, numSamples) * term.weight;

				stats += term.loss->GetNumMeanStatistics();
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

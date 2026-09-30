#pragma once

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
	static constexpr T epsilon = T(1e-8);

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
			totEnergy += (target[t] * target[t]);
		}

		totEnergy += epsilon;

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

template <typename T, typename LossType1, double Loss1Weight, typename LossType2, double Loss2Weight>
class JointLossT : public LossT<T>
{
	public:
		JointLossT()
		{
			this->name = loss1.GetName() + "/"  + loss2.GetName();
		}

		void ComputeLoss(const T* output, const T* target, T* outGradient, size_t numSamples, double scaleFactor) override
		{
			loss1.ComputeLoss(output, target, outGradient, numSamples, scaleFactor * Loss1Weight);
			loss2.ComputeLoss(output, target, outGradient, numSamples, scaleFactor * Loss2Weight);
		}

		double GetMeanLoss(const T* output, const T* target, size_t numSamples) override
		{
			return (loss1.GetMeanLoss(output, target, numSamples) * Loss1Weight) + (loss2.GetMeanLoss(output, target, numSamples) * Loss2Weight);
		}

	private:
		LossType1 loss1;
		LossType2 loss2;	
};

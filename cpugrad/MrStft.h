#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>
#include <cstring>
#include <numbers>

#include "Loss.h"
#include "Fft.h"

struct StftWindowConfig
{
    size_t FftSize;
    size_t HopSize;
    size_t WindowSize;
};

// STFT framing (reflect padding, Hann window, overlap-add) for one resolution. The time and spectrum buffers are passed
// in by the caller: GetFftSize() floats and GetFreqBins() bins, aligned to cpugrad::Fft::RequiredAlignment.
template <typename T>
class StftFrameProcessor
{
public:
    StftFrameProcessor(size_t fftSize, size_t hopSize, size_t windowSize)
        : fftSize(fftSize), hopSize(hopSize), windowSize(windowSize),
        freqBins(fftSize / 2 + 1), padSize(fftSize / 2),
        frameOffset((fftSize - windowSize) / 2),
        numSamples(0), paddedLength(0), outputFrames(0),
        fft(fftSize)
    {
        frameOverlapNormalization = CalculateFrameOverlapNormalization();

        hannWindow.resize(windowSize);

        if (windowSize > 1)
        {
            // Periodic (not symmetric!) Hann window
            const float denominator = static_cast<float>(windowSize);

            for (size_t i = 0; i < windowSize; ++i)
            {
                hannWindow[i] = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> *static_cast<float>(i) / denominator));
            }
        }
    }

    // Configures or updates the processing geometry dynamically when audio chunk size variations occur
    void SetNumSamples(size_t newNumSamples) noexcept
    {
        numSamples = newNumSamples;
        paddedLength = numSamples + 2 * padSize;
        outputFrames = (paddedLength - fftSize) / hopSize + 1;
    }

    // Informational structural query methods
    size_t GetFftSize() const noexcept { return fftSize; }
    size_t GetOutputFrames() const noexcept { return outputFrames; }
    size_t GetFreqBins() const noexcept { return freqBins; }

    // Extracts and windows one frame into timeBuffer, then transforms it into spectrum
    void ProcessForwardFrame(const T* __restrict audioTimeline, size_t frameIdx, float* timeBuffer, std::complex<float>* spectrum)
    {
        size_t sampleStart = frameIdx * hopSize;

        std::fill(timeBuffer, timeBuffer + fftSize, 0.0f);

        for (size_t i = 0; i < windowSize; ++i)
        {
            size_t paddedIdx = sampleStart + frameOffset + i;
            float sample = GetReflectSample(audioTimeline, paddedIdx);

            timeBuffer[frameOffset + i] = sample * hannWindow[i];
        }

        fft.Forward(timeBuffer, spectrum);
    }

    // Transforms one frame's gradient spectrum back (via timeBuffer) and overlap-adds it into outGradient.
    // gradSpectrum is used as scratch space.
    void ProcessBackwardFrame(T* __restrict outGradient, size_t frameIdx, std::complex<float>* gradSpectrum, float* timeBuffer, double externalScale)
    {
        fft.Inverse(gradSpectrum, timeBuffer);

        const double scale = externalScale * fft.InverseScale();

        size_t sampleStart = frameIdx * hopSize;

        for (size_t i = 0; i < windowSize; ++i)
        {
            size_t paddedIdx = sampleStart + frameOffset + i;
            float windowedGradSample = timeBuffer[frameOffset + i] * hannWindow[i];

            AccumulateReflectGradient(outGradient, paddedIdx, static_cast<float>(windowedGradSample * scale));
        }
    }

private:
    double CalculateFrameOverlapNormalization() noexcept
    {
        double overlapRatio = static_cast<double>(windowSize) / static_cast<double>(hopSize);
        double linearOverlapSum = 0.5 * overlapRatio;
        double squaredOverlapSum = 0.375 * overlapRatio;

        return (linearOverlapSum / squaredOverlapSum) * 0.5;
    }

    inline float GetReflectSample(const T* audio, size_t paddedIdx) const
    {
        if (paddedIdx < padSize)
        {
            return static_cast<float>(audio[padSize - paddedIdx]);
        }

        size_t actualIdx = paddedIdx - padSize;

        if (actualIdx < numSamples)
        {
            return static_cast<float>(audio[actualIdx]);
        }

        size_t rightReflectIdx = numSamples - 2 - (actualIdx - numSamples);

        return static_cast<float>(audio[rightReflectIdx]);
    }

    inline void AccumulateReflectGradient(T* outGradient, size_t paddedIdx, float value) const
    {
        if (paddedIdx < padSize)
        {
            size_t reflectIdx = padSize - paddedIdx;
            if (reflectIdx < numSamples)
            {
                outGradient[reflectIdx] += static_cast<T>(value);
            }
            return;
        }

        size_t actualIdx = paddedIdx - padSize;

        if (actualIdx < numSamples)
        {
            outGradient[actualIdx] += static_cast<T>(value);
            return;
        }

        size_t rightReflectIdx = numSamples - 2 - (actualIdx - numSamples);

        if (rightReflectIdx < numSamples)
        {
            outGradient[rightReflectIdx] += static_cast<T>(value);
        }
    }

    size_t fftSize;
    size_t hopSize;
    size_t windowSize;
    size_t freqBins;
    size_t padSize;
    size_t frameOffset;
    double frameOverlapNormalization;

    // Dynamic tracking elements calculated on SetNumSamples() execution invocation
    size_t numSamples;
    size_t paddedLength;
    size_t outputFrames;

    cpugrad::Fft fft;
    std::vector<float> hannWindow;
};

struct StftLossMetrics
{
    double FrobeniusDiff;
    double FrobeniusTarget;
    double L1LogDiff;
};

template <typename T>
class MultiResolutionStftLossT : public LossT<T>
{
private:
    std::vector<std::unique_ptr<StftFrameProcessor<T>>> stftProcessors;
    std::vector<StftWindowConfig> configurations;

    // Working buffers, sized for the largest FFT. The target spectrum is reused for the gradient spectrum.
    cpugrad::FftVector<float> timeBuffer;
    cpugrad::FftVector<std::complex<float>> targetSpectrum;
    cpugrad::FftVector<std::complex<float>> outputSpectrum;

    const T epsilon = static_cast<T>(1e-8);
    // Spectral convergence defaults to off for training: per-segment SC blows up on near-silent segments,
    // and in auraloss's batch-wide form it contributes little gradient. Use SetWeights(1, 1) to match auraloss's metric.
    double specWeight = 0.0;
    double logMagWeight = 1.0;

public:
    MultiResolutionStftLossT(std::vector<StftWindowConfig> configs = { { 2048, 240, 1200 }, { 1024, 120, 600 }, {512, 50, 240} }) :
        configurations(configs)
    {
        this->name = "MR-STFT";

        size_t maxFft = 0;

        for (StftWindowConfig& config : configurations)
        {
            stftProcessors.emplace_back(
                std::make_unique<StftFrameProcessor<T>>(
                    config.FftSize,
                    config.HopSize,
                    config.WindowSize
                )
            );

            maxFft = std::max(maxFft, config.FftSize);
        }

        timeBuffer.resize(maxFft);
        targetSpectrum.resize((maxFft / 2) + 1);
        outputSpectrum.resize((maxFft / 2) + 1);
    }

    ~MultiResolutionStftLossT() override = default;

    MultiResolutionStftLossT(const MultiResolutionStftLossT&) = delete;
    MultiResolutionStftLossT& operator=(const MultiResolutionStftLossT&) = delete;

    // Maintain move safety
    MultiResolutionStftLossT(MultiResolutionStftLossT&&) noexcept = default;
    MultiResolutionStftLossT& operator=(MultiResolutionStftLossT&&) noexcept = default;

    void SetWeights(double specWeight, double logMagWeight)
    {
        this->specWeight = specWeight;
        this->logMagWeight = logMagWeight;
    }

    // Spectral metrics summed over frames [firstFrame, endFrame) of resolution c
    StftLossMetrics ComputeForwardMetricsForScale(size_t c, const T* output, const T* target, size_t firstFrame, size_t endFrame, size_t freqBins)
    {
        StftLossMetrics metrics = { 0.0, 0.0, 0.0 };

        for (size_t frame = firstFrame; frame < endFrame; frame++)
        {
            stftProcessors[c]->ProcessForwardFrame(target, frame, timeBuffer.data(), targetSpectrum.data());
            stftProcessors[c]->ProcessForwardFrame(output, frame, timeBuffer.data(), outputSpectrum.data());

            for (size_t k = 0; k < freqBins; k++)
            {
                const std::complex<float> t = targetSpectrum[k];
                const std::complex<float> o = outputSpectrum[k];

                float targetPower = t.real() * t.real() + t.imag() * t.imag();
                float outputPower = o.real() * o.real() + o.imag() * o.imag();

                float targetMag = std::sqrt(std::max(targetPower, static_cast<float>(epsilon)));
                float outputMag = std::sqrt(std::max(outputPower, static_cast<float>(epsilon)));

                float magDiff = targetMag - outputMag;
                metrics.FrobeniusDiff += magDiff * magDiff;
                metrics.FrobeniusTarget += targetMag * targetMag;

                metrics.L1LogDiff += std::abs(std::log(targetMag) - std::log(outputMag));
            }
        }

        return metrics;
    }

    void ComputeLoss(const T* __restrict output, const T* __restrict target, T* __restrict outGradient, size_t numSamples, double scaleFactor) override
    {
        // Per-resolution losses are already means (over frames x bins), so only average across resolutions
        double globalScale = scaleFactor / static_cast<double>(configurations.size());

        for (size_t c = 0; c < configurations.size(); c++)
        {
            stftProcessors[c]->SetNumSamples(numSamples);

            size_t fftSize = configurations[c].FftSize;
            size_t outputFrames = stftProcessors[c]->GetOutputFrames();
            size_t freqBins = stftProcessors[c]->GetFreqBins();

            // The forward metrics pass only supplies the spectral convergence norms, so skip it when SC is off
            StftLossMetrics metrics = { 0.0, 0.0, 0.0 };

            if (specWeight != 0.0)
            {
                metrics = ComputeForwardMetricsForScale(c, output, target, 0, outputFrames, freqBins);
            }

            // --- PASS 2: Adjoint Backpropagation Pass ---
            double sqrtDiff = std::sqrt(metrics.FrobeniusDiff);
            double sqrtTarget = std::sqrt(metrics.FrobeniusTarget);
            double logDenom = static_cast<double>(outputFrames * freqBins);
            double denominator = sqrtDiff * sqrtTarget;
            double safe_denominator = std::max(denominator, static_cast<double>(epsilon));

            for (size_t frame = 0; frame < outputFrames; frame++)
            {
                // Rematerialize forward states seamlessly for this frame
                stftProcessors[c]->ProcessForwardFrame(target, frame, timeBuffer.data(), targetSpectrum.data());
                stftProcessors[c]->ProcessForwardFrame(output, frame, timeBuffer.data(), outputSpectrum.data());

                // The gradient spectrum overwrites the target spectrum, bin by bin
                std::complex<float>* gradSpectrum = targetSpectrum.data();

                for (size_t k = 0; k < freqBins; k++)
                {
                    float outReal = outputSpectrum[k].real();
                    float outImag = outputSpectrum[k].imag();
                    float tReal = targetSpectrum[k].real();
                    float tImag = targetSpectrum[k].imag();

                    float rawTargetPower = tReal * tReal + tImag * tImag;
                    float rawOutputPower = outReal * outReal + outImag * outImag;

                    float targetMag = std::sqrt(std::max(rawTargetPower, static_cast<float>(epsilon)));
                    float outputMag = std::sqrt(std::max(rawOutputPower, static_cast<float>(epsilon)));

                    double dLossDMag = 0.0;

                    // Match torch clamp(power, eps) backward: no gradient flows through clamped bins
                    if (rawOutputPower >= static_cast<float>(epsilon))
                    {
                        double dScDMag = 0.0;

                        dScDMag = (outputMag - targetMag) / safe_denominator;

                        double dLogDMag = 0.0;

                        double magDifference = static_cast<double>(outputMag) - static_cast<double>(targetMag);

                        double sign = 0.0;

                        if (std::abs(magDifference) > 0)
                        {
                            sign = (magDifference > 0.0) ? 1.0 : -1.0;

                            dLogDMag = (sign / (logDenom * static_cast<double>(outputMag)));
                        }

                        dLossDMag = ((dScDMag * specWeight) + (dLogDMag * logMagWeight)) * (static_cast<double>(fftSize) * 0.5);

                        gradSpectrum[k] = std::complex<float>(static_cast<float>(dLossDMag * outReal / outputMag), static_cast<float>(dLossDMag * outImag / outputMag));
                    }
                    else
                    {
                        gradSpectrum[k] = std::complex<float>(0.0f, 0.0f);
                    }
                }

                // The inverse real FFT weights DC and Nyquist by 1/N but interior bins by 2/N.
                // After the N/2 scale above, DC and Nyquist need doubling to get the full gradient.
                gradSpectrum[0].real(gradSpectrum[0].real() * 2.0f);
                gradSpectrum[freqBins - 1].real(gradSpectrum[freqBins - 1].real() * 2.0f);

                // Process backward pass and accumulate straight into outGradient via our processor instance
                stftProcessors[c]->ProcessBackwardFrame(outGradient, frame, gradSpectrum, timeBuffer.data(), globalScale);
            }
        }
    }

    // GetMeanLoss() comes from LossT, via these.
    // Statistics: for each resolution, the FrobeniusDiff, FrobeniusTarget and L1LogDiff sums over frames.
    // Parts split each resolution's frames, so the result doesn't depend on how the data is split (up to summation order).
    size_t GetNumMeanStatistics() override
    {
        return 3 * configurations.size();
    }

    void AccumulateMeanStatistics(const T* output, const T* target, size_t numSamples, size_t part, size_t numParts, double* stats) override
    {
        for (size_t c = 0; c < configurations.size(); c++)
        {
            stftProcessors[c]->SetNumSamples(numSamples);

            auto [firstFrame, endFrame] = this->PartRange(stftProcessors[c]->GetOutputFrames(), part, numParts);

            StftLossMetrics metrics = ComputeForwardMetricsForScale(c, output, target, firstFrame, endFrame, stftProcessors[c]->GetFreqBins());

            stats[3 * c] += metrics.FrobeniusDiff;
            stats[3 * c + 1] += metrics.FrobeniusTarget;
            stats[3 * c + 2] += metrics.L1LogDiff;
        }
    }

    double MeanLossFromStatistics(const double* stats, size_t numSamples) override
    {
        float totalMrLoss = 0.0f;

        for (size_t c = 0; c < configurations.size(); c++)
        {
            stftProcessors[c]->SetNumSamples(numSamples);

            size_t outputFrames = stftProcessors[c]->GetOutputFrames();
            size_t freqBins = stftProcessors[c]->GetFreqBins();

            const double frobeniusDiff = stats[3 * c];
            const double frobeniusTarget = stats[3 * c + 1];
            const double l1LogDiff = stats[3 * c + 2];

            float spectralConvergence = (frobeniusTarget > 0.0) ? static_cast<float>(std::sqrt(frobeniusDiff) / std::sqrt(frobeniusTarget)) : 0.0f;
            float logMagnitudeLoss = static_cast<float>(l1LogDiff / (static_cast<double>(outputFrames) * static_cast<double>(freqBins)));

            totalMrLoss += ((specWeight * spectralConvergence) + (logMagWeight * logMagnitudeLoss));
        }

        return totalMrLoss / static_cast<float>(configurations.size());
    }
 };

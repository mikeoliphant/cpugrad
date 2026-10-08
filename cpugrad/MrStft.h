#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstring>
#include <numbers>

#include "Loss.h"
#include "AudioFFT.h"

struct StftWindowConfig
{
    size_t FftSize;
    size_t HopSize;
    size_t WindowSize;
};

template <typename T>
class StftFrameProcessor
{
public:
    StftFrameProcessor(size_t fftSize, size_t hopSize, size_t windowSize)
        : fftSize(fftSize), hopSize(hopSize), windowSize(windowSize),
        freqBins(fftSize / 2 + 1), padSize(fftSize / 2),
        frameOffset((fftSize - windowSize) / 2),
        numSamples(0), paddedLength(0), outputFrames(0),
        fftEngine()
    {
        frameOverlapNormalization = CalculateFrameOverlapNormalization();

        fftEngine.init(fftSize);

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
    size_t GetOutputFrames() const noexcept { return outputFrames; }
    size_t GetFreqBins() const noexcept { return freqBins; }

    // Extracts, windows, and transforms a single frame slice forward into the frequency domain
    void ProcessForwardFrame(const T* __restrict audioTimeline, size_t frameIdx,
        std::vector<float>& timeScratchpad, std::vector<float>& realDest, std::vector<float>& imagDest) const
    {
        size_t sampleStart = frameIdx * hopSize;
        std::fill(timeScratchpad.begin(), timeScratchpad.begin() + fftSize, 0.0f);

        for (size_t i = 0; i < windowSize; ++i)
        {
            size_t paddedIdx = sampleStart + frameOffset + i;
            float sample = GetReflectSample(audioTimeline, paddedIdx);

            timeScratchpad[frameOffset + i] = sample * hannWindow[i];
        }

        fftEngine.fft(timeScratchpad.data(), realDest.data(), imagDest.data());
    }

    // Transforms a single complex gradient frame backward and overlap-adds it into the master gradient track
    void ProcessBackwardFrame(T* __restrict outGradient, size_t frameIdx,
        std::vector<float>& timeScratchpad, std::vector<float>& realSrc, std::vector<float>& imagSrc,
        double externalScale) const
    {
        size_t sampleStart = frameIdx * hopSize;
        fftEngine.ifft(timeScratchpad.data(), realSrc.data(), imagSrc.data());

        for (size_t i = 0; i < windowSize; ++i)
        {
            size_t paddedIdx = sampleStart + frameOffset + i;
            float windowedGradSample = timeScratchpad[frameOffset + i] * hannWindow[i];

            AccumulateReflectGradient(outGradient, paddedIdx, static_cast<float>(windowedGradSample * externalScale));
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

    mutable audiofft::AudioFFT fftEngine;
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
    std::vector<std::vector<T>> hannWindows;
    std::vector<float> targetReal;
    std::vector<float> targetImag;
    std::vector<float> outputReal;
    std::vector<float> outputImag;
    std::vector<float> targetFrame;
    std::vector<float> outputFrame;
    const T epsilon = static_cast<T>(1e-8);

public:
    MultiResolutionStftLossT() :
        configurations({ { 2048, 240, 1200 }, { 1024, 120, 600 }, {512, 50, 240} })
    {
        this->name = "MR-STFT";

        size_t maxFft = (configurations[0].FftSize);

        targetFrame.resize(maxFft);
        outputFrame.resize(maxFft);

        for (StftWindowConfig& config : configurations)
        {
            stftProcessors.emplace_back(
                std::make_unique<StftFrameProcessor<T>>(
                    config.FftSize,
                    config.HopSize,
                    config.WindowSize
                )
            );
        }

        size_t maxBins = (maxFft / 2) + 1;

        targetReal.resize(maxBins);
        targetImag.resize(maxBins);
        outputReal.resize(maxBins);
        outputImag.resize(maxBins);
    }

    ~MultiResolutionStftLossT() override = default;

    MultiResolutionStftLossT(const MultiResolutionStftLossT&) = delete;
    MultiResolutionStftLossT& operator=(const MultiResolutionStftLossT&) = delete;

    // Maintain move safety
    MultiResolutionStftLossT(MultiResolutionStftLossT&&) noexcept = default;
    MultiResolutionStftLossT& operator=(MultiResolutionStftLossT&&) noexcept = default;

    StftLossMetrics ComputeForwardMetricsForScale(size_t c, const T* output, const T* target, size_t outputFrames, size_t freqBins)
    {
        StftLossMetrics metrics = { 0.0, 0.0, 0.0 };

        for (size_t frame = 0; frame < outputFrames; frame++)
        {
            stftProcessors[c]->ProcessForwardFrame(target, frame, targetFrame, targetReal, targetImag);
            stftProcessors[c]->ProcessForwardFrame(output, frame, outputFrame, outputReal, outputImag);

            for (size_t k = 0; k < freqBins; k++)
            {
                float targetPower = targetReal[k] * targetReal[k] + targetImag[k] * targetImag[k];
                float outputPower = outputReal[k] * outputReal[k] + outputImag[k] * outputImag[k];

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
        double globalScale = scaleFactor / (static_cast<double>(configurations.size() * numSamples));

        for (size_t c = 0; c < configurations.size(); c++)
        {
            stftProcessors[c]->SetNumSamples(numSamples);

            size_t fftSize = configurations[c].FftSize;
            size_t outputFrames = stftProcessors[c]->GetOutputFrames();
            size_t freqBins = stftProcessors[c]->GetFreqBins();

            StftLossMetrics metrics = ComputeForwardMetricsForScale(c, output, target, outputFrames, freqBins);

            // --- PASS 2: Adjoint Backpropagation Pass ---
            double sqrtDiff = std::sqrt(metrics.FrobeniusDiff);
            double sqrtTarget = std::sqrt(metrics.FrobeniusTarget);
            double logDenom = static_cast<double>(outputFrames * freqBins);
            double denominator = sqrtDiff * sqrtTarget;
            double safe_denominator = std::max(denominator, static_cast<double>(epsilon));

            for (size_t frame = 0; frame < outputFrames; frame++)
            {
                // Rematerialize forward states seamlessly for this frame
                stftProcessors[c]->ProcessForwardFrame(target, frame, targetFrame, targetReal, targetImag);
                stftProcessors[c]->ProcessForwardFrame(output, frame, outputFrame, outputReal, outputImag);

                for (size_t k = 0; k < freqBins; k++)
                {
                    float outReal = outputReal[k];
                    float outImag = outputImag[k];
                    float tReal = targetReal[k];
                    float tImag = targetImag[k];

                    float rawTargetPower = tReal * tReal + tImag * tImag;
                    float rawOutputPower = outReal * outReal + outImag * outImag;

                    float targetMag = std::sqrt(std::max(rawTargetPower, static_cast<float>(epsilon)));
                    float outputMag = std::sqrt(std::max(rawOutputPower, static_cast<float>(epsilon)));

                    double dLossDMag = 0.0;

                    if (rawOutputPower > epsilon)
                    {
                        double dScDMag = 0.0;

                        dScDMag = (outputMag - targetMag) / safe_denominator;

                        double dLogDMag = 0.0;

                        double magDifference = static_cast<double>(outputMag - targetMag);

                        double sign = 0.0;

                        if (std::abs(magDifference) > 0)
                        {
                            sign = (magDifference > 0.0) ? 1.0 : -1.0;

                            dLogDMag = (sign / (logDenom * outputMag));
                        }

                        dLossDMag = (dScDMag + dLogDMag) * (static_cast<double>(fftSize) * 0.5);

                        targetReal[k] = static_cast<float>(dLossDMag * outReal / outputMag);
                        targetImag[k] = static_cast<float>(dLossDMag * outImag / outputMag);
                    }
                    else
                    {
                        targetReal[k] = 0.0f;
                        targetImag[k] = 0.0f;
                    }
                }

                // Process backward pass and accumulate straight into outGradient via our processor instance
                stftProcessors[c]->ProcessBackwardFrame(outGradient, frame, targetFrame, targetReal, targetImag, globalScale);
            }
        }
    }

    double GetMeanLoss(const T* __restrict output, const T* __restrict target, size_t numSamples) override
    {
        float totalMrLoss = 0.0f;

        for (size_t c = 0; c < configurations.size(); c++)
        {
            stftProcessors[c]->SetNumSamples(numSamples);

            size_t outputFrames = stftProcessors[c]->GetOutputFrames();
            size_t freqBins = stftProcessors[c]->GetFreqBins();

            StftLossMetrics metrics = ComputeForwardMetricsForScale(c, output, target, outputFrames, freqBins);

            float spectralConvergence = (metrics.FrobeniusTarget > 0.0) ? static_cast<float>(std::sqrt(metrics.FrobeniusDiff) / std::sqrt(metrics.FrobeniusTarget)) : 0.0f;
            float logMagnitudeLoss = static_cast<float>(metrics.L1LogDiff / (static_cast<double>(outputFrames) * static_cast<double>(freqBins)));

            totalMrLoss += (spectralConvergence + logMagnitudeLoss);
        }

        return totalMrLoss / static_cast<float>(configurations.size());
    }
 };
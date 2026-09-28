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
class MultiResolutionStftLossT : public LossT<T>
{
private:
    std::vector<std::unique_ptr<audiofft::AudioFFT>> fftEngines;
    std::vector<StftWindowConfig> configurations;
    std::vector<std::vector<T>> hannWindows;
    std::vector<float> targetReal;
    std::vector<float> targetImag;
    std::vector<float> outputReal;
    std::vector<float> outputImag;
    std::vector<float> targetFrame;
    std::vector<float> outputFrame;
    const T epsilon = static_cast<T>(1e-8);
    const double baseScale = 0.0005; // hard-code weight for now

    std::vector<T> GenerateHannWindow(size_t size)
    {
        std::vector<T> window(size);

        for (size_t i = 0; i < size; ++i)
        {
            window[i] = static_cast<T>(0.5) * (static_cast<T>(1.0) - std::cos(static_cast<T>(2.0) * std::numbers::pi_v<T> *i / (size - 1)));
        }

        return window;
    }

    inline float GetReflectSample(const float* signal, size_t length, size_t paddedIdx, size_t padSize)
    {
        if (paddedIdx < padSize)
        {
            return signal[padSize - paddedIdx];
        }

        size_t actualIdx = paddedIdx - padSize;

        if (actualIdx >= length)
        {
            return signal[length - 2 - (actualIdx - length)];
        }

        return signal[actualIdx];
    }

    inline void AccumulateReflectGradient(T* outGradient, size_t length, size_t paddedIdx, size_t padSize, float value)
    {
        if (paddedIdx < padSize)
        {
            size_t reflectIdx = padSize - paddedIdx;
            if (reflectIdx < length)
            {
                outGradient[reflectIdx] += static_cast<T>(value);
            }
            return;
        }
        size_t actualIdx = paddedIdx - padSize;
        if (actualIdx < length)
        {
            outGradient[actualIdx] += static_cast<T>(value);
            return;
        }
        size_t rightReflectIdx = length - 2 - (actualIdx - length);
        if (rightReflectIdx < length)
        {
            outGradient[rightReflectIdx] += static_cast<T>(value);
        }
    }

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
            hannWindows.emplace_back(GenerateHannWindow(config.WindowSize));

            auto fftEngine = std::make_unique<audiofft::AudioFFT>();
            fftEngine->init(config.FftSize);
            fftEngines.emplace_back(std::move(fftEngine));
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

    void ComputeLoss(const T* __restrict output, const T* __restrict target, T* __restrict outGradient, size_t numSamples, double scaleFactor) override
    {
        double globalScale = scaleFactor / static_cast<double>(configurations.size());

        for (size_t c = 0; c < configurations.size(); c++)
        {
            const auto& config = configurations[c];

            size_t fftSize = config.FftSize;
            size_t hopSize = config.HopSize;
            size_t windowSize = config.WindowSize;
            size_t freqBins = fftSize / 2 + 1;

            size_t padSize = fftSize / 2;
            size_t paddedLength = numSamples + 2 * padSize;
            size_t outputFrames = (paddedLength - fftSize) / hopSize + 1;

            double frobeniusDiff = 0.0;
            double frobeniusTarget = 0.0;

            size_t frameOffset = (fftSize - windowSize) / 2;
            const auto& currentHannWindow = hannWindows[c];
            auto& engine = const_cast<std::vector<std::unique_ptr<audiofft::AudioFFT>>&>(fftEngines)[c];

            // --- PASS 1: Lightweight Forward Pass to gather global scalars ---
            for (size_t frame = 0; frame < outputFrames; frame++)
            {
                size_t sampleStart = frame * hopSize;

                std::fill(targetFrame.begin(), targetFrame.begin() + fftSize, 0.0f);
                std::fill(outputFrame.begin(), outputFrame.begin() + fftSize, 0.0f);

                for (size_t i = 0; i < windowSize; ++i)
                {
                    size_t paddedIdx = sampleStart + frameOffset + i;

                    float targetSample = GetReflectSample(target, numSamples, paddedIdx, padSize);
                    float outputSample = GetReflectSample(output, numSamples, paddedIdx, padSize);

                    targetFrame[frameOffset + i] = targetSample * currentHannWindow[i];
                    outputFrame[frameOffset + i] = outputSample * currentHannWindow[i];
                }

                engine->fft(targetFrame.data(), targetReal.data(), targetImag.data());
                engine->fft(outputFrame.data(), outputReal.data(), outputImag.data());

                for (size_t k = 0; k < freqBins; k++)
                {
                    float targetPower = targetReal[k] * targetReal[k] + targetImag[k] * targetImag[k];
                    float outputPower = outputReal[k] * outputReal[k] + outputImag[k] * outputImag[k];

                    float targetMag = std::sqrt(std::max(targetPower, static_cast<float>(epsilon)));
                    float outputMag = std::sqrt(std::max(outputPower, static_cast<float>(epsilon)));

                    float magDiff = targetMag - outputMag;
                    frobeniusDiff += magDiff * magDiff;
                    frobeniusTarget += targetMag * targetMag;
                }
            }

            // --- PASS 2: Adjoint Backpropagation Pass (Zero allocations inside) ---
            double sqrtDiff = std::sqrt(frobeniusDiff);
            double sqrtTarget = std::sqrt(frobeniusTarget);
            double logDenom = static_cast<double>(outputFrames * freqBins);

            for (size_t frame = 0; frame < outputFrames; frame++)
            {
                size_t sampleStart = frame * hopSize;

                std::fill(targetFrame.begin(), targetFrame.begin() + fftSize, 0.0f);
                std::fill(outputFrame.begin(), outputFrame.begin() + fftSize, 0.0f);

                for (size_t i = 0; i < windowSize; ++i)
                {
                    size_t paddedIdx = sampleStart + frameOffset + i;

                    float targetSample = GetReflectSample(target, numSamples, paddedIdx, padSize);
                    float outputSample = GetReflectSample(output, numSamples, paddedIdx, padSize);

                    targetFrame[frameOffset + i] = targetSample * currentHannWindow[i];
                    outputFrame[frameOffset + i] = outputSample * currentHannWindow[i];
                }

                // Recompute forward FFT components to restore local spectral states on the fly
                engine->fft(targetFrame.data(), targetReal.data(), targetImag.data());
                engine->fft(outputFrame.data(), outputReal.data(), outputImag.data());

                // Read cached values into local registers then OVERWRITE targetReal and targetImag 
                // to serve as our zero-allocation complex gradient buffers before IFFT
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
                        // 1. Derivative of Spectral Convergence w.r.t outputMag
                        double dScDMag = 0.0;
                        if (sqrtTarget > 0.0 && sqrtDiff > 0.0)
                        {
                            double term1 = (outputMag - targetMag) / (sqrtDiff * sqrtTarget);
                            double term2 = (rawTargetPower > epsilon) ? ((sqrtDiff * outputMag) / (frobeniusTarget * sqrtTarget)) : 0.0;

                            dScDMag = (term1 - term2);
                        }

                        // 2. Derivative of Log Magnitude w.r.t outputMag
                        double dLogDMag = 0.0;
                        if (outputMag > 0.0)
                        {
                            double sign = (outputMag > targetMag) ? 1.0 : ((outputMag < targetMag) ? -1.0 : 0.0);
                            dLogDMag = (sign / (logDenom * outputMag));
                        }

                        dLossDMag = (dScDMag + dLogDMag) * globalScale * (static_cast<double>(fftSize) * 0.5);
                    }

                    // 3. Complex chain rule mapping directly into repurposed class workspace vectors
                    if (rawOutputPower > epsilon)
                    {
                        float outputRawMag = std::sqrt(rawOutputPower);
                        targetReal[k] = static_cast<float>(dLossDMag * (outReal / outputRawMag)); // Repurposed as Real Grad Workspace
                        targetImag[k] = static_cast<float>(dLossDMag * (outImag / outputRawMag)); // Repurposed as Imag Grad Workspace
                    }
                    else
                    {
                        targetReal[k] = 0.0f;
                        targetImag[k] = 0.0f;
                    }
                }

                // 4. Pass back through AudioFFT inverse engine 
                engine->ifft(targetFrame.data(), targetReal.data(), targetImag.data());

                // 5. Symmetric window weighting and accumulation back into master time-domain outGradient array
                for (size_t i = 0; i < windowSize; ++i)
                {
                    size_t paddedIdx = sampleStart + frameOffset + i;
                    float windowedGradSample = targetFrame[frameOffset + i] * currentHannWindow[i];

                    AccumulateReflectGradient(outGradient, numSamples, paddedIdx, padSize, windowedGradSample);
                }
            }
        }
    }

    double GetMeanLoss(const T* __restrict output, const T* __restrict target, size_t numSamples) override
    {
        float totalMrLoss = 0.0f;

        for (size_t c = 0; c < configurations.size(); c++)
        {
            const auto& config = configurations[c];

            size_t fftSize = config.FftSize;
            size_t hopSize = config.HopSize;
            size_t windowSize = config.WindowSize;
            size_t freqBins = fftSize / 2 + 1;

            size_t padSize = fftSize / 2;
            size_t paddedLength = numSamples + 2 * padSize;
            size_t outputFrames = (paddedLength - fftSize) / hopSize + 1;

            double frobeniusDiff = 0.0;
            double frobeniusTarget = 0.0;
            double l1LogDiff = 0.0;

            size_t frameOffset = (fftSize - windowSize) / 2;
            const auto& currentHannWindow = hannWindows[c];

            for (size_t frame = 0; frame < outputFrames; frame++)
            {
                size_t sampleStart = frame * hopSize;

                // Only fill up to the current active fftSize
                std::fill(targetFrame.begin(), targetFrame.begin() + fftSize, 0.0f);
                std::fill(outputFrame.begin(), outputFrame.begin() + fftSize, 0.0f);

                // Extract and window current frame on-the-fly with reflect padding
                for (size_t i = 0; i < windowSize; ++i)
                {
                    size_t paddedIdx = sampleStart + frameOffset + i;

                    float targetSample = GetReflectSample(target, numSamples, paddedIdx, padSize);
                    float outputSample = GetReflectSample(output, numSamples, paddedIdx, padSize);

                    targetFrame[frameOffset + i] = targetSample * currentHannWindow[i];
                    outputFrame[frameOffset + i] = outputSample * currentHannWindow[i];
                }

                // Compute independent FFTs using the pre-allocated workspaces restricted to active pointers
                auto& engine = const_cast<std::vector<std::unique_ptr<audiofft::AudioFFT>>&>(fftEngines)[c];

                engine->fft(targetFrame.data(), targetReal.data(), targetImag.data());
                engine->fft(outputFrame.data(), outputReal.data(), outputImag.data());

                // Frame-level reduction limited strictly to current freqBins
                for (size_t k = 0; k < freqBins; k++)
                {
                    float targetPower = targetReal[k] * targetReal[k] + targetImag[k] * targetImag[k];
                    float outputPower = outputReal[k] * outputReal[k] + outputImag[k] * outputImag[k];

                    // FIX: Replicate auraloss clamping directly on the power spectrum before the sqrt
                    float targetMag = std::sqrt(std::max(targetPower, epsilon));
                    float outputMag = std::sqrt(std::max(outputPower, epsilon));

                    // Spectral Convergence component
                    float magDiff = targetMag - outputMag;
                    frobeniusDiff += magDiff * magDiff;
                    frobeniusTarget += targetMag * targetMag;

                    // Log STFT Magnitude component
                    l1LogDiff += std::abs(std::log(targetMag) - std::log(outputMag));
                }
            }

            // Finalize loss components for this resolution scale
            float spectralConvergence = (frobeniusTarget > 0.0) ? std::sqrt(frobeniusDiff) / std::sqrt(frobeniusTarget) : 0.0f;
            float logMagnitudeLoss = static_cast<float>(l1LogDiff / (outputFrames * freqBins));

            //std::cout << "fft: " << fftSize << " spec: " << spectralConvergence << " logmag: " << logMagnitudeLoss << std::endl;

            // Auraloss sums the two sub-losses per scale
            totalMrLoss += (spectralConvergence + logMagnitudeLoss);
        }

        // Return the mean MR-STFT loss across all configurations
        return totalMrLoss / configurations.size();
    }
};
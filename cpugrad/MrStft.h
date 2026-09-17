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
    size_t fftSize;
    size_t hopSize;
};

template <typename T>
class MultiResolutionStftLossT : public LossT<T>
{
private:
    std::vector<std::unique_ptr<audiofft::AudioFFT>> fftEngines;
    std::vector<StftWindowConfig> configurations;
    const T epsilon = static_cast<T>(1e-7);

    std::vector<std::vector<T>> precalculatedWindows;

    std::vector<float> audioScratch;      // Handles both Prediction and Target audio windows sequentially
    std::vector<float> rScratch;          // Handles Real component for Pred FFT, Target FFT, and Inverse Gradient
    std::vector<float> iScratch;          // Handles Imaginary component for Pred FFT, Target FFT, and Inverse Gradient
    std::vector<float> targetMagScratch;  // Holds calculated target magnitudes, replacing rTarget and iTarget entirely

    // Resolution workspace array matching native type T
    std::vector<T> resolutionGradScratch;

    std::vector<T> GenerateHannWindow(size_t size)
    {
        std::vector<T> window(size);

        for (size_t i = 0; i < size; ++i)
        {
            window[i] = static_cast<T>(0.5) * (static_cast<T>(1.0) - std::cos(static_cast<T>(2.0) * std::numbers::pi_v<T> *i / (size - 1)));
        }

        return window;
    }

    void InitializeAndAllocate()
    {
        size_t maxFftSize = 0;
        size_t maxComplexSize = 0;

        precalculatedWindows.resize(configurations.size());
        fftEngines.reserve(configurations.size());

        for (size_t c = 0; c < configurations.size(); ++c)
        {
            size_t fftLen = configurations[c].fftSize;
            size_t complexLen = (fftLen / 2) + 1;

            maxFftSize = std::max(maxFftSize, fftLen);
            maxComplexSize = std::max(maxComplexSize, complexLen);

            precalculatedWindows[c] = GenerateHannWindow(fftLen);

            auto engine = std::make_unique<audiofft::AudioFFT>();
            engine->init(fftLen);
            fftEngines.push_back(std::move(engine));
        }

        // Allocate the collapsed memory spaces down to the maximum size bounds
        audioScratch.resize(maxFftSize);
        rScratch.resize(maxComplexSize);
        iScratch.resize(maxComplexSize);
        targetMagScratch.resize(maxComplexSize);
    }

public:
    MultiResolutionStftLossT()
    {
        this->name = "MR-STFT";
        configurations = { {2048, 512}, {1024, 256}, {512, 128} };
        InitializeAndAllocate();
    }

    MultiResolutionStftLossT(const std::vector<StftWindowConfig>& configs) :
        configurations(configs)
    {
        this->name = "MR-STFT";
        InitializeAndAllocate();
    }

    ~MultiResolutionStftLossT() override = default;


    MultiResolutionStftLossT(const MultiResolutionStftLossT&) = delete;
    MultiResolutionStftLossT& operator=(const MultiResolutionStftLossT&) = delete;

    // Maintain move safety
    MultiResolutionStftLossT(MultiResolutionStftLossT&&) noexcept = default;
    MultiResolutionStftLossT& operator=(MultiResolutionStftLossT&&) noexcept = default;


    void ComputeLoss(const T* __restrict output, const T* __restrict target, T* __restrict outGradient, size_t numSamples, double scaleFactor) override
    {
        std::memset(outGradient, 0, numSamples * sizeof(T));

        if (resolutionGradScratch.size() != numSamples)
        {
            resolutionGradScratch.resize(numSamples);
        }

        double scaledWeight = (scaleFactor / static_cast<double>(numSamples)) * 0.0005; // hard-code weight for now

        for (size_t c = 0; c < configurations.size(); ++c)
        {
            size_t fftLen = configurations[c].fftSize;
            size_t hopLen = configurations[c].hopSize;
            size_t binCount = (fftLen / 2) + 1;
            const std::vector<T>& window = precalculatedWindows[c];

            std::fill(resolutionGradScratch.begin(), resolutionGradScratch.end(), static_cast<T>(0.0));
            float frameNormalization = 1.0f / static_cast<float>(fftLen);

            for (size_t startIdx = 0; startIdx + fftLen <= numSamples; startIdx += hopLen)
            {
                // 1. Process target audio window first to get its magnitude spectrum
                for (size_t i = 0; i < fftLen; ++i)
                {
                    audioScratch[i] = static_cast<float>(target[startIdx + i] * window[i]);
                }

                fftEngines[c]->fft(audioScratch.data(), rScratch.data(), iScratch.data());

                // Cache target magnitudes directly into targetMagScratch
                for (size_t k = 0; k < binCount; ++k)
                {
                    targetMagScratch[k] = std::sqrt(rScratch[k] * rScratch[k] + iScratch[k] * iScratch[k]);
                }

                // 2. Overwrite the scratch buffers to process the predicted output audio
                for (size_t i = 0; i < fftLen; ++i)
                {
                    audioScratch[i] = static_cast<float>(output[startIdx + i] * window[i]);
                }

                // rScratch and iScratch now hold prediction frequency data
                fftEngines[c]->fft(audioScratch.data(), rScratch.data(), iScratch.data());

                // 3. Compute analytical gradients in-place right back into rScratch and iScratch
                float* __restrict rG = rScratch.data();
                float* __restrict iG = iScratch.data();
                const float* __restrict mT = targetMagScratch.data();

                for (size_t k = 0; k < binCount; ++k)
                {
                    float rP = rG[k];
                    float iP = iG[k];
                    float mP = std::sqrt(rP * rP + iP * iP);

                    if (mP < static_cast<float>(epsilon))
                    {
                        rG[k] = 0.0f;
                        iG[k] = 0.0f;
                        continue;
                    }

                    // Linear component gradient (L1 derivative)
                    float dLdMLinear = (mP > mT[k]) ? 1.0f : ((mP < mT[k]) ? -1.0f : 0.0f);

                    // Logarithmic component gradient (Log derivative)
                    float logDiff = std::log(mP + 1e-7f) - std::log(mT[k] + 1e-7f);
                    float signLog = (logDiff > 0.0f) ? 1.0f : ((logDiff < 0.0f) ? -1.0f : 0.0f);
                    float dLdMLog = signLog / (mP + 1e-7f);

                    float dLdM = 0.5f * dLdMLinear + 0.5f * dLdMLog;

                    // Overwrite memory bins in-place to prepare for IFFT step
                    rG[k] = dLdM * (rP / mP) * frameNormalization;
                    iG[k] = dLdM * (iP / mP) * frameNormalization;
                }

                // 4. Transform gradients back to time domain, overwriting audioScratch
                fftEngines[c]->ifft(audioScratch.data(), rG, iG);

                // 5. Apply backward window and overlap-add accumulate
                const float* __restrict tG = audioScratch.data();
                for (size_t i = 0; i < fftLen; ++i) {
                    resolutionGradScratch[startIdx + i] += static_cast<T>(tG[i]) * window[i];
                }
            }

            for (size_t idx = 0; idx < numSamples; ++idx) {
                outGradient[idx] += resolutionGradScratch[idx] * static_cast<T>(scaledWeight);
            }
        }
    }

    // ========================================================================
    // LossT Interface: Absolute Scalar Objective Metric Evaluation
    // ========================================================================
    double GetMeanLoss(const T* __restrict output, const T* __restrict target, size_t numSamples) override
    {
        double totalLossAccumulator = 0.0;

        for (size_t c = 0; c < configurations.size(); ++c)
        {
            size_t fftLen = configurations[c].fftSize;
            size_t hopLen = configurations[c].hopSize;
            size_t binCount = (fftLen / 2) + 1;

            const std::vector<T>& window = precalculatedWindows[c];

            double resolutionLossSum = 0.0;
            size_t totalFramesProcessed = 0;

            for (size_t startIdx = 0; startIdx + fftLen <= numSamples; startIdx += hopLen)
            {
                // Compute target spectrum magnitude
                for (size_t i = 0; i < fftLen; ++i)
                {
                    audioScratch[i] = static_cast<float>(target[startIdx + i] * window[i]);
                }

                fftEngines[c]->fft(audioScratch.data(), rScratch.data(), iScratch.data());

                for (size_t k = 0; k < binCount; ++k)
                {
                    targetMagScratch[k] = std::sqrt(rScratch[k] * rScratch[k] + iScratch[k] * iScratch[k]);
                }

                // Overwrite and compute predicted spectrum magnitude
                for (size_t i = 0; i < fftLen; ++i)
                {
                    audioScratch[i] = static_cast<float>(output[startIdx + i] * window[i]);
                }

                fftEngines[c]->fft(audioScratch.data(), rScratch.data(), iScratch.data());

                double frameLoss = 0.0;
                const float* __restrict mT = targetMagScratch.data();

                for (size_t k = 0; k < binCount; ++k)
                {
                    double mP = std::sqrt(static_cast<double>(rScratch[k] * rScratch[k] + iScratch[k] * iScratch[k]));

                    double linLoss = std::abs(mP - static_cast<double>(mT[k]));

                    double logLoss = std::abs(std::log(mP + 1e-7) - std::log(static_cast<double>(mT[k]) + 1e-7));
                    frameLoss += 0.5 * linLoss + 0.5 * logLoss;
                }
                
                resolutionLossSum += (frameLoss / static_cast<double>(fftLen));
                
                totalFramesProcessed++;
            }
            
            if (totalFramesProcessed > 0)
            {
                totalLossAccumulator += (resolutionLossSum / static_cast<double>(totalFramesProcessed));
            }
        }
        
        return totalLossAccumulator;
    }};
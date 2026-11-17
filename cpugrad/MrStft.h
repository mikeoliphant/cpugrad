#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "Loss.h"
#include "Fft.h"

struct StftWindowConfig
{
    size_t FftSize;
    size_t HopSize;
    size_t WindowSize;
};

// STFT framing (Hann window, overlap-add) for one resolution. Frames read from, and write their gradient into, an
// already reflect-padded signal (padded by GetPadSize() = FftSize / 2 on each side). Buffers are passed in by the caller:
// the FFT input/output are GetFftSize() floats and the spectrum GetFreqBins() bins, aligned to cpugrad::Fft::RequiredAlignment.
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
    size_t GetHopSize() const noexcept { return hopSize; }
    size_t GetPadSize() const noexcept { return padSize; }
    size_t GetOutputFrames() const noexcept { return outputFrames; }
    size_t GetFreqBins() const noexcept { return freqBins; }

    // Zeroes the FFT input buffer. ForwardFrame() only rewrites the window part, so this is needed once per resolution
    // (not per frame), as long as the buffer isn't used for anything else in between.
    void ClearFftInput(float* fftInput) const
    {
        std::fill(fftInput, fftInput + fftSize, 0.0f);
    }

    // Windows one frame into fftInput and transforms it into spectrum. frameSamples points at the frame's first sample
    // in the padded signal.
    void ForwardFrame(const float* __restrict frameSamples, float* __restrict fftInput, std::complex<float>* spectrum)
    {
        const float* __restrict windowSamples = frameSamples + frameOffset;
        float* __restrict windowDest = fftInput + frameOffset;

        for (size_t i = 0; i < windowSize; ++i)
        {
            windowDest[i] = windowSamples[i] * hannWindow[i];
        }

        fft.Forward(fftInput, spectrum);
    }

    // Transforms one frame's gradient spectrum back into fftOutput and overlap-adds it into the padded gradient, from the
    // frame's first sample (frameGradient). gradSpectrum is used as scratch space.
    void BackwardFrame(float* __restrict frameGradient, std::complex<float>* gradSpectrum, float* __restrict fftOutput, double externalScale)
    {
        fft.Inverse(gradSpectrum, fftOutput);

        const double scale = externalScale * fft.InverseScale();

        const float* __restrict windowResult = fftOutput + frameOffset;
        float* __restrict windowGradient = frameGradient + frameOffset;

        for (size_t i = 0; i < windowSize; ++i)
        {
            float windowedGradSample = windowResult[i] * hannWindow[i];

            windowGradient[i] += static_cast<float>(windowedGradSample * scale);
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

    // The output, target and gradient are reflect-padded once, by maxPad (half the largest FFT size). Reflect padding
    // nests, so a resolution with a smaller pad p finds its padded signal at offset maxPad - p.
    size_t maxPad = 0;
    cpugrad::FftVector<float> paddedOutput;
    cpugrad::FftVector<float> paddedTarget;
    cpugrad::FftVector<float> paddedGradient;

    // FFT working buffers, sized for the largest FFT. The input and output are separate so the input's zeros outside
    // the window survive from frame to frame.
    cpugrad::FftVector<float> fftInput;
    cpugrad::FftVector<float> fftOutput;
    cpugrad::FftVector<std::complex<float>> targetSpectrum;
    cpugrad::FftVector<std::complex<float>> outputSpectrum;
    cpugrad::FftVector<std::complex<float>> gradSpectrum;

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

        maxPad = maxFft / 2;

        fftInput.resize(maxFft);
        fftOutput.resize(maxFft);
        targetSpectrum.resize((maxFft / 2) + 1);
        outputSpectrum.resize((maxFft / 2) + 1);
        gradSpectrum.resize((maxFft / 2) + 1);
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

    void ComputeLoss(const T* __restrict output, const T* __restrict target, T* __restrict outGradient, size_t numSamples, double scaleFactor) override
    {
        // Per-resolution losses are already means (over frames x bins), so only average across resolutions
        double globalScale = scaleFactor / static_cast<double>(configurations.size());

        const size_t paddedLength = numSamples + (2 * maxPad);

        PadSignal(output, numSamples, 0, paddedLength, paddedOutput);
        PadSignal(target, numSamples, 0, paddedLength, paddedTarget);

        // The padded gradient's interior starts as the incoming gradient, so interior samples get their additions in the
        // same order as adding into outGradient directly. The pads start at zero and are folded back in at the end.
        if (paddedGradient.size() < paddedLength)
            paddedGradient.resize(paddedLength);

        std::fill(paddedGradient.begin(), paddedGradient.begin() + maxPad, 0.0f);
        std::copy(outGradient, outGradient + numSamples, paddedGradient.begin() + maxPad);
        std::fill(paddedGradient.begin() + maxPad + numSamples, paddedGradient.begin() + paddedLength, 0.0f);

        for (size_t c = 0; c < configurations.size(); c++)
        {
            StftFrameProcessor<T>& processor = *stftProcessors[c];

            processor.SetNumSamples(numSamples);

            size_t fftSize = configurations[c].FftSize;
            size_t outputFrames = processor.GetOutputFrames();
            size_t freqBins = processor.GetFreqBins();

            // The forward metrics pass only supplies the spectral convergence norms, so skip it when SC is off
            StftLossMetrics metrics = { 0.0, 0.0, 0.0 };

            if (specWeight != 0.0)
            {
                metrics = ComputeForwardMetricsForScale(c, paddedOutput.data(), paddedTarget.data(), 0, 0, outputFrames, freqBins);
            }

            // --- PASS 2: Adjoint Backpropagation Pass ---
            double sqrtDiff = std::sqrt(metrics.FrobeniusDiff);
            double sqrtTarget = std::sqrt(metrics.FrobeniusTarget);
            double logDenom = static_cast<double>(outputFrames * freqBins);
            double denominator = sqrtDiff * sqrtTarget;
            double safe_denominator = std::max(denominator, static_cast<double>(epsilon));

            // d(loss)/d|X| = scDiffScale * (|X| - |T|) + logScale * sign(|X| - |T|) / |X|, where X is the output's and T the
            // target's spectrum. Both scales include the N/2 factor for the inverse FFT's scaling.
            const float scDiffScale = static_cast<float>(specWeight * (static_cast<double>(fftSize) * 0.5) / safe_denominator);
            const float logScale = static_cast<float>(logMagWeight * (static_cast<double>(fftSize) * 0.5) / logDenom);
            const float eps = static_cast<float>(epsilon);

            processor.ClearFftInput(fftInput.data());

            for (size_t frame = 0; frame < outputFrames; frame++)
            {
                const size_t frameStart = FrameStart(c, frame, 0);

                // Rematerialize forward states seamlessly for this frame
                processor.ForwardFrame(paddedTarget.data() + frameStart, fftInput.data(), targetSpectrum.data());
                processor.ForwardFrame(paddedOutput.data() + frameStart, fftInput.data(), outputSpectrum.data());

                // Interleaved real/imaginary values. The gradient gets its own buffer, so the loop can vectorize.
                const float* __restrict tgt = reinterpret_cast<const float*>(targetSpectrum.data());
                const float* __restrict out = reinterpret_cast<const float*>(outputSpectrum.data());
                float* __restrict grad = reinterpret_cast<float*>(gradSpectrum.data());

                for (size_t k = 0; k < freqBins; k++)
                {
                    const float outReal = out[2 * k];
                    const float outImag = out[2 * k + 1];
                    const float tReal = tgt[2 * k];
                    const float tImag = tgt[2 * k + 1];

                    const float outputPower = outReal * outReal + outImag * outImag;
                    const float targetPower = tReal * tReal + tImag * tImag;

                    const float outputMag = std::sqrt(std::max(outputPower, eps));
                    const float targetMag = std::sqrt(std::max(targetPower, eps));
                    const float invOutputMag = 1.0f / outputMag;

                    const float magDiff = outputMag - targetMag;
                    const float sign = (magDiff > 0.0f) ? 1.0f : ((magDiff < 0.0f) ? -1.0f : 0.0f);

                    // Match torch clamp(power, eps) backward: no gradient flows through clamped bins
                    const float dLossDMag = (outputPower >= eps) ? ((scDiffScale * magDiff) + (logScale * sign * invOutputMag)) : 0.0f;

                    // d|X|/dX = X / |X|
                    const float gradScale = dLossDMag * invOutputMag;

                    grad[2 * k] = gradScale * outReal;
                    grad[2 * k + 1] = gradScale * outImag;
                }

                // The inverse real FFT weights DC and Nyquist by 1/N but interior bins by 2/N.
                // After the N/2 scale above, DC and Nyquist need doubling to get the full gradient.
                gradSpectrum[0].real(gradSpectrum[0].real() * 2.0f);
                gradSpectrum[freqBins - 1].real(gradSpectrum[freqBins - 1].real() * 2.0f);

                processor.BackwardFrame(paddedGradient.data() + frameStart, gradSpectrum.data(), fftOutput.data(), globalScale);
            }
        }

        UnpadGradient(numSamples, outGradient);
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
        // This part's frames at each resolution, and the stretch of the padded signal they cover between them
        std::vector<std::pair<size_t, size_t>> frameRanges(configurations.size());

        size_t spanStart = std::numeric_limits<size_t>::max();
        size_t spanEnd = 0;

        for (size_t c = 0; c < configurations.size(); c++)
        {
            stftProcessors[c]->SetNumSamples(numSamples);

            frameRanges[c] = this->PartRange(stftProcessors[c]->GetOutputFrames(), part, numParts);

            auto [firstFrame, endFrame] = frameRanges[c];

            if (firstFrame < endFrame)
            {
                spanStart = std::min(spanStart, FrameStart(c, firstFrame, 0));
                spanEnd = std::max(spanEnd, FrameStart(c, endFrame - 1, 0) + stftProcessors[c]->GetFftSize());
            }
        }

        if (spanStart >= spanEnd)
            return;	// no frames in this part

        // Only pad the stretch this part needs
        PadSignal(output, numSamples, spanStart, spanEnd, paddedOutput);
        PadSignal(target, numSamples, spanStart, spanEnd, paddedTarget);

        for (size_t c = 0; c < configurations.size(); c++)
        {
            auto [firstFrame, endFrame] = frameRanges[c];

            StftLossMetrics metrics = ComputeForwardMetricsForScale(c, paddedOutput.data(), paddedTarget.data(), spanStart, firstFrame, endFrame, stftProcessors[c]->GetFreqBins());

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

private:
    // Position of frame "frame" of resolution c in the maxPad-padded signal, relative to a buffer that starts at
    // padded position bufferStart
    size_t FrameStart(size_t c, size_t frame, size_t bufferStart) const
    {
        return (maxPad - stftProcessors[c]->GetPadSize()) + (frame * stftProcessors[c]->GetHopSize()) - bufferStart;
    }

    // Spectral metrics summed over frames [firstFrame, endFrame) of resolution c. The padded buffers start at padded
    // position bufferStart.
    StftLossMetrics ComputeForwardMetricsForScale(size_t c, const float* paddedOut, const float* paddedTgt, size_t bufferStart, size_t firstFrame, size_t endFrame, size_t freqBins)
    {
        StftLossMetrics metrics = { 0.0, 0.0, 0.0 };

        StftFrameProcessor<T>& processor = *stftProcessors[c];

        processor.ClearFftInput(fftInput.data());

        for (size_t frame = firstFrame; frame < endFrame; frame++)
        {
            const size_t frameStart = FrameStart(c, frame, bufferStart);

            processor.ForwardFrame(paddedTgt + frameStart, fftInput.data(), targetSpectrum.data());
            processor.ForwardFrame(paddedOut + frameStart, fftInput.data(), outputSpectrum.data());

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

    // Fills dest with positions [spanStart, spanEnd) of the signal reflect-padded by maxPad on each side
    // (like torch's reflect padding: the edge sample itself isn't repeated)
    void PadSignal(const T* signal, size_t numSamples, size_t spanStart, size_t spanEnd, cpugrad::FftVector<float>& dest) const
    {
        if (maxPad >= numSamples)
            throw std::invalid_argument("MR-STFT needs more samples than half its largest FFT size, for reflect padding");

        if (dest.size() < (spanEnd - spanStart))
            dest.resize(spanEnd - spanStart);

        float* out = dest.data();

        for (size_t pos = spanStart; pos < spanEnd; pos++)
        {
            size_t sampleIndex;

            if (pos < maxPad)
                sampleIndex = maxPad - pos;		// left reflection
            else if ((pos - maxPad) < numSamples)
                sampleIndex = pos - maxPad;		// the signal itself
            else
                sampleIndex = numSamples - 2 - (pos - maxPad - numSamples);	// right reflection

            *(out++) = static_cast<float>(signal[sampleIndex]);
        }
    }

    // Folds the padded gradient's reflected parts back onto the samples they mirror, and writes the result to outGradient
    void UnpadGradient(size_t numSamples, T* outGradient) const
    {
        const float* padded = paddedGradient.data();

        for (size_t i = 0; i < numSamples; i++)
        {
            outGradient[i] = static_cast<T>(padded[maxPad + i]);
        }

        // Left pad: padded position p (< maxPad) mirrors sample maxPad - p
        for (size_t p = 0; p < maxPad; p++)
        {
            outGradient[maxPad - p] += static_cast<T>(padded[p]);
        }

        // Right pad: padded position maxPad + numSamples + m mirrors sample numSamples - 2 - m
        for (size_t m = 0; m < maxPad; m++)
        {
            outGradient[numSamples - 2 - m] += static_cast<T>(padded[maxPad + numSamples + m]);
        }
    }
 };

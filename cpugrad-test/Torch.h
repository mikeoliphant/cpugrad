#pragma once

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>

#include "MrStft.h"

std::vector<float> TensorToVector(const torch::Tensor& tensor)
{
    auto cpuTensor = tensor.detach().to(torch::kCPU).to(torch::kFloat).contiguous();

    std::vector<float> vec(cpuTensor.data_ptr<float>(), cpuTensor.data_ptr<float>() + cpuTensor.numel());

    return vec;
}

// Helper to write a 1D tensor to a flat comma-separated CSV text file
void save_txt(const std::string& filename, const torch::Tensor& tensor)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    // Ensure tensor is CPU, float, and contiguous
    auto cpu_tensor = tensor.detach().to(torch::kCPU).to(torch::kFloat).contiguous();
    float* data = cpu_tensor.data_ptr<float>();
    int64_t num_elements = cpu_tensor.numel();

    for (int64_t i = 0; i < num_elements; ++i) {
        file << data[i];
        if (i < num_elements - 1) file << "\n"; // Flat sequence line by line or comma
    }
    file.close();
}

// C++ equivalent of auraloss.freq.STFTLoss
// Mathematically aligned C++ equivalent of auraloss.freq.STFTLoss
struct STFTLoss
{
    int64_t fft_size;
    int64_t hop_size;
    int64_t win_length;
    float w_sc = 1.0;
    float w_log_mag = 1.0;
    float w_lin_mag = 0.0;
    torch::Tensor window;

    STFTLoss(int64_t fft, int64_t hop, int64_t win) :
        fft_size(fft), hop_size(hop), win_length(win)
    {
        // Exact Scipy periodic window calculation formula 
        auto n = torch::arange(win_length, torch::TensorOptions().dtype(torch::kFloat));
        window = 0.5 - 0.5 * torch::cos(2.0 * M_PI * n / static_cast<double>(win_length));
    }

    void SetWeights(float sc = 1.0, float log_mag = 1.0, float lin_mag = 0.0)
    {
        w_sc = sc;
        w_log_mag = log_mag;
        w_lin_mag = lin_mag;
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y)
    {
        // 1. Maintain standard batch/channel spatial matrix tracking [Batch * Channel, Time]
        auto x_flat = x.view({ x.size(0) * x.size(1), -1 });
        auto y_flat = y.view({ y.size(0) * y.size(1), -1 });

        // 2. MANUAL PADDING CORRECTION: Replicate Python's center=True (n_fft // 2 on both sides)
        // using explicit reflection padding to ensure identical frame counts in C++
        int64_t pad_amount = fft_size / 2;
        auto x_padded = torch::reflection_pad1d(x_flat.unsqueeze(1), { pad_amount, pad_amount }).squeeze(1);
        auto y_padded = torch::reflection_pad1d(y_flat.unsqueeze(1), { pad_amount, pad_amount }).squeeze(1);

        // 3. Execute STFT with center=false since padding is handled manually
        auto x_stft = torch::stft(x_padded, fft_size, hop_size, win_length, window,
            /*center=*/false, std::string("reflect"),
            /*normalized=*/false, /*onesided=*/true, /*return_complex=*/true);

        auto y_stft = torch::stft(y_padded, fft_size, hop_size, win_length, window,
            /*center=*/false, std::string("reflect"),
            /*normalized=*/false, /*onesided=*/true, /*return_complex=*/true);

        auto x_real = torch::real(x_stft);
        auto x_imag = torch::imag(x_stft);
        auto y_real = torch::real(y_stft);
        auto y_imag = torch::imag(y_stft);

        // Ground truth epsilon matching auraloss default
        float eps = 1e-8;

        // Exact magnitude extraction loop: power clamp BEFORE square-rooting
        auto x_mag = torch::sqrt(torch::clamp((x_real.pow(2) + x_imag.pow(2)), eps));
        auto y_mag = torch::sqrt(torch::clamp((y_real.pow(2) + y_imag.pow(2)), eps));

        torch::Tensor loss = torch::zeros({}, torch::kFloat);

        // 4. Spectral Convergence (SC) Loss using true Frobenius matrix dimensions
        if (w_sc > 0.0) {
            auto norm_diff = torch::linalg_norm(y_mag - x_mag, at::nullopt, { 1, 2 });
            auto norm_target = torch::linalg_norm(y_mag, at::nullopt, { 1, 2 });

            // Replicate element-wise matrix resolution ratio mean
            auto sc_loss = (norm_diff / norm_target).mean();
            loss = loss + w_sc * sc_loss;
        }

        // 5. Log-Magnitude Loss (reduction="mean" maps directly to default l1_loss reduction)
        if (w_log_mag > 0.0) {
            auto log_x = torch::log(x_mag);
            auto log_y = torch::log(y_mag);
            auto log_mag_loss = torch::nn::functional::l1_loss(log_x, log_y);
            loss = loss + w_log_mag * log_mag_loss;
        }

        // 6. Linear Magnitude Loss
        if (w_lin_mag > 0.0) {
            auto lin_mag_loss = torch::nn::functional::l1_loss(x_mag, y_mag);
            loss = loss + w_lin_mag * lin_mag_loss;
        }

        return loss;
    }
};


// C++ equivalent of auraloss.freq.MultiResolutionSTFTLoss
struct MultiResolutionSTFTLoss
{
    std::vector<STFTLoss> stft_losses;

    MultiResolutionSTFTLoss(std::vector<StftWindowConfig> configs =
        { { 2048, 240, 1200 },
        { 1024, 120, 600 },
        { 512, 50, 240 } })
    {
        for (const auto& config : configs)
        {
            // Default configuration uses w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0
            stft_losses.emplace_back(config.FftSize, config.HopSize, config.WindowSize);
        }
    }

    void SetWeights(float sc = 1.0, float log_mag = 1.0, float lin_mag = 0.0)
    {
        for (auto& stft_loss : stft_losses)
        {
            stft_loss.SetWeights(sc, log_mag, lin_mag);
        }
    }

    auto GetLossTensor(const torch::Tensor& outputTensor, const torch::Tensor& targetTensor)
    {
        torch::Tensor total_loss = torch::zeros({}, torch::kFloat);

        for (auto& stft_loss : stft_losses)
        {
            auto resolution_loss = stft_loss.forward(outputTensor, targetTensor);
            total_loss = total_loss + resolution_loss;
        }

        auto final_loss = total_loss / static_cast<float>(stft_losses.size());

        return final_loss;
    }

    double GetMeanLoss(const torch::Tensor& outputSignal, const torch::Tensor& targetSignal)
    {
        float loss;

        auto targetTensor = targetSignal.view({ 1, 1, -1 });
        auto outputTensor = outputSignal.view({ 1, 1, -1 }).clone();

        auto lossTensor = GetLossTensor(outputTensor, targetTensor);

        loss = static_cast<double>(lossTensor.item<float>());

        return static_cast<double>(loss);
    }

    std::vector<float> GetLossGradient(const torch::Tensor& outputSignal, const torch::Tensor& targetSignal)
    {
        auto targetTensor = targetSignal.view({ 1, 1, -1 });
        auto outputTensor = outputSignal.view({ 1, 1, -1 }).clone();
        outputTensor.set_requires_grad(true);

        auto lossTensor = GetLossTensor(outputTensor, targetTensor);

        lossTensor.backward();

        auto outputGradients = outputTensor.grad().detach().flatten();

        return TensorToVector(outputGradients);
    }
};


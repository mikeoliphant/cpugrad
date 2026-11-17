#pragma once

// Real FFT with a backend chosen at compile time. cpugrad's CMake option CPUGRAD_FFT (PFFFT or AUDIOFFT) defines
// CPUGRAD_FFT_PFFFT or CPUGRAD_FFT_AUDIOFFT and compiles that backend's source.
//
// The Fft object holds only the backend's setup state. Callers provide the buffers:
//   - time domain: GetSize() floats
//   - spectrum: GetNumBins() = GetSize() / 2 + 1 interleaved complex bins, DC through Nyquist
// Both must be aligned to Fft::RequiredAlignment. FftVector<T> is a std::vector that satisfies this.

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(CPUGRAD_FFT_PFFFT)
#include "pffft.h"
#elif defined(CPUGRAD_FFT_AUDIOFFT)
#include "AudioFFT.h"
#else
#error "No FFT backend selected: define CPUGRAD_FFT_PFFFT or CPUGRAD_FFT_AUDIOFFT (set by cpugrad's CMake option CPUGRAD_FFT)"
#endif

namespace cpugrad
{
	class Fft
	{
	public:
		// PFFFT needs 16-byte aligned buffers. BatchBufferArenaT buffers (32-byte aligned) qualify.
		static constexpr size_t RequiredAlignment = 16;

		explicit Fft(size_t size) :
			size(size)
		{
#if defined(CPUGRAD_FFT_PFFFT)
			setup = pffft_new_setup(static_cast<int>(size), PFFFT_REAL);

			if (setup == nullptr)
				throw std::invalid_argument("FFT size not supported by PFFFT (needs 2^a * 3^b * 5^c, with a >= 5)");
#else
			engine.init(size);
			real.resize(GetNumBins());
			imag.resize(GetNumBins());
#endif
		}

		~Fft()
		{
#if defined(CPUGRAD_FFT_PFFFT)
			pffft_destroy_setup(setup);
#endif
		}

		Fft(const Fft&) = delete;
		Fft& operator=(const Fft&) = delete;

		size_t GetSize() const { return size; }
		size_t GetNumBins() const { return (size / 2) + 1; }

		// time (GetSize() floats) -> spectrum (GetNumBins() bins)
		void Forward(const float* time, std::complex<float>* spectrum)
		{
			assert(IsAligned(time) && IsAligned(spectrum));

			const size_t nyquist = size / 2;

#if defined(CPUGRAD_FFT_PFFFT)
			// PFFFT's ordered output is the interleaved bins 0 .. N/2 - 1, except that bin 0's imaginary slot holds the
			// (real) Nyquist bin. Unpack that into bin N/2.
			pffft_transform_ordered(setup, time, reinterpret_cast<float*>(spectrum), nullptr, PFFFT_FORWARD);

			spectrum[nyquist] = std::complex<float>(spectrum[0].imag(), 0.0f);
			spectrum[0] = std::complex<float>(spectrum[0].real(), 0.0f);
#else
			engine.fft(time, real.data(), imag.data());

			for (size_t k = 0; k <= nyquist; k++)
				spectrum[k] = std::complex<float>(real[k], imag[k]);
#endif
		}

		// spectrum (GetNumBins() bins) -> time (GetSize() floats), scaled by InverseScale() relative to the true inverse.
		// The imaginary parts of the DC and Nyquist bins are ignored. The spectrum is used as scratch space, so its
		// contents afterwards are unspecified.
		void Inverse(std::complex<float>* spectrum, float* time)
		{
			assert(IsAligned(time) && IsAligned(spectrum));

			const size_t nyquist = size / 2;

#if defined(CPUGRAD_FFT_PFFFT)
			// Pack the Nyquist bin back into bin 0's imaginary slot, as PFFFT expects
			spectrum[0] = std::complex<float>(spectrum[0].real(), spectrum[nyquist].real());

			pffft_transform_ordered(setup, reinterpret_cast<const float*>(spectrum), time, nullptr, PFFFT_BACKWARD);
#else
			for (size_t k = 0; k <= nyquist; k++)
			{
				real[k] = spectrum[k].real();
				imag[k] = spectrum[k].imag();
			}

			engine.ifft(time, real.data(), imag.data());
#endif
		}

		// Inverse() returns InverseScale() times the true inverse (PFFFT's inverse is unnormalized). Callers fold it into
		// their own scaling rather than paying for an extra pass here.
		double InverseScale() const
		{
#if defined(CPUGRAD_FFT_PFFFT)
			return 1.0 / static_cast<double>(size);
#else
			return 1.0;
#endif
		}

	private:
		static bool IsAligned(const void* ptr)
		{
			return (reinterpret_cast<std::uintptr_t>(ptr) % RequiredAlignment) == 0;
		}

		size_t size;

#if defined(CPUGRAD_FFT_PFFFT)
		PFFFT_Setup* setup = nullptr;
#else
		// AudioFFT works with separate real/imaginary arrays (and keeps its own internal work buffer)
		audiofft::AudioFFT engine;
		std::vector<float> real;
		std::vector<float> imag;
#endif
	};

	// Allocator for buffers that meet Fft::RequiredAlignment
	template <typename T, size_t Alignment>
	struct AlignedAllocator
	{
		using value_type = T;

		template <typename U>
		struct rebind
		{
			using other = AlignedAllocator<U, Alignment>;
		};

		AlignedAllocator() noexcept = default;

		template <typename U>
		AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

		T* allocate(size_t n)
		{
			return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(Alignment)));
		}

		void deallocate(T* ptr, size_t) noexcept
		{
			::operator delete(ptr, std::align_val_t(Alignment));
		}

		template <typename U>
		bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }

		template <typename U>
		bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
	};

	template <typename T>
	using FftVector = std::vector<T, AlignedAllocator<T, Fft::RequiredAlignment>>;
}

#pragma once

#include <cassert>
#include <cmath>

#include "BackpropModel.h"

namespace cpugrad
{
	// Activations take the pre-activation values as "input" in Backward, and support dOutput and dInput
	// being the same buffer (in-place backward). Forward also supports input and output being the same buffer.
	//
	// An activation with CanBackwardFromOutput = true also has BackwardFromOutput(), which computes the same
	// gradient from the activation's *output*. Layers use it to avoid storing the pre-activation values.

	template <typename T, int Channels>
	class LeakyReLUT : public BackpropModelT<T, Channels, Channels>
	{
		public:
			// The slope is positive, so output < 0 exactly when input < 0 (barring a tiny negative input whose
			// 0.01x result flushes to -0 under FTZ, where the slope choice makes no practical difference)
			static constexpr bool CanBackwardFromOutput = true;

			void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& output) override
			{
				assert(input.GetNumCols() == output.GetNumCols());

				auto outputMap = output.GetEigenMap();
				auto inputMap = input.GetEigenMapConst();

				outputMap = (inputMap.array() < T(0.0)).select(inputMap.array() * T(0.01), inputMap.array());
			}

			void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput) override
			{
				assert(input.GetNumCols() == dOutput.GetNumCols());
				assert(input.GetNumCols() == dInput.GetNumCols());

				auto inputMap = input.GetEigenMapConst();
				auto dOutputMap = dOutput.GetEigenMapConst();
				auto dInputMap = dInput.GetEigenMap();

				dInputMap = (inputMap.array() < T(0.0)).select(dOutputMap.array() * T(0.01), dOutputMap.array());
			}

			void BackwardFromOutput(const ChannelRowSpan<T, Channels>& output, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput)
			{
				Backward(output, dOutput, dInput);	// same mask, since sign(output) == sign(input)
			}
	};

	// Rational tanh approximation, identical to NAM Core's fast_tanh ("Fasttanh" in .nam configs), so a model
	// trained with it matches inference exactly. Unlike LeakyReLU it has no kink, so it avoids the level-dependent
	// stepping ("zippering") a piecewise-linear network produces as distortion fades out.
	template <typename T, int Channels>
	class FastTanhT : public BackpropModelT<T, Channels, Channels>
	{
		public:
			// The derivative needs the input, so layers must keep it
			static constexpr bool CanBackwardFromOutput = false;

			// f(x) = N(x) / D(x)
			//   N = x * (A + A|x| + (B + C|x|) x^2)
			//   D = E + (E + x^2) * |x + F x |x||
			static inline T Value(const T x)
			{
				const T ax = std::abs(x);
				const T x2 = x * x;

				return (x * (A + A * ax + (B + C * ax) * x2))
					/ (E + (E + x2) * std::abs(x + F * x * ax));
			}

			// Since x + F x|x| = x (1 + F|x|) has the sign of x, |x + F x|x|| = |x| + F x^2, giving
			//   N' = A + 2A|x| + 3B x^2 + 4C x^2 |x|
			//   D  = E + (E + x^2)(|x| + F x^2)
			//   D' = 2x(|x| + F x^2) + (E + x^2)(sign(x) + 2F x)
			// The sign(x) jump in D' at 0 is multiplied by N(0) = 0, so f' is continuous (f'(0) = A / E).
			static inline T Derivative(const T x)
			{
				const T ax = std::abs(x);
				const T x2 = x * x;
				const T sx = (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0));

				const T n = x * (A + A * ax + (B + C * ax) * x2);
				const T dn = A + T(2) * A * ax + T(3) * B * x2 + T(4) * C * x2 * ax;

				const T g = ax + F * x2;
				const T d = E + (E + x2) * g;
				const T dd = T(2) * x * g + (E + x2) * (sx + T(2) * F * x);

				return (dn * d - n * dd) / (d * d);
			}

			void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& output) override
			{
				assert(input.GetNumCols() == output.GetNumCols());

				auto outputMap = output.GetEigenMap();
				auto inputMap = input.GetEigenMapConst();

				outputMap = inputMap.unaryExpr([](T x) { return Value(x); });
			}

			void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput) override
			{
				assert(input.GetNumCols() == dOutput.GetNumCols());
				assert(input.GetNumCols() == dInput.GetNumCols());

				auto inputMap = input.GetEigenMapConst();
				auto dOutputMap = dOutput.GetEigenMapConst();
				auto dInputMap = dInput.GetEigenMap();

				// Coefficient-wise, so safe when dOutput and dInput alias
				dInputMap = inputMap.binaryExpr(dOutputMap, [](T x, T dy) { return dy * Derivative(x); });
			}

		private:
			static constexpr T A = T(2.45550750702956);
			static constexpr T B = T(0.893229853513558);
			static constexpr T C = T(0.821226666969744);
			static constexpr T E = T(2.44506634652299);
			static constexpr T F = T(0.814642734961073);
	};

} // namespace cpugrad

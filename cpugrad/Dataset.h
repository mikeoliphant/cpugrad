#pragma once

#include <random>

namespace cpugrad
{
	class DataGen
	{
		public:
			DataGen() :
				randGen(rd()),
				dis(-1.0f, 1.0f)
			{
			}

			DataGen(std::mt19937& rand) :
				randGen(rand),
				dis(-1.0f, 1.0f)
			{
			
			}

			std::vector<float> GenerateRandom(size_t numSamples)
			{
				std::vector<float> rand(numSamples);

				for (size_t i = 0; i < numSamples; i++)
					rand[i] = dis(randGen);

				return rand;
			}

			std::vector<float> GenerateSin(size_t numSamples, size_t sweepSamples)
			{
				std::vector<float> data(numSamples);

				for (size_t i = 0; i < numSamples; i++)
					data[i] = (float)std::sin(i * 0.01) * ((float)(i % sweepSamples) / (float)sweepSamples);

				return data;
			}

			std::pair<std::vector<float>, std::vector<float>> GenerateDelay(size_t delay, size_t numSamples)
			{
				auto rand = GenerateRandom(numSamples);

				std::vector<float> target(numSamples);

				for (size_t i = 0; i < numSamples; i++)
				{
					if (i < delay)
						target[i] = 0;
					else
						target[i] = rand[i - delay];
				}

				return { rand, target };
			}

			std::pair<std::vector<float>, std::vector<float>> GenerateXOR(size_t delay, size_t numSamples)
			{
				auto rand = GenerateRandom(numSamples);

				for (size_t i = 0; i < numSamples; i++)
					rand[i] = std::copysign(1.0f, rand[i]);

				std::vector<float> target(numSamples);

				for (size_t i = 0; i < numSamples; i++)
				{
					if (i < delay)
						target[i] = 0;
					else
						target[i] = -1 * rand[i] * rand[i - delay];
				}

				return { rand, target };
			}

		private:
			std::random_device rd;
			std::mt19937 randGen;
			std::uniform_real_distribution<float> dis;
	};
}
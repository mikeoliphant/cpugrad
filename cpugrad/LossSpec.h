#pragma once

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "Loss.h"
#include "MrStft.h"

// Loss configuration from a string, e.g. "mse:1,mrstft:0.0005:sc=0:logmag=1"
//
//   spec := term ("," term)*
//   term := name [":" weight] (":" key "=" value)*
//
// The weight defaults to 1. Parameters not given take their defaults. A term with weight 0 is kept in
// the spec (so it shows up when the spec is printed) but is not computed.

namespace cpugrad
{
	struct LossTerm
	{
		std::string Name;
		double Weight = 1.0;
		std::map<std::string, double> Params;	// always holds every parameter of the loss type, defaults filled in
	};

	using LossSpec = std::vector<LossTerm>;

	struct LossTypeInfo
	{
		std::string Name;
		std::string Description;
		std::vector<std::pair<std::string, double>> Params;	// name and default value, in print order
	};

	// The available loss types. When adding one, also add it to CreateLossTerm() below.
	inline const std::vector<LossTypeInfo>& GetLossTypes()
	{
		static const std::vector<LossTypeInfo> lossTypes =
		{
			{ "mse", "mean squared error", {} },
			{ "esr", "error-to-signal ratio", {} },
			{ "mrstft", "multi-resolution STFT (sc = spectral convergence weight, logmag = log-magnitude weight)", { { "sc", 0.0 }, { "logmag", 1.0 } } },
		};

		return lossTypes;
	}

	namespace LossSpecDetail
	{
		inline std::string_view Trim(std::string_view str)
		{
			while (!str.empty() && std::isspace((unsigned char)str.front()))
				str.remove_prefix(1);

			while (!str.empty() && std::isspace((unsigned char)str.back()))
				str.remove_suffix(1);

			return str;
		}

		inline std::vector<std::string_view> Split(std::string_view str, char separator)
		{
			std::vector<std::string_view> parts;

			size_t start = 0;

			while (true)
			{
				size_t pos = str.find(separator, start);

				if (pos == std::string_view::npos)
				{
					parts.push_back(Trim(str.substr(start)));

					break;
				}

				parts.push_back(Trim(str.substr(start, pos - start)));

				start = pos + 1;
			}

			return parts;
		}

		inline std::string ToLower(std::string_view str)
		{
			std::string lower(str);

			for (char& c : lower)
				c = (char)std::tolower((unsigned char)c);

			return lower;
		}

		inline double ParseNumber(std::string_view str, const std::string& what)
		{
			std::string numStr(str);

			if (numStr.empty())
				throw std::invalid_argument("missing value for " + what);

			char* end = nullptr;

			errno = 0;

			double value = std::strtod(numStr.c_str(), &end);

			if ((end != numStr.c_str() + numStr.size()) || (errno == ERANGE) || !std::isfinite(value))
				throw std::invalid_argument("invalid number \"" + numStr + "\" for " + what);

			return value;
		}

		// Shortest %g-style form that reads back as the same value (so 0.0005 prints as "0.0005", not "5e-04")
		inline std::string FormatNumber(double value)
		{
			for (int precision = 1; precision < 17; precision++)
			{
				std::string str = std::format("{:.{}g}", value, precision);

				if (std::strtod(str.c_str(), nullptr) == value)
					return str;
			}

			return std::format("{:.17g}", value);
		}

		inline std::string LossTypeNames()
		{
			std::string names;

			for (auto& type : GetLossTypes())
			{
				if (!names.empty())
					names += ", ";

				names += type.Name;
			}

			return names;
		}
	}

	// Throws std::invalid_argument with a readable message if the spec is malformed
	inline LossSpec ParseLossSpec(std::string_view specStr)
	{
		using namespace LossSpecDetail;

		LossSpec spec;

		if (Trim(specStr).empty())
			throw std::invalid_argument("loss spec is empty");

		for (std::string_view termStr : Split(specStr, ','))
		{
			if (termStr.empty())
				throw std::invalid_argument("empty term in loss spec \"" + std::string(specStr) + "\"");

			auto fields = Split(termStr, ':');

			LossTerm term;

			term.Name = ToLower(fields[0]);

			const LossTypeInfo* typeInfo = nullptr;

			for (auto& type : GetLossTypes())
			{
				if (type.Name == term.Name)
					typeInfo = &type;
			}

			if (typeInfo == nullptr)
				throw std::invalid_argument("unknown loss \"" + std::string(fields[0]) + "\" (available: " + LossTypeNames() + ")");

			for (auto& [paramName, defaultValue] : typeInfo->Params)
				term.Params[paramName] = defaultValue;

			for (size_t f = 1; f < fields.size(); f++)
			{
				size_t equals = fields[f].find('=');

				if (equals == std::string_view::npos)
				{
					if (f != 1)
						throw std::invalid_argument("\"" + std::string(fields[f]) + "\" in \"" + std::string(termStr) + "\": the weight must come straight after the loss name, parameters are key=value");

					term.Weight = ParseNumber(fields[f], "the weight of " + term.Name);

					if (term.Weight < 0)
						throw std::invalid_argument("negative weight for " + term.Name);

					continue;
				}

				std::string key = ToLower(Trim(fields[f].substr(0, equals)));

				if (!term.Params.contains(key))
				{
					std::string allowed;

					for (auto& [paramName, defaultValue] : typeInfo->Params)
						allowed += (allowed.empty() ? "" : ", ") + paramName;

					throw std::invalid_argument("unknown parameter \"" + key + "\" for " + term.Name + (allowed.empty() ? " (it takes no parameters)" : " (available: " + allowed + ")"));
				}

				term.Params[key] = ParseNumber(Trim(fields[f].substr(equals + 1)), term.Name + " parameter " + key);
			}

			for (auto& other : spec)
			{
				if (other.Name == term.Name)
					throw std::invalid_argument("loss \"" + term.Name + "\" is given more than once");
			}

			spec.push_back(std::move(term));
		}

		bool anyActive = false;

		for (auto& term : spec)
		{
			if (term.Weight > 0)
				anyActive = true;
		}

		if (!anyActive)
			throw std::invalid_argument("all loss weights are zero");

		return spec;
	}

	// Canonical form with every parameter shown, e.g. "mse:1,mrstft:0.0005:sc=0:logmag=1". It parses back to the same spec.
	inline std::string FormatLossSpec(const LossSpec& spec)
	{
		std::string str;

		for (auto& term : spec)
		{
			if (!str.empty())
				str += ",";

			str += term.Name + ":" + LossSpecDetail::FormatNumber(term.Weight);

			const LossTypeInfo* typeInfo = nullptr;

			for (auto& type : GetLossTypes())
			{
				if (type.Name == term.Name)
					typeInfo = &type;
			}

			// Print parameters in the registry's order rather than alphabetically
			if (typeInfo != nullptr)
			{
				for (auto& [paramName, defaultValue] : typeInfo->Params)
				{
					auto it = term.Params.find(paramName);

					str += ":" + paramName + "=" + LossSpecDetail::FormatNumber((it != term.Params.end()) ? it->second : defaultValue);
				}
			}
		}

		return str;
	}

	template <typename T>
	std::unique_ptr<LossT<T>> CreateLossTerm(const LossTerm& term)
	{
		auto param = [&](const std::string& name)
		{
			return term.Params.at(name);
		};

		if (term.Name == "mse")
			return std::make_unique<MSELossT<T>>();

		if (term.Name == "esr")
			return std::make_unique<ESRLossT<T>>();

		if (term.Name == "mrstft")
		{
			auto loss = std::make_unique<MultiResolutionStftLossT<T>>();

			loss->SetWeights(param("sc"), param("logmag"));

			return loss;
		}

		throw std::invalid_argument("no implementation for loss \"" + term.Name + "\"");
	}

	// Builds a fresh loss instance. Each trainer worker needs its own, since losses keep scratch state.
	template <typename T>
	std::unique_ptr<LossT<T>> CreateLoss(const LossSpec& spec)
	{
		auto joint = std::make_unique<JointLossT<T>>();

		for (auto& term : spec)
		{
			if (term.Weight > 0)
				joint->AddLoss(CreateLossTerm<T>(term), term.Weight);
		}

		return joint;
	}
}

#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace RocketSim {

enum class GpuAllocationKind { DEVICE, PINNED_HOST };

class GpuAllocationError : public std::runtime_error {
public:
	GpuAllocationError(GpuAllocationKind kind, const char* buffer, size_t bytes, const char* cause)
		: std::runtime_error(std::string("RocketSim CUDA allocation failed: ") + buffer + " (" +
			std::to_string(bytes) + " bytes): " + cause), kind(kind) {}

	GpuAllocationKind kind;
};

}

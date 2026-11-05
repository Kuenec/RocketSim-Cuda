#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rsc/math/glibc_atan2_hypot.hpp"
#include "rsc/math/libm.hpp"

namespace {

enum class UnaryFn { Sin, Cos, Acos };

enum class BinaryFn { Atan2f, Atan2, Hypot };

constexpr uint64_t k_chunk = 1ULL << 24;

void check_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", what,
                 cudaGetErrorString(err));
    std::exit(2);
  }
}

__global__ void unary_kernel(UnaryFn fn, uint64_t base, uint64_t count,
                             uint32_t *out) {
  uint64_t idx = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (idx >= count) {
    return;
  }
  float x = rsc::u32_to_f32(static_cast<uint32_t>(base + idx));
  float r = 0.0f;
  switch (fn) {
  case UnaryFn::Sin:
    r = rsc::sinf_glibc(x);
    break;
  case UnaryFn::Cos:
    r = rsc::cosf_glibc(x);
    break;
  case UnaryFn::Acos:
    r = rsc::acosf_glibc(x);
    break;
  }
  out[idx] = rsc::f32_to_u32(r);
}

__global__ void binary_kernel(BinaryFn fn, const uint32_t *ys,
                              const uint32_t *xs, uint64_t count,
                              uint64_t *out) {
  uint64_t idx = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (idx >= count) {
    return;
  }
  float y = rsc::u32_to_f32(ys[idx]);
  float x = rsc::u32_to_f32(xs[idx]);
  switch (fn) {
  case BinaryFn::Atan2f:
    out[idx] = rsc::f32_to_u32(rsc::atan2f_glibc(y, x));
    break;
  case BinaryFn::Atan2:
    out[idx] = rsc::f64_to_u64(
        rsc::atan2_glibc(static_cast<double>(y), static_cast<double>(x)));
    break;
  case BinaryFn::Hypot:
    out[idx] = rsc::f64_to_u64(
        rsc::hypot_glibc(static_cast<double>(y), static_cast<double>(x)));
    break;
  }
}

bool is_nan_bits32(uint32_t u) { return (u & 0x7fffffffu) > 0x7f800000u; }

bool is_nan_bits64(uint64_t u) {
  return (u & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
}

uint32_t host_ref_unary(UnaryFn fn, float x) {
  switch (fn) {
  case UnaryFn::Sin:
    return rsc::f32_to_u32(::sinf(x));
  case UnaryFn::Cos:
    return rsc::f32_to_u32(::cosf(x));
  case UnaryFn::Acos:
    return rsc::f32_to_u32(::acosf(x));
  }
  return 0;
}

uint32_t host_port_unary(UnaryFn fn, float x) {
  switch (fn) {
  case UnaryFn::Sin:
    return rsc::f32_to_u32(rsc::sinf_glibc(x));
  case UnaryFn::Cos:
    return rsc::f32_to_u32(rsc::cosf_glibc(x));
  case UnaryFn::Acos:
    return rsc::f32_to_u32(rsc::acosf_glibc(x));
  }
  return 0;
}

uint64_t host_ref_binary(BinaryFn fn, float y, float x) {
  switch (fn) {
  case BinaryFn::Atan2f:
    return rsc::f32_to_u32(::atan2f(y, x));
  case BinaryFn::Atan2:
    return rsc::f64_to_u64(
        ::atan2(static_cast<double>(y), static_cast<double>(x)));
  case BinaryFn::Hypot:
    return rsc::f64_to_u64(
        ::hypot(static_cast<double>(y), static_cast<double>(x)));
  }
  return 0;
}

uint64_t host_port_binary(BinaryFn fn, float y, float x) {
  switch (fn) {
  case BinaryFn::Atan2f:
    return rsc::f32_to_u32(rsc::atan2f_glibc(y, x));
  case BinaryFn::Atan2:
    return rsc::f64_to_u64(
        rsc::atan2_glibc(static_cast<double>(y), static_cast<double>(x)));
  case BinaryFn::Hypot:
    return rsc::f64_to_u64(
        rsc::hypot_glibc(static_cast<double>(y), static_cast<double>(x)));
  }
  return 0;
}

struct Tally {
  uint64_t tested = 0;
  uint64_t device_mismatch = 0;
  uint64_t host_mismatch = 0;
  uint64_t nan_payload_diff = 0;
};

bool same_bits(uint64_t a, uint64_t b, bool wide, Tally &tally) {
  if (a == b) {
    return true;
  }
  bool both_nan = wide ? (is_nan_bits64(a) && is_nan_bits64(b))
                       : (is_nan_bits32(static_cast<uint32_t>(a)) &&
                          is_nan_bits32(static_cast<uint32_t>(b)));
  if (both_nan) {
    ++tally.nan_payload_diff;
    return true;
  }
  return false;
}

bool report(const char *name, const Tally &tally) {
  bool pass = tally.device_mismatch == 0 && tally.host_mismatch == 0;
  std::printf("%-8s %-4s tested=%llu device_mismatch=%llu host_mismatch=%llu "
              "nan_payload_diff=%llu\n",
              name, pass ? "PASS" : "FAIL",
              static_cast<unsigned long long>(tally.tested),
              static_cast<unsigned long long>(tally.device_mismatch),
              static_cast<unsigned long long>(tally.host_mismatch),
              static_cast<unsigned long long>(tally.nan_payload_diff));
  return pass;
}

bool check_unary(UnaryFn fn, const char *name, uint64_t limit) {
  uint32_t *d_out = nullptr;
  check_cuda(cudaMalloc(&d_out, k_chunk * sizeof(uint32_t)), "cudaMalloc");
  std::vector<uint32_t> h_out(k_chunk);
  Tally tally;
  uint64_t printed = 0;
  for (uint64_t base = 0; base < limit; base += k_chunk) {
    uint64_t count = std::min(k_chunk, limit - base);
    unary_kernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        fn, base, count, d_out);
    check_cuda(cudaGetLastError(), "unary_kernel launch");
    check_cuda(cudaMemcpy(h_out.data(), d_out, count * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               "unary copy");
    for (uint64_t i = 0; i < count; ++i) {
      uint32_t bits = static_cast<uint32_t>(base + i);
      float x = rsc::u32_to_f32(bits);
      uint32_t ref = host_ref_unary(fn, x);
      uint32_t port = host_port_unary(fn, x);
      bool dev_ok = same_bits(h_out[i], ref, false, tally);
      bool host_ok = same_bits(port, ref, false, tally);
      tally.device_mismatch += dev_ok ? 0 : 1;
      tally.host_mismatch += host_ok ? 0 : 1;
      if ((!dev_ok || !host_ok) && printed < 8) {
        ++printed;
        std::printf("  %s(0x%08x) ref=0x%08x device=0x%08x host=0x%08x\n", name,
                    bits, ref, h_out[i], port);
      }
    }
    tally.tested += count;
  }
  check_cuda(cudaFree(d_out), "cudaFree");
  return report(name, tally);
}

uint64_t splitmix(uint64_t &state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

uint32_t sample_input(uint64_t &state, int mode) {
  uint64_t r = splitmix(state);
  switch (mode) {
  case 0:
    return static_cast<uint32_t>(r);
  case 1: {
    float f = static_cast<float>(static_cast<int64_t>(r >> 11)) *
              (1.0f / 4503599627370496.0f);
    return rsc::f32_to_u32(f * 2.0f - 1.0f);
  }
  case 2: {
    uint32_t exp = 100u + static_cast<uint32_t>((r >> 40) % 30u);
    uint32_t mant = static_cast<uint32_t>(r) & 0x7fffffu;
    uint32_t sign = static_cast<uint32_t>(r >> 63) << 31;
    return sign | (exp << 23) | mant;
  }
  default: {
    float base =
        static_cast<float>(static_cast<int64_t>((r >> 40) % 2001) - 1000) /
        1000.0f;
    uint32_t bits = rsc::f32_to_u32(base);
    int32_t delta = static_cast<int32_t>(r & 0xff) - 128;
    return bits + static_cast<uint32_t>(delta);
  }
  }
}

bool check_binary(BinaryFn fn, const char *name, uint64_t samples,
                  uint64_t seed) {
  uint32_t *d_y = nullptr;
  uint32_t *d_x = nullptr;
  uint64_t *d_out = nullptr;
  check_cuda(cudaMalloc(&d_y, k_chunk * sizeof(uint32_t)), "cudaMalloc");
  check_cuda(cudaMalloc(&d_x, k_chunk * sizeof(uint32_t)), "cudaMalloc");
  check_cuda(cudaMalloc(&d_out, k_chunk * sizeof(uint64_t)), "cudaMalloc");
  std::vector<uint32_t> ys(k_chunk);
  std::vector<uint32_t> xs(k_chunk);
  std::vector<uint64_t> h_out(k_chunk);
  bool wide = fn != BinaryFn::Atan2f;
  Tally tally;
  uint64_t printed = 0;
  uint64_t state = seed;
  for (uint64_t done = 0; done < samples; done += k_chunk) {
    uint64_t count = std::min(k_chunk, samples - done);
    int mode = static_cast<int>((done / k_chunk) % 4);
    for (uint64_t i = 0; i < count; ++i) {
      ys[i] = sample_input(state, mode);
      xs[i] = sample_input(state, mode);
    }
    check_cuda(cudaMemcpy(d_y, ys.data(), count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "copy ys");
    check_cuda(cudaMemcpy(d_x, xs.data(), count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "copy xs");
    binary_kernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        fn, d_y, d_x, count, d_out);
    check_cuda(cudaGetLastError(), "binary_kernel launch");
    check_cuda(cudaMemcpy(h_out.data(), d_out, count * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "binary copy");
    for (uint64_t i = 0; i < count; ++i) {
      float y = rsc::u32_to_f32(ys[i]);
      float x = rsc::u32_to_f32(xs[i]);
      uint64_t ref = host_ref_binary(fn, y, x);
      uint64_t port = host_port_binary(fn, y, x);
      bool dev_ok = same_bits(h_out[i], ref, wide, tally);
      bool host_ok = same_bits(port, ref, wide, tally);
      tally.device_mismatch += dev_ok ? 0 : 1;
      tally.host_mismatch += host_ok ? 0 : 1;
      if ((!dev_ok || !host_ok) && printed < 8) {
        ++printed;
        std::printf("  %s(y=0x%08x, x=0x%08x) ref=0x%016llx device=0x%016llx "
                    "host=0x%016llx\n",
                    name, ys[i], xs[i], static_cast<unsigned long long>(ref),
                    static_cast<unsigned long long>(h_out[i]),
                    static_cast<unsigned long long>(port));
      }
    }
    tally.tested += count;
  }
  check_cuda(cudaFree(d_y), "cudaFree");
  check_cuda(cudaFree(d_x), "cudaFree");
  check_cuda(cudaFree(d_out), "cudaFree");
  return report(name, tally);
}

bool check_sincos_consistency(uint64_t limit) {
  Tally tally;
  for (uint64_t bits = 0; bits < limit; ++bits) {
    float x = rsc::u32_to_f32(static_cast<uint32_t>(bits));
    float s = 0.0f;
    float c = 0.0f;
    ::sincosf(x, &s, &c);
    bool ok_s =
        same_bits(rsc::f32_to_u32(s), rsc::f32_to_u32(::sinf(x)), false, tally);
    bool ok_c =
        same_bits(rsc::f32_to_u32(c), rsc::f32_to_u32(::cosf(x)), false, tally);
    tally.host_mismatch += (ok_s && ok_c) ? 0 : 1;
    ++tally.tested;
  }
  return report("sincosf", tally);
}

}

int main(int argc, char **argv) {
  uint64_t unary_limit = 1ULL << 32;
  uint64_t binary_samples = 1ULL << 30;
  if (argc > 1 && std::string(argv[1]) == "--quick") {
    unary_limit = 1ULL << 26;
    binary_samples = 1ULL << 24;
  }
  bool pass = true;
  pass &= check_unary(UnaryFn::Sin, "sinf", unary_limit);
  pass &= check_unary(UnaryFn::Cos, "cosf", unary_limit);
  pass &= check_unary(UnaryFn::Acos, "acosf", unary_limit);
  pass &= check_sincos_consistency(unary_limit);
  pass &= check_binary(BinaryFn::Atan2f, "atan2f", binary_samples, 1);
  pass &= check_binary(BinaryFn::Atan2, "atan2", binary_samples, 2);
  pass &= check_binary(BinaryFn::Hypot, "hypot", binary_samples, 3);
  std::printf("LIBM %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

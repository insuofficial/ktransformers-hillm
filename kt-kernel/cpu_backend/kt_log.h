/**
 * @Description  : Env-gated setup logging for kt-kernel.
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Startup chatter -- worker-pool topology, per-layer MoE construction banners -- is useful
 * when bringing a new machine up and pure noise everywhere else: a 60-layer model prints a
 * couple hundred lines before the first token, and it lands on stdout ahead of anything the
 * caller wants to see. Route those prints through KT_LOG_INFO so they stay off by default.
 *
 * Set KT_KERNEL_VERBOSE=1 (or the existing KT_KERNEL_DEBUG=1) to restore the original output.
 *
 * Error and failure paths are deliberately NOT routed through this macro -- they print
 * unconditionally, as before.
 */
#ifndef CPUINFER_CPU_BACKEND_KT_LOG_H
#define CPUINFER_CPU_BACKEND_KT_LOG_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline bool kt_log_verbose() {
  static const bool enabled = [] {
    auto truthy = [](const char* name) {
      const char* v = std::getenv(name);
      return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    };
    return truthy("KT_KERNEL_VERBOSE") || truthy("KT_KERNEL_DEBUG");
  }();
  return enabled;
}

#define KT_LOG_INFO(...)        \
  do {                          \
    if (kt_log_verbose()) {     \
      std::printf(__VA_ARGS__); \
    }                           \
  } while (0)

#endif  // CPUINFER_CPU_BACKEND_KT_LOG_H

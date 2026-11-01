/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#elif KTRANSFORMERS_USE_ASCEND_NPU
#include "vendors/ascend_npu.h"
#endif

#include "./vendors/vendor.h"
#include "kt_log.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    KT_LOG_INFO("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    KT_LOG_INFO("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    KT_LOG_INFO("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    KT_LOG_INFO("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    delete backend_;
    delete task_queue_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }

  // Submit TWO tasks in ONE cudaLaunchHostFunc. A caller that always issues a pair (HiLLM's shard
  // gate_up + CPU forward) then pays ONE GPU->CPU->GPU host-node round-trip instead of two. The
  // callback runs f1 then f2 in order, so the tasks enter the FIFO queue in exactly the order two
  // back-to-back submit_with_cuda_stream calls would -- identical semantics (e.g. a later
  // sync_with_cuda_stream(allow_n_pending=1) still drains f1 and leaves f2), only cheaper. Each
  // args[0] is patched to `this` at submit time, as submit_with_cuda_stream does. PairArgs is
  // leaked like SyncArgs: under CUDA graph capture the callback re-runs every replay with the same
  // args pointer, so it must NOT be freed in the callback.
  struct PairArgs {
    void (*f1)(void*);
    void* a1;
    void (*f2)(void*);
    void* a2;
  };
  static void submit_pair_(void* pair_args) {
    PairArgs* pa = (PairArgs*)pair_args;
    pa->f1(pa->a1);
    pa->f2(pa->a2);
  }
  void submit_pair_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params1,
                                    std::pair<intptr_t, intptr_t> params2) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    void* a1 = (void*)params1.second;
    void* a2 = (void*)params2.second;
    *((CPUInfer**)a1) = this;
    *((CPUInfer**)a2) = this;
    PairArgs* pa = new PairArgs{(void (*)(void*))params1.first, a1, (void (*)(void*))params2.first, a2};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&submit_pair_, (void*)pa);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    sync_(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
#endif
  }
#endif
 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
};

#endif

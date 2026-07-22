# Base classes for MoE CPU inference operations
# SPDX-License-Identifier: Apache-2.0

"""
Base infrastructure for CPU-based MoE inference.

This module contains base classes and utilities shared across all backend implementations.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import os
import ctypes

from kt_kernel import kt_kernel_ext


def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
) -> torch.Tensor:
    """
    Generate GPU experts masks based on activation frequency.

    Selects the top `num_gpu_experts` experts with highest activation frequency
    across all layers to be placed on GPU.

    Args:
        activation_freq: Activation frequency table of shape (num_layers, num_experts).
                         Higher values indicate more frequently activated experts.
        num_gpu_experts: Total number of experts to place on GPU across all layers.

    Returns:
        gpu_experts_masks: Boolean mask of shape (num_layers, num_experts) on CPU.
                           True means the expert should be on GPU.

    Example:
        >>> activation_freq = torch.tensor([
        ...     [0.1, 0.5, 0.3, 0.8],  # layer 0
        ...     [0.2, 0.4, 0.9, 0.1],  # layer 1
        ... ])
        >>> masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)
        >>> # Top 3: layer0-expert3 (0.8), layer1-expert2 (0.9), layer0-expert1 (0.5)
        >>> masks
        tensor([[False,  True, False,  True],
                [False, False,  True, False]])
    """
    num_layers, num_experts_per_layer = activation_freq.shape
    total_experts = num_layers * num_experts_per_layer

    # Clamp num_gpu_experts to valid range
    num_gpu_experts = min(num_gpu_experts, total_experts)
    num_gpu_experts = max(num_gpu_experts, 0)

    if num_gpu_experts == 0:
        return torch.zeros(num_layers, num_experts_per_layer, dtype=torch.bool, device="cpu")

    # Flatten and find top-k indices
    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(flat_freq, k=num_gpu_experts, largest=True, sorted=False)

    # Create mask
    gpu_experts_masks = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    gpu_experts_masks[top_indices] = True

    # Reshape to (num_layers, num_experts)
    gpu_experts_masks = gpu_experts_masks.view(num_layers, num_experts_per_layer)

    return gpu_experts_masks


class KExpertsCPUBuffer:
    """
    CPU buffer management for expert computation.

    Manages pinned memory buffers for efficient GPU-CPU data transfer.
    """

    capture_bs: List = list()
    capture_buffers: Dict = dict()
    temp_bs: int = 0
    temp_key: tuple = tuple()
    temp_buffer: tuple = tuple()
    buffer_depth: int = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]
        buffer_key = (
            batch_size,
            hidden_size,
            hidden_states.device.type,
            hidden_states.device.index,
            hidden_states.dtype,
            int(num_experts_per_tok),
        )

        pin_memory = True

        if buffer_key in cls.capture_buffers:
            return cls.capture_buffers[buffer_key]
        if buffer_key == cls.temp_key:
            return cls.temp_buffer

        input_tensor_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        immediate_experts_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        deferred_experts_ids_cpu = [
            torch.full((batch_size, num_experts_per_tok), -1, device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        bsz_tensor_cpu = [
            torch.full((1,), batch_size, device="cpu", dtype=torch.int32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        )
        if batch_size in cls.capture_bs:
            cls.capture_buffers[buffer_key] = cur_buffer
        cls.temp_bs = batch_size
        cls.temp_key = buffer_key
        cls.temp_buffer = cur_buffer
        return cur_buffer


class _MoEBase:
    """
    Shared base class for inference and SFT MoE wrappers.

    Provides:
    - CPUInfer singleton management
    - Basic configuration validation

    This class is shared between BaseMoEWrapper (inference) and BaseSFTMoEWrapper (SFT).
    """

    _cpu_infer_instance = None

    @classmethod
    def _get_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
    ):
        """
        Get or create the CPUInfer singleton instance.

        Args:
            cpuinfer_threads: Total number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            numa_nodes: Explicit list of NUMA node IDs. If None, defaults to sequential.

        Returns:
            CPUInfer singleton instance
        """
        if cls._cpu_infer_instance is None:
            worker_config = kt_kernel_ext.WorkerPoolConfig()

            if numa_nodes is not None:
                if len(numa_nodes) != threadpool_count:
                    raise ValueError(
                        f"numa_nodes length ({len(numa_nodes)}) must match "
                        f"threadpool_count ({threadpool_count})"
                    )
                subpool_numa_map = list(numa_nodes)
            else:
                subpool_numa_map = list(range(threadpool_count))
            subpool_thread_count = [
                cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0)
                for i in range(threadpool_count)
            ]

            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            cls._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)

        return cls._cpu_infer_instance

    @staticmethod
    def _validate_base_config(
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
    ) -> None:
        """
        Validate basic configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if moe_intermediate_size <= 0:
            raise ValueError(f"moe_intermediate_size must be positive, got {moe_intermediate_size}")
        if num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be positive, got {num_experts_per_tok}")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed " f"num_experts ({num_experts})"
            )


class BaseMoEWrapper(_MoEBase, ABC):
    """
    Base class for MoE CPU inference operations.
    Provides common functionality for all backend implementations.
    """

    _layer_has_pending_deferred: Dict[int, bool] = {}

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "AMXINT4",
        numa_nodes: Optional[List[int]] = None,
        swiglu_limit: float = 0.0,
    ):
        """
        Initialize base MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            gpu_experts_mask: Boolean mask indicating which experts are on GPU.
                              Shape: [num_experts], dtype: torch.bool.
                              mask[i] = True means expert i is on GPU.
                              If None, all experts are on CPU.
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer on this layer. Defaults to 0 (no defer).
            method: Backend method string
            numa_nodes: Explicit list of NUMA node IDs for subpool mapping.
                        If None, defaults to [0, 1, ..., threadpool_count-1].
        """
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        # Process gpu_experts_mask: convert to bool tensor on CPU, pinned memory for async copy
        # This mask is shared between C and Python (C uses uint8_t*), both can read/write it
        if gpu_experts_mask is None:
            # No GPU experts - all experts on CPU
            self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
        else:
            # Create a new pinned tensor and copy data into it
            self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
            self.gpu_experts_mask.copy_(gpu_experts_mask)

        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())

        # GPU copy for mask operations in forward pass (e.g., mask_cpu_expert_ids)
        # This will be lazily initialized when needed
        self._gpu_experts_mask_gpu: Optional[torch.Tensor] = None
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save
        self.max_deferred_experts_per_token = (
            int(max_deferred_experts_per_token) if max_deferred_experts_per_token is not None else 0
        )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        self.method = method
        # V4-Flash 2604B SwiGLU clamp limit; 0.0 = disabled. NativeMoEWrapper
        # (MXFP4 path) reads this in load_weights() and writes it into
        # MOEConfig.swiglu_limit. Other backends ignore it (C++ act_fn skips
        # the clamp branch when limit==0). Origin: kt-sglang 耦合.
        self.swiglu_limit = float(swiglu_limit)

        # Initialize CPU inference engine (singleton via shared base class)
        self.cpu_infer = self._get_cpu_infer(cpuinfer_threads, threadpool_count, numa_nodes=numa_nodes)

        # Backend-specific initialization happens in subclasses
        self.moe = None

    @abstractmethod
    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        """
        Load and quantize weights from BF16/FP16 tensors (online quantization).

        Args:
            gate_proj: Gate projection weights [num_experts, intermediate_size, hidden_size]
            up_proj: Up projection weights [num_experts, intermediate_size, hidden_size]
            down_proj: Down projection weights [num_experts, hidden_size, intermediate_size]
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    def select_deferred_experts(
        self,
        expert_ids: torch.Tensor,
        expert_scores: torch.Tensor,
        protected_k: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, topk = expert_ids.shape
        device = expert_ids.device

        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:
            deferred_ids = expert_ids.clone()
            immediate_ids = torch.full_like(expert_ids, -1)
            return immediate_ids, deferred_ids

        topk_result = torch.topk(expert_scores, k=protected_k, dim=-1, largest=True, sorted=False)
        protected_indices = topk_result.indices
        protected_ids = torch.gather(expert_ids, -1, protected_indices)

        protected_flag = torch.zeros((self.num_experts,), dtype=torch.int32, device=device)
        protected_flag.scatter_(0, protected_ids.reshape(-1), 1)

        protected_mask_flat = torch.gather(protected_flag, 0, expert_ids.reshape(-1)).ne(0)
        protected_mask = protected_mask_flat.view(batch, topk)

        immediate_ids = expert_ids.clone().masked_fill(~protected_mask, -1)
        deferred_ids = expert_ids.clone().masked_fill(protected_mask, -1)

        return immediate_ids, deferred_ids

    def stage_inputs(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """D2H this layer's hidden states + routing into the shared pinned slot buffers.

        kt itself never copies -- its tasks take raw HOST pointers -- so this is where the
        inputs land. submit_forward (CPU tier) and submit_gate_up (shard tier) read the
        IDENTICAL inputs, so a caller running both should stage once and hand the returned
        handle to each, instead of letting them each repeat the D2H. Returns the slot bundle
        the two submits consume."""
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)
        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth

        topk_ids_long = topk_ids.to(torch.long)
        immediate_ids: torch.Tensor
        deferred_ids: Optional[torch.Tensor]
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token
            immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)

        return (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            current_slot,
            deferred_ids,
        )

    def submit_gate_up(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
        mask,
        output: torch.Tensor,
        staged=None,
    ):
        """Submit a gate_up-only task to CPU (non-blocking) -- HiLLM's shard tier.

        Same inputs as submit_forward: kt gathers the routed tokens of the selected experts
        itself, runs gate/up + activation, and writes the packed intermediate
        [total_tokens, intermediate_size] bf16 into `output` (ordered by ascending expert id;
        tokens within an expert keep the routing order). The down projection is the caller's
        to run on the GPU.

        `mask` is this tier's own pinned uint8 [num_experts] expert mask (1 = skip). kt reads
        it AT FORWARD TIME instead of consulting its shared gpu_experts_mask, so the shard
        and CPU tiers select disjoint experts without contending -- and the selection is
        graph-capturable. `staged` reuses a previous stage_inputs() (see submit_forward).
        """
        if staged is None:
            staged = self.stage_inputs(hidden_states, topk_ids, topk_weights)
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            weights_cpu,
            _output_cpu,
            _bsz_tensor_cpu,
            current_slot,
            _deferred_ids,
        ) = staged

        ids = immediate_experts_ids_cpu[current_slot]
        task = self.moe.gate_up_task(
            ids.size(0),          # qlen
            ids.size(-1),         # k
            ids.data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output.data_ptr(),
            mask.data_ptr(),      # only here, at the native boundary, does a tensor become an address
        )
        self.cpu_infer.submit_with_cuda_stream(cuda_stream, task)

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
        mask=None,
        staged=None,
    ):
        """
        Submit forward inference task to CPU (non-blocking).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization
            mask: OPTIONAL pinned uint8 [num_experts] CPU-tier mask tensor, already in
                gpu_experts_mask format (1 = skip, 0 = compute on the CPU tier) -- e.g. the
                scheduler host node's pinned output. Fixed length, so no count travels with
                it. When given, the forward host node adopts it (memcpy into
                gpu_experts_mask) instead of a host-side mask flip; it is read at forward
                time, so the selection is capturable.
            staged: OPTIONAL handle from a previous stage_inputs() for this same layer/step
                -- reuses that D2H instead of repeating it (see stage_inputs).
        """
        if staged is None:
            staged = self.stage_inputs(hidden_states, topk_ids, topk_weights)
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            current_slot,
            deferred_ids,
        ) = staged

        next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        args = (
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
        )
        if mask is not None:
            # The forward host node adopts the caller's pinned CPU-tier mask (capturable) --
            # no host-side mask flip. Only here, at the native boundary, does the tensor
            # become a raw address.
            task = self.moe.forward_task(*args, mask.data_ptr())
        else:
            task = self.moe.forward_task(*args)
        self.cpu_infer.submit_with_cuda_stream(cuda_stream, task)

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(
                cuda_stream,
                self.moe.forward_task(
                    bsz_slot_tensor.data_ptr(),
                    deferred_experts_ids_cpu[current_slot].size(-1),
                    deferred_experts_ids_cpu[current_slot].data_ptr(),
                    weights_cpu[current_slot].data_ptr(),
                    input_tensor_cpu[current_slot].data_ptr(),
                    output_cpu[next_slot].data_ptr(),
                    False,
                ),
            )
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

    def submit_gate_up_and_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
        forward_mask,
        gate_up_mask,
        gate_up_output: torch.Tensor,
        staged=None,
    ):
        """Submit the shard gate_up task AND the CPU forward task in ONE cudaLaunchHostFunc.

        Equivalent to submit_gate_up(mask=gate_up_mask, output=gate_up_output) FOLLOWED BY
        submit_forward(mask=forward_mask): gate_up enters the FIFO first, forward second, so a
        later sync_with_cuda_stream(allow_n_pending=1) still drains gate_up and leaves forward
        pending -- identical semantics. The only difference is ONE host-node GPU<->CPU round-trip
        instead of two (see CPUInfer.submit_pair_with_cuda_stream). A caller that always issues both
        (HiLLM's shard + CPU tiers) halves this dispatch's host-node cost, which under CUDA graph
        capture is a per-replay saving.

        Deferred experts are not supported here (they would need a third task); the caller must use
        the two separate submits when max_deferred_experts_per_token > 0.
        """
        if staged is None:
            staged = self.stage_inputs(hidden_states, topk_ids, topk_weights)
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            current_slot,
            deferred_ids,
        ) = staged
        if deferred_ids is not None:
            raise RuntimeError(
                "submit_gate_up_and_forward does not support deferred experts; use submit_gate_up "
                "+ submit_forward when max_deferred_experts_per_token > 0."
            )

        # gate_up task (shard tier) -- byte-identical to submit_gate_up.
        ids = immediate_experts_ids_cpu[current_slot]
        gate_up_task = self.moe.gate_up_task(
            ids.size(0),
            ids.size(-1),
            ids.data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            gate_up_output.data_ptr(),
            gate_up_mask.data_ptr(),
        )

        # forward task (CPU tier) -- byte-identical to submit_forward's immediate path.
        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        args = (
            bsz_tensor_cpu[current_slot].data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
        )
        if forward_mask is not None:
            forward_task = self.moe.forward_task(*args, forward_mask.data_ptr())
        else:
            forward_task = self.moe.forward_task(*args)

        # ONE host node: gate_up first (FIFO), then forward.
        self.cpu_infer.submit_pair_with_cuda_stream(cuda_stream, gate_up_task, forward_task)
        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU
        """
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)
        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
        return output_gpu[current_slot]

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        """
        Execute forward inference synchronously (submit + sync).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization

        Returns:
            Output tensor on GPU
        """
        self.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        return self.sync_forward(hidden_states, cuda_stream)

    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])

        Example:
            >>> BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])
        """
        KExpertsCPUBuffer.capture_bs = capture_bs

    @staticmethod
    def get_capture_batch_sizes() -> List[int]:
        """
        Get currently configured capture batch sizes.

        Returns:
            List of batch sizes that are being captured
        """
        return KExpertsCPUBuffer.capture_bs

    @staticmethod
    def clear_buffer_cache():
        """
        Clear all cached buffers.

        This frees up memory by clearing the buffer cache. Useful when you want
        to reset the buffer state or free memory.
        """
        KExpertsCPUBuffer.capture_buffers.clear()
        KExpertsCPUBuffer.temp_bs = 0
        KExpertsCPUBuffer.temp_key = tuple()
        KExpertsCPUBuffer.temp_buffer = tuple()

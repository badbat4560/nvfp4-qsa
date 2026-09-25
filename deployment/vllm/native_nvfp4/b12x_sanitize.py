"""Compatibility guard for invalid/padded FlashInfer B12x MoE routes."""

from __future__ import annotations

import torch

_INSTALLED = False
_WORKSPACE_CACHE = {}


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from flashinfer.fused_moe.cute_dsl.b12x_moe import B12xMoEWrapper

    original_run = B12xMoEWrapper.run
    original_allocate_buffers = B12xMoEWrapper._allocate_buffers

    def allocate_shared_buffers(self):
        key = (
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.max_num_tokens,
            self.num_local_experts,
            self.output_dtype,
            str(self.device),
            self.activation,
            self.quant_mode,
            self.source_format,
        )
        cached = _WORKSPACE_CACHE.get(key)
        if cached is None:
            original_allocate_buffers(self)
            cached = (
                self._static_workspace,
                self._dynamic_workspace,
                self._moe_output,
            )
            _WORKSPACE_CACHE[key] = cached
        else:
            (
                self._static_workspace,
                self._dynamic_workspace,
                self._moe_output,
            ) = cached

    def run_with_sanitized_routes(self, *args, **kwargs):
        ids = kwargs["token_selected_experts"]
        weights = kwargs["token_final_scales"]
        valid = (ids >= 0) & (ids < self.num_experts)
        kwargs["token_selected_experts"] = torch.where(
            valid,
            ids,
            torch.zeros((), dtype=ids.dtype, device=ids.device),
        )
        kwargs["token_final_scales"] = torch.where(
            valid,
            weights,
            torch.zeros((), dtype=weights.dtype, device=weights.device),
        )
        return original_run(self, *args, **kwargs)

    B12xMoEWrapper._allocate_buffers = allocate_shared_buffers
    B12xMoEWrapper.run = run_with_sanitized_routes
    _INSTALLED = True

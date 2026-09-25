# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_8_flash_next import (
    Qwen3_8FlashNextTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


def _is_qsa_fp8_dtype(dtype) -> bool:
    """True for the fp8-E4M3 QSA main-KV cache dtypes (KYC 2026-08-28)."""
    return isinstance(dtype, str) and dtype in ("fp8", "fp8_e4m3", "fp8_e4m3fn")


def _is_qsa_nvfp4_dtype(dtype) -> bool:
    """True for the QSA-owned packed E2M1 + group-scale cache format."""
    return isinstance(dtype, str) and dtype in ("nvfp4", "nvfp4_4over6")


_QSA_FP8_SCALES: dict | None = None


def _maybe_load_qsa_fp8_scales(layer, layer_id: int) -> None:
    """Load calibrated per-layer K/V fp8 scales from KYC_QSA_SCALES_JSON.

    JSON schema: {"<layer_name or layer_id>": {"k_scale": float, "v_scale": float}}
    where scale dequantizes: x ~= fp8_stored * scale (== reshape_and_cache_flash
    write scale). Missing entries keep the 1.0 defaults (i.e. scale=1 fallback).
    """
    global _QSA_FP8_SCALES
    import json
    import os

    if _QSA_FP8_SCALES is None:
        path = os.environ.get("KYC_QSA_SCALES_JSON", "")
        try:
            with open(path) as f:
                _QSA_FP8_SCALES = json.load(f)
        except (OSError, ValueError):
            _QSA_FP8_SCALES = {}
    entry = _QSA_FP8_SCALES.get(getattr(layer, "layer_name", None))
    if entry is None:
        entry = _QSA_FP8_SCALES.get(str(layer_id))
    if not entry:
        return
    ks = float(entry["k_scale"])
    vs = float(entry["v_scale"])
    layer._k_scale.fill_(ks)
    layer._v_scale.fill_(vs)
    layer._k_scale_float = ks
    layer._v_scale_float = vs
    layer._k_scale_cpu.fill_(ks)
    layer._v_scale_cpu.fill_(vs)


# --- serve-path calibration (KYC_QSA_CALIBRATE=1) -------------------------- #
# Avoids the offline-LLM()+PLE-offload deadlock by capturing per-layer K/V
# absmax during a normal `vllm serve` (bf16 kv) run and periodically dumping
# scales.json (scale = absmax/448, so x ~= fp8_stored * scale). Runs in the
# worker process; dumps every N updates so SIGTERM never loses the result.
_QSA_CALIB_ON = __import__("os").environ.get("KYC_QSA_CALIBRATE") == "1"
_QSA_CALIB: dict = {}
_QSA_CALIB_N = [0]


def _qsa_calib_record(name: str, key, value) -> None:
    st = _QSA_CALIB.setdefault(name, {"k": 0.0, "v": 0.0})
    st["k"] = max(st["k"], float(key.detach().abs().amax().item()))
    st["v"] = max(st["v"], float(value.detach().abs().amax().item()))
    _QSA_CALIB_N[0] += 1
    if _QSA_CALIB_N[0] % 200 == 0:
        _qsa_calib_dump()


def _qsa_calib_dump() -> None:
    import json
    import os

    out = os.environ.get("KYC_QSA_CALIBRATE_OUT", "/model-cal/scales.json")
    e4m3 = 448.0
    data = {
        n: {"k_scale": max(s["k"] / e4m3, 1e-8), "v_scale": max(s["v"] / e4m3, 1e-8),
            "k_absmax": s["k"], "v_absmax": s["v"]}
        for n, s in _QSA_CALIB.items()
    }
    tmp = out + ".tmp"
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, out)
    except OSError:
        pass


if _QSA_CALIB_ON:
    import atexit

    atexit.register(_qsa_calib_dump)


class Qwen3_8FlashNextQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen3_8FlashNextQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3 route to the QSA-owned dequant-on-read Triton kernel, NOT to
    # the generic FlashAttention path (which rejects fp8 kv on SM120).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
        "nvfp4_4over6",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen3_8FlashNextQSAFlashAttentionImpl]:
        return Qwen3_8FlashNextQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen3_8FlashNextQSAMetadataBuilder]:
        return Qwen3_8FlashNextQSAMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: FullAttentionSpec) -> FullAttentionSpec:
        """Expose the packed NVFP4 byte size to the hybrid cache planner.

        The platform computes the attention block size before the attention
        instance returns its final physical spec.  At that point ``head_size``
        is still the logical QSA width, so advertise the packed K+V bytes via
        ``state_content_bytes``.  The final instance spec uses quant mode NONE
        and physical 144-byte K/V regions, making this transform idempotent.
        """
        if spec.kv_quant_mode != get_kv_quant_mode("nvfp4"):
            return spec
        packed_region = spec.head_size // 2 + spec.head_size // 16
        return replace(
            spec,
            # Per head slot: one packed K region plus one packed V region.
            # AttentionSpec multiplies this by ``num_heads`` itself.
            state_content_bytes=2 * packed_region,
        )

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen3_8FlashNextQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # KYC fp8-QSA (§3.3): the generic FlashAttention parent rejects an fp8
        # kv-cache on SM120. QSA never uses the parent attention path — it owns a
        # Triton kernel — so hand the parent a bf16 dtype to clear its guard and
        # keep the real fp8 write dtype on the QSA owner only.
        args = list(args)
        self._qsa_kv_fp8 = False
        self._qsa_kv_nvfp4 = False
        self._qsa_write_dtype = "auto"
        if len(args) > 6 and _is_qsa_fp8_dtype(args[6]):
            self._qsa_kv_fp8 = True
            self._qsa_write_dtype = args[6]
            args[6] = "bfloat16"
        elif _is_qsa_fp8_dtype(kwargs.get("kv_cache_dtype")):
            self._qsa_kv_fp8 = True
            self._qsa_write_dtype = kwargs["kv_cache_dtype"]
            kwargs["kv_cache_dtype"] = "bfloat16"
        elif len(args) > 6 and _is_qsa_nvfp4_dtype(args[6]):
            self._qsa_kv_nvfp4 = True
            self._qsa_write_dtype = args[6]
            args[6] = "bfloat16"
        elif _is_qsa_nvfp4_dtype(kwargs.get("kv_cache_dtype")):
            self._qsa_kv_nvfp4 = True
            self._qsa_write_dtype = kwargs["kv_cache_dtype"]
            kwargs["kv_cache_dtype"] = "bfloat16"
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support decode context parallelism"
            )
        if self._qsa_kv_fp8 or self._qsa_kv_nvfp4:
            # do_kv_cache_update -> reshape_and_cache_flash quantizes on write
            # using layer._k_scale / _v_scale.
            self.kv_cache_dtype = self._qsa_write_dtype
        elif self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        if getattr(self, "_qsa_kv_nvfp4", False):
            packed_dim = self.head_size // 2
            scale_dim = self.head_size // 16
            storage_dim = packed_dim + scale_dim
            if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != 2 * storage_dim:
                raise RuntimeError(
                    "QSA NVFP4 cache must be uint8 with packed data and fp8 "
                    f"group scales (last dim {2 * storage_dim})"
                )
            packed_cache = kv_cache.transpose(1, 2)
            key_region, value_region = packed_cache.split(storage_dim, dim=-1)
            key_data, key_scale = key_region.split((packed_dim, scale_dim), dim=-1)
            value_data, value_scale = value_region.split((packed_dim, scale_dim), dim=-1)
            from .ops.nvfp4_qsa import qsa_sparse_paged_attention_nvfp4

            qsa_sparse_paged_attention_nvfp4(
                query[:num_tokens],
                key_data,
                key_scale,
                value_data,
                value_scale,
                float(getattr(layer, "_qsa_nvfp4_k_global", 1.0)),
                float(getattr(layer, "_qsa_nvfp4_v_global", 1.0)),
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                output[:num_tokens],
            )
            return output

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        if getattr(self, "_qsa_kv_fp8", False) and key_cache.dtype == torch.uint8:
            # fp8 storage: zero-copy reinterpret the uint8 bytes as fp8_e4m3 and
            # dequant inside the kernel with the per-layer K/V scales that
            # reshape_and_cache_flash used on write (§4.2). Host-side *_scale_float
            # avoids a per-step device sync.
            if query.dtype != torch.bfloat16:
                raise NotImplementedError(
                    "Qwen3.8-Flash-Next QSA requires a BF16 query"
                )
            from .ops.qsa import qsa_sparse_paged_attention_fp8

            qsa_sparse_paged_attention_fp8(
                query[:num_tokens],
                key_cache.view(torch.float8_e4m3fn),
                value_cache.view(torch.float8_e4m3fn),
                float(getattr(layer, "_k_scale_float", 1.0)),
                float(getattr(layer, "_v_scale_float", 1.0)),
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                output[:num_tokens],
            )
            return output

        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
        )
        return output


class Qwen3_8FlashNextQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3_8FlashNextTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")
        if not (
            _is_qsa_fp8_dtype(cache_config.cache_dtype)
            or _is_qsa_nvfp4_dtype(cache_config.cache_dtype)
        ) and (
            cache_config.cache_dtype not in ("auto", "bfloat16")
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16, fp8_e4m3, or NVFP4 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support KV quantization"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires causal decoder attention"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support dual-chunk RoPE"
            )
        # Qwen3.8-Flash-Next full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        # fp8 kv maps to a uint8 storage tensor (float8 bytes); the QSA kernel
        # reinterprets it as fp8_e4m3 on read.
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16, fp8_e4m3, or NVFP4 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)
        if _is_qsa_fp8_dtype(self.kv_cache_dtype):
            # Apply calibrated per-layer fp8 scales (KYC_QSA_SCALES_JSON) if any;
            # otherwise the 1.0 defaults keep a safe scale=1 fallback.
            _maybe_load_qsa_fp8_scales(self, layer_id)
        self._qsa_nvfp4_k_global = 1.0
        self._qsa_nvfp4_v_global = 1.0

        self.attn_backend = Qwen3_8FlashNextQSAFlashAttentionBackend
        self.impl = Qwen3_8FlashNextQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if _is_qsa_nvfp4_dtype(self.kv_cache_dtype):
            # vLLM binds one tensor per attention layer. Pack each K/V region as
            # [head_dim/2 data bytes | head_dim/16 fp8 group-scale bytes].
            # Advertise the physical width here and keep generic NVFP4 layout
            # customization disabled: QSA owns both the writer and reader.
            storage_dim = self.head_dim // 2 + self.head_dim // 16
            return FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=storage_dim,
                head_size_v=storage_dim,
                dtype=torch.uint8,
                kv_quant_mode=get_kv_quant_mode("auto"),
            )
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen3_8FlashNextQSAFlashAttentionImpl, self.impl)
        if _QSA_CALIB_ON:
            _qsa_calib_record(self.layer_name, key, value)
        if _is_qsa_nvfp4_dtype(self.kv_cache_dtype):
            packed_dim = self.head_dim // 2
            scale_dim = self.head_dim // 16
            storage_dim = packed_dim + scale_dim
            packed_cache = self.kv_cache.transpose(1, 2)
            key_region, value_region = packed_cache.split(storage_dim, dim=-1)
            key_data, key_scale = key_region.split((packed_dim, scale_dim), dim=-1)
            value_data, value_scale = value_region.split((packed_dim, scale_dim), dim=-1)
            from .ops.nvfp4_qsa import qsa_write_cache_nvfp4

            qsa_write_cache_nvfp4(
                key,
                value,
                key_data,
                key_scale,
                value_data,
                value_scale,
                main_metadata.slot_mapping,
                float(self._qsa_nvfp4_k_global),
                float(self._qsa_nvfp4_v_global),
            )
        else:
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen3_8_flash_next_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen3_8_flash_next_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen3_8FlashNextQSAAttention",
    "Qwen3_8FlashNextQSAFlashAttentionBackend",
    "Qwen3_8FlashNextQSAFlashAttentionImpl",
    "qwen3_8_flash_next_qsa_with_output",
]

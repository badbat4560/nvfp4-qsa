"""Guarded experimental cache interface for the isolated QSA kernels.

Not thread-safe: callers own stream ordering and buffer lifetime. Validation
uses host-visible reductions and is intentionally outside CUDA graph capture.
No unguarded fast mode is exposed here; vLLM integration is a separate patch.
"""
import math
import torch


class PackedCache:
    def __init__(self, blocks, page_size=16, kv_heads=2, head_dim=256,
                 key_scale=1.0, value_scale=1.0, device='cuda'):
        if min(blocks, page_size, kv_heads) < 1 or head_dim != 256:
            raise ValueError('Positive dimensions and head_dim=256 required')
        if any(not math.isfinite(g) or g <= 0 for g in (key_scale, value_scale)):
            raise ValueError('Global scales must be finite and positive')
        self._global_scales = (float(key_scale), float(value_scale))
        self.head_dim = head_dim
        self.storage = torch.zeros(blocks, kv_heads, page_size, 2*(head_dim//2+head_dim//16),
                                   dtype=torch.uint8, device=device)
        self._parts = self.storage.transpose(1, 2).split(head_dim//2+head_dim//16, dim=-1)
        self.kd,self.ks=self._parts[0].split((head_dim//2,head_dim//16),dim=-1)
        self.vd,self.vs=self._parts[1].split((head_dim//2,head_dim//16),dim=-1)

    @property
    def global_scales(self):
        return self._global_scales

    @property
    def bytes(self):
        return self.storage.numel()

    def _same_device(self, tensors):
        if self.storage.device.type != 'cuda' or any(t.device != self.storage.device for t in tensors):
            raise ValueError('All tensors must be on the cache CUDA device')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Checked API cannot execute during CUDA graph capture')

    def write(self, key, value, slots):
        self._same_device((key,value,slots))
        if key.ndim != 3 or key.shape != value.shape or key.shape[1:] != (self.kd.shape[2],self.head_dim):
            raise ValueError('K/V shape must be [tokens, kv_heads, 256]')
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16 or key.stride(-1)!=1 or value.stride(-1)!=1:
            raise ValueError('BF16 K/V with contiguous last dimension required')
        if slots.dtype != torch.int64 or slots.shape != (key.shape[0],) or not slots.is_contiguous():
            raise ValueError('Contiguous int64 slot per token required')
        if not torch.isfinite(key).all().item() or not torch.isfinite(value).all().item():
            raise ValueError('Non-finite K/V input')
        capacity=self.kd.shape[0]*self.kd.shape[1]
        valid=slots[(slots>=0)&(slots<capacity)]
        if valid.unique().numel()!=valid.numel():
            raise ValueError('Duplicate valid physical slots would race')
        if not key.shape[0]:return
        from .kernels import qsa_write_cache_nvfp4
        qsa_write_cache_nvfp4(key,value,self.kd,self.ks,self.vd,self.vs,slots,*self.global_scales)

    def attend(self, query, indices, block_table, token_to_request):
        self._same_device((query,indices,block_table,token_to_request))
        if query.ndim!=3 or query.shape[1:]!=(24,256) or query.dtype!=torch.bfloat16 or query.stride(-1)!=1 or self.kd.shape[2]!=2:
            raise ValueError('This checked release supports BF16 query [rows,24,256] and 2 KV heads')
        if not torch.isfinite(query).all().item():raise ValueError('Non-finite query')
        if indices.ndim!=2 or indices.shape[0]!=query.shape[0] or indices.shape[1]<1:
            raise ValueError('Nonempty selection row per query required')
        if block_table.ndim!=2 or min(block_table.shape)<1 or token_to_request.shape!=(query.shape[0],):
            raise ValueError('Nonempty request block table and one request ID per query required')
        if any(t.dtype!=torch.int32 or not t.is_contiguous() for t in (indices,block_table,token_to_request)):
            raise ValueError('Contiguous int32 metadata required')
        output=torch.empty_like(query)
        if not query.shape[0]:return output
        from .kernels import qsa_sparse_paged_attention_nvfp4
        qsa_sparse_paged_attention_nvfp4(query,self.kd,self.ks,self.vd,self.vs,*self.global_scales,
                                        indices,block_table,token_to_request,output)
        return output

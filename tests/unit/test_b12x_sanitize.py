import importlib.util
import sys
import types
from pathlib import Path

import torch


PATCH_PATH = (
    Path(__file__).parents[2]
    / "deployment"
    / "vllm"
    / "native_nvfp4"
    / "b12x_sanitize.py"
)


def _install_patch(monkeypatch):
    class FakeWrapper:
        allocations = 0

        def __init__(self):
            self.num_experts = 512
            self.top_k = 10
            self.hidden_size = 2560
            self.intermediate_size = 640
            self.max_num_tokens = 2048
            self.num_local_experts = 512
            self.output_dtype = torch.bfloat16
            self.device = "cuda"
            self.activation = "silu"
            self.quant_mode = "nvfp4"
            self.source_format = "modelopt"

        def _allocate_buffers(self):
            type(self).allocations += 1
            self._static_workspace = object()
            self._dynamic_workspace = object()
            self._moe_output = object()

        def run(self, *args, **kwargs):
            return kwargs

    module_name = "flashinfer.fused_moe.cute_dsl.b12x_moe"
    fake_module = types.ModuleType(module_name)
    fake_module.B12xMoEWrapper = FakeWrapper
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    spec = importlib.util.spec_from_file_location("b12x_sanitize_test", PATCH_PATH)
    patch_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(patch_module)
    patch_module.install()
    return FakeWrapper


def test_invalid_routes_are_zero_weighted(monkeypatch):
    wrapper_cls = _install_patch(monkeypatch)
    wrapper = wrapper_cls()
    result = wrapper.run(
        token_selected_experts=torch.tensor([[-1, 7, 512]], dtype=torch.int32),
        token_final_scales=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32),
    )

    assert result["token_selected_experts"].tolist() == [[0, 7, 0]]
    assert result["token_final_scales"].tolist() == [[0.0, 0.5, 0.0]]


def test_identical_wrappers_share_workspace(monkeypatch):
    wrapper_cls = _install_patch(monkeypatch)
    first = wrapper_cls()
    second = wrapper_cls()

    first._allocate_buffers()
    second._allocate_buffers()

    assert wrapper_cls.allocations == 1
    assert second._static_workspace is first._static_workspace
    assert second._dynamic_workspace is first._dynamic_workspace
    assert second._moe_output is first._moe_output

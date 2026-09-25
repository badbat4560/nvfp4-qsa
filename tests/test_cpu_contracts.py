"""CPU checks of codec, planner transformation and historical accounting.

These tests do not validate GPU execution, model quality or server concurrency.
"""
import ast
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nvfp4_oracle import encode, decode, _round_e2m1_magnitude


def test_codebook_roundtrip_and_nibble_order():
    values = torch.tensor([[0, .5, 1, 1.5, 2, 3, 4, 6,
                            0, -.5, -1, -1.5, -2, -3, -4, -6]])
    nibbles, packed, scales, _ = encode(values, torch.tensor(1.0))
    assert packed[0, 0].item() == 0x10
    assert packed[0, 7].item() == 0xFE
    assert torch.equal(decode(packed, scales, torch.tensor(1.0), 16), values)


def test_midpoints_use_even_code():
    midpoint = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert _round_e2m1_magnitude(midpoint).tolist() == [0, 2, 2, 4, 4, 6, 6]


def test_zero_groups_finite():
    _, data, scales, _ = encode(torch.zeros(2, 256), torch.tensor(1.0))
    result = decode(data, scales, torch.tensor(1.0), 256)
    assert torch.isfinite(result).all() and not result.count_nonzero()


def test_actual_planner_customization_contract():
    tree = ast.parse((ROOT / 'integration/snapshot/qsa.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'Qwen3_8FlashNextQSAFlashAttentionBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'customize_spec')
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {'replace': replace, 'get_kv_quant_mode': lambda x: x}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 '<actual customize_spec>', 'exec'), namespace)
    @dataclass(frozen=True)
    class Spec:
        head_size: int
        kv_quant_mode: str
        state_content_bytes: int | None = None
    customize = lambda s: namespace['customize_spec'](None, s)
    packed = customize(Spec(256, 'nvfp4'))
    assert packed.state_content_bytes * 2 == 576
    assert customize(packed) == packed
    physical = Spec(144, 'NONE')
    assert customize(physical) is physical
    fp8 = Spec(256, 'fp8')
    assert customize(fp8) is fp8
    assert 5568 * 576 == 3207168
    assert 3136 * 576 < 3207168


def test_saved_streaming_accounting():
    result = json.loads((ROOT / 'results/2026-08-28/streaming.json').read_text())
    for name in ['decode_single', 'decode_concurrency_7',
                 'prefill_single', 'prefill_concurrency_7']:
        case = result[name]
        assert len(case['requests']) == case['concurrency']
        assert sum(r['output_tokens'] for r in case['requests']) == case['output_tokens_total']
        assert sum(r['prompt_tokens'] for r in case['requests']) == case['prompt_tokens_total']
        expected = case['output_tokens_total'] / case['wall_s']
        assert abs(expected - case['aggregate_output_tok_s_e2e']) < 1e-9
    # Historical p95 helper rounds to the final observation for a group of 7.
    case = result['prefill_concurrency_7']
    assert case['ttft_p95_s'] == max(r['ttft_s'] for r in case['requests'])

"""Execute the public port's actual spec method without importing all of vLLM."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.parametrize('mode,width,dtype,quant_mode',[
    ('bfloat16',256,torch.bfloat16,'bfloat16'),
    ('nvfp4',144,torch.uint8,'NONE'),
])
def test_physical_spec(mode,width,dtype,quant_mode):
    source=Path(__file__).resolve().parents[1]/'integration/public/qsa.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpQSAAttention')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_kv_cache_spec')
    method.returns=None
    for arg in method.args.args:arg.annotation=None
    namespace={'torch':torch,'FullAttentionSpec':lambda **kw:kw,'get_kv_quant_mode':lambda x:'NONE' if x=='auto' else x}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
    owner=SimpleNamespace(kv_cache_dtype=mode,head_dim=256,num_kv_heads=2,kv_cache_torch_dtype=torch.bfloat16)
    config=SimpleNamespace(cache_config=SimpleNamespace(block_size=5568))
    result=namespace['get_kv_cache_spec'](owner,config)
    assert result['head_size']==result['head_size_v']==width
    assert result['dtype']==dtype
    assert result['kv_quant_mode']==quant_mode
    assert result['block_size']==5568

"""Small real-GPU checks of the public API, independent of vLLM."""
import pytest
import torch
from nvfp4_qsa.cache import PackedCache
from nvfp4_qsa.oracle import encode,decode

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('amplitude',[0.,1e-4,1.,10000.])
def test_writer_and_slot_reuse(amplitude):
    torch.manual_seed(42)
    cache=PackedCache(4)
    key=(torch.randn(8,2,256)*amplitude).bfloat16().cuda()
    slots=torch.tensor([0,1,15,16,35,63,-1,64],dtype=torch.int64,device='cuda')
    cache.write(key,key,slots)
    for row,slot in enumerate(slots.tolist()[:6]):
        _,packed,scales,_=encode(key[row].cpu(),torch.tensor(1.))
        actual=decode(cache.kd[slot//16,slot%16].cpu(),cache.ks[slot//16,slot%16].cpu(),torch.tensor(1.),256)
        expected=decode(packed,scales,torch.tensor(1.),256)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    cache.write(torch.zeros_like(key[:1]),torch.zeros_like(key[:1]),slots[:1])
    assert not cache.kd[0,0].any().item()

def test_guard_rejects_duplicate_and_nonfinite():
    cache=PackedCache(1)
    key=torch.zeros(2,2,256,dtype=torch.bfloat16,device='cuda')
    with pytest.raises(ValueError,match='Duplicate'):
        cache.write(key,key,torch.zeros(2,dtype=torch.int64,device='cuda'))
    key[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):
        cache.write(key,key,torch.arange(2,dtype=torch.int64,device='cuda'))

def test_invalid_attention_metadata_returns_finite_zero():
    cache=PackedCache(1)
    query=torch.ones(3,24,256,device='cuda',dtype=torch.bfloat16)
    indices=torch.tensor([[-1,999],[-1,-1],[0,1]],device='cuda',dtype=torch.int32)
    blocks=torch.tensor([[0]],device='cuda',dtype=torch.int32)
    requests=torch.tensor([0,0,99],device='cuda',dtype=torch.int32)
    result=cache.attend(query,indices,blocks,requests)
    assert torch.isfinite(result).all().item()
    assert not result.any().item()

def test_sparse_attention_fragmented_pages_against_cpu_dense():
    torch.manual_seed(37)
    cache=PackedCache(4)
    key=torch.randn(32,2,256,device='cuda',dtype=torch.bfloat16)
    value=torch.randn_like(key)
    slots=torch.cat((torch.arange(32,48),torch.arange(0,16))).long().cuda()
    cache.write(key,value,slots)
    query=torch.randn(2,24,256,device='cuda',dtype=torch.bfloat16)
    indices=torch.tensor([[0,5,12,17,31],[1,8,15,16,29]],device='cuda',dtype=torch.int32)
    blocks=torch.tensor([[2,0]],device='cuda',dtype=torch.int32)
    output=cache.attend(query,indices,blocks,torch.zeros(2,device='cuda',dtype=torch.int32)).cpu().float()
    def decoded(data,scales):
        out=decode(data.contiguous().reshape(-1,128).cpu(),scales.contiguous().reshape(-1,16).cpu(),torch.tensor(1.),256).bfloat16().float()
        return out.reshape(4,16,2,256)
    kd=decoded(cache.kd,cache.ks);vd=decoded(cache.vd,cache.vs)
    expected=[]
    for row in range(2):
        logical=indices[row].cpu().long();physical=blocks[0].cpu().long()[logical//16]
        k=kd[physical,logical%16].repeat_interleave(12,dim=1)
        v=vd[physical,logical%16].repeat_interleave(12,dim=1)
        logits=torch.einsum('hd,thd->ht',query[row].cpu().float(),k)/16
        expected.append(torch.einsum('ht,thd->hd',logits.softmax(-1),v))
    torch.testing.assert_close(output,torch.stack(expected),rtol=.02,atol=.025)

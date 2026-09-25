from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
repo=Path(__file__).resolve().parents[1]
summary=json.loads((repo/'results/2026-09-25/summary.json').read_text())
modes=[m for m in ('bfloat16','fp8','nvfp4') if m in summary and summary[m]['speed']]
labels={'bfloat16':'BF16','fp8':'FP8','nvfp4':'NVFP4'}
rows=[]
speed=['| KV cache | Concurrency | Requests | Median TTFT | Batch E2E output | Errors |','|---|---:|---:|---:|---:|---:|']
for mode in modes:
    for c,r in summary[mode]['speed'].items():
        speed.append(f"| {labels[mode]} | {c} | {r['n']} | {r['ttft_median_s']*1000:.1f} ms | {r['batch_output_tok_s']:.2f} tok/s | {r['errors']} |")
        rows.append({'mode':mode,'concurrency':c,**r})
with (repo/'results/current-summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
quality=['| Task | '+ ' | '.join(labels[m] for m in modes)+' |','|---|'+ '---:|'*len(modes)]
for ds in ('ag_news','sib200_en','sib200_ru','synthetic_needle'):
    quality.append('| '+ds+' | '+' | '.join(f"{summary[m]['quality'][ds]['correct']}/{summary[m]['quality'][ds]['n']}" for m in modes)+' |')
paired={}
if 'bfloat16' in modes and 'nvfp4' in modes:
    raw={m:[json.loads(x) for x in (repo/f'results/2026-09-25/{m}-measurements.jsonl').read_text().splitlines()] for m in ('bfloat16','nvfp4')}
    cases={m:{r['id']:r for r in raw[m] if r['kind']=='quality'} for m in raw}
    common=cases['bfloat16'].keys()&cases['nvfp4'].keys()
    paired={'same_stripped_answer':sum(cases['bfloat16'][k]['answer'].strip()==cases['nvfp4'][k]['answer'].strip() for k in common),'n':len(common),'bf16_correct_nvfp4_wrong':sum(cases['bfloat16'][k]['correct'] and not cases['nvfp4'][k]['correct'] for k in common),'nvfp4_correct_bf16_wrong':sum(cases['nvfp4'][k]['correct'] and not cases['bfloat16'][k]['correct'] for k in common)}
(repo/'results/2026-09-25/paired.json').write_text(json.dumps(paired,indent=2))
capacity=json.loads((repo/'results/2026-09-25/cache-capacity.json').read_text())
capacity_table='| Cache | Available KV budget | Reported planner token pool |\n|---|---:|---:|\n'+'\n'.join(f"| {labels[m]} | {r['available_kv_cache_gib']:.2f} GiB | {r['planner_token_pool']:,} |" for m,r in capacity.items())
en='''## Fresh server evaluation: September 25, 2026 (UTC)

The serving model was temporarily stopped after its active and waiting request
counts reached zero. Evaluation used an isolated container on the same RTX PRO
6000 Blackwell, with no production traffic routed to it. The other services were
left running; a residual GPU process used approximately 712 MiB before the window.
This is a controlled application comparison on a shared host, not an otherwise
empty laboratory machine.

The completed runs use the same checkpoint, MTP=3, maximum length 169,984,
maximum sequences 12, scheduler budget 8,192, memory utilization 0.94 and float32
recurrent state. CUDA graphs and compilation are enabled. The cache mode is the
changed parameter. The wrapper's persisted MTP choice was inspected rather than
assuming its command-line value was effective.

'''+capacity_table+'\n\nThe reported planner pool is about 2.32 times larger with NVFP4 than BF16 at a nearly equal KV budget. This is an allocator observation, not demonstrated sustained full-context concurrency or a 2.32x total-VRAM reduction.\n\n'+ '\n'.join(speed)+'\n\n'+ '\n'.join(quality)+'''

Each quality subset has 16 examples selected before evaluation from a pinned
AG News or SIB-200 test file. Nine synthetic retrieval prompts place a code at
three positions in three text sizes. The maximum observed input is 9,280 tokens;
this does not validate retrieval at 120K. Exact stripped answers are scored,
including formatting compliance. Raw answers and token counts are retained.
Wilson intervals are included in the JSON and chart to show how uncertain these
small subset scores are. They are descriptive and do not account for dataset
dependence or prove quality equivalence.

For speed, two single-request warmups precede evaluation. There are 12 requests
per concurrency, each requesting 128 output tokens with ignore_eos. Concurrency
four is not independently warmed before its measured batch. Unique user-message
prefixes reduce whole-request reuse; shared chat-template prefix caching remains
enabled. TTFT means first nonempty content chunk. Batch throughput divides actual
output tokens by the complete client batch wall time, including queueing and
client overhead. These are one-batch observations, not stable p95 or pure kernel
performance. The order was NVFP4 then BF16; FP8 was not measured.
There was no randomized repeated crossover.

Earlier BF16 baseline attempts with MTP disabled failed during startup with a CUDA illegal
memory access, both with compilation and in eager mode. They produced no task
scores or timing samples. A standalone GDN warmup with the same head dimensions
passed on Blackwell; that does not isolate the earlier fault. The failed
configurations remain an integration regression to investigate. The completed
MTP=3 runs must not be described as validation of the MTP=0 path.

![September server observations](../figures/current-performance.png)

![Small paired quality subsets](../figures/current-quality.png)
'''
if paired:en+=f"\nThe two modes produced the same stripped answer on {paired['same_stripped_answer']}/{paired['n']} cases. NVFP4 corrected {paired['nvfp4_correct_bf16_wrong']} BF16 error and changed {paired['bf16_correct_nvfp4_wrong']} BF16-correct answer to an error. Equal aggregate accuracy does not establish equivalence.\n"
(repo/'docs/current-evaluation.md').write_text(en,encoding='utf8')
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained');colors=['#415a77','#b57b36','#287c73']
x=np.arange(2);width=.75/max(len(modes),1)
for i,m in enumerate(modes):
    pos=x-.375+width*(i+.5)
    for ax,key,mult in [(axes[0],'ttft_median_s',1000),(axes[1],'batch_output_tok_s',1)]:
        vals=[summary[m]['speed'][str(c)][key]*mult for c in (1,4)]
        bars=ax.bar(pos,vals,width,color=colors[i],label=labels[m]);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
for ax,title,ylabel in [(axes[0],'Time to first content chunk','Median TTFT (ms)'),(axes[1],'Complete batch throughput','Output tokens / second')]:
    ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xticks(x,['1 concurrent','4 concurrent']);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,ax.get_ylim()[1]*1.17)
axes[0].legend(frameon=False);fig.suptitle('MTP=3 • 12 requests per condition • one batch per cache mode',fontsize=12)
fig.savefig(repo/'figures/current-performance.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(4)
for i,m in enumerate(modes):
    values=[summary[m]['quality'][ds] for ds in ('ag_news','sib200_en','sib200_ru','synthetic_needle')]
    means=np.array([r['accuracy'] for r in values]);intervals=np.array([r['wilson95'] for r in values])
    ax.bar(x-.375+width*(i+.5),means,width,label=labels[m],color=colors[i],yerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),capsize=3)
ax.set_xticks(x,['AG News\nn=16','SIB-200 EN\nn=16','SIB-200 RU\nn=16','Synthetic retrieval\nn=9']);ax.set_ylim(0,1.08);ax.set_ylabel('Exact-match accuracy');ax.set_title('Small regression subsets, not full-dataset benchmark scores\nError bars: descriptive 95% Wilson intervals');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
fig.savefig(repo/'figures/current-quality.png',dpi=180);plt.close(fig)
print(en[:2500])

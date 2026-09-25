"""Regenerate figures and summary from saved observations, without a GPU/server."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'results/2026-08-28/streaming.json').read_text())
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#f7f8fa', 'axes.facecolor': '#f7f8fa'})

fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
fig.get_layout_engine().set(rect=(0, .07, 1, .93))
bars = ax.barh(['BF16', 'FP8', 'NVFP4'], [2048, 1024, 576],
              color=['#8995a7', '#4576a9', '#147d73'])
ax.invert_yaxis()
ax.bar_label(bars, padding=6, fmt='%d B')
ax.set(xlim=(0, 2350), xlabel='Bytes / token / QSA layer (2 KV heads, head_dim=256)',
       title='Packed QSA K/V storage: 576 bytes per token')
fig.text(.16, .005, 'Includes group scales. Excludes GDN state, indexer, weights and runtime buffers.', fontsize=9)
fig.savefig(OUT / 'kv-storage.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
fig.get_layout_engine().set(rect=(0, .07, 1, .93))
rates = [r['decode_tok_s'] for r in DATA['decode_concurrency_7']['requests']]
axes[0].bar(range(1, 8), rates, color='#147d73')
axes[0].axhline(DATA['decode_concurrency_7']['per_session_decode_tok_s_p50'], color='#d58137', linestyle='--',label='Median: 52.48')
axes[0].set(xlabel='Request', ylabel='Decode tokens / second', title='7 short requests, 256 output tokens each')
axes[0].legend()
ttfts=[r['ttft_s'] for r in DATA['prefill_concurrency_7']['requests']]
axes[1].bar(range(1, 8), ttfts, color='#4576a9')
axes[1].set(xlabel='Request', ylabel='Time to first text chunk (seconds)',title='7 long requests, 120,064 prompt tokens each')
fig.suptitle('Historical streaming observations | 2026-08-28 | MTP=2')
fig.text(.15,.005,'One group per scenario; no repeated-run confidence intervals. Different requests in each panel.',fontsize=9)
fig.savefig(OUT/'streaming-observations.png',dpi=180)
plt.close(fig)

rows=[]
for name in ['decode_single','decode_concurrency_7','prefill_single','prefill_concurrency_7']:
 case=DATA[name]
 rows.append({'scenario':name,'requests':case['concurrency'],
              'ttft_p50_s':case['ttft_p50_s'],'ttft_max_s':case['ttft_max_s'],
              'wall_s':case['wall_s'],'prompt_tokens':case['prompt_tokens_total'],
              'output_tokens':case['output_tokens_total'],
              'decode_tok_s_median':case['per_session_decode_tok_s_p50'],
              'output_tok_s_e2e':case['aggregate_output_tok_s_e2e'],
              'prompt_tok_s_wall':case['prompt_tokens_total']/case['wall_s']})
with (ROOT/'results/summary.csv').open('w',newline='',encoding='utf8') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
print('Regenerated two figures and results/summary.csv from saved JSON.')

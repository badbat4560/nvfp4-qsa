from pathlib import Path
import json,statistics,math,csv
root=Path(__file__).resolve().parents[1]/'results/2026-09-25'
def wilson(k,n):
    z=1.95996398454;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]
summary={}
for mode in ('bfloat16','fp8','nvfp4'):
    p=root/(mode+'-measurements.jsonl')
    if not p.exists():continue
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    record={'quality':{},'speed':{}}
    for ds in ('ag_news','sib200_en','sib200_ru','synthetic_needle'):
        q=[r for r in rows if r.get('kind')=='quality' and r.get('dataset')==ds]
        if q:
            k=sum(r.get('correct',False) for r in q)
            record['quality'][ds]={'correct':k,'n':len(q),'accuracy':k/len(q),'wilson95':wilson(k,len(q)),'max_prompt_tokens':max(r.get('usage',{}).get('prompt_tokens',0) for r in q),'truncated':sum(r.get('finish_reason')=='length' for r in q)}
    for concurrency in (1,4):
        speed=[r for r in rows if r.get('kind')=='speed' and r.get('concurrency')==concurrency and r.get('ok')]
        batches=[r for r in rows if r.get('kind')=='batch' and r['concurrency']==concurrency]
        if speed and batches:
            b=batches[-1]
            record['speed'][str(concurrency)]={'n':len(speed),'ttft_median_s':statistics.median(r['ttft_s'] for r in speed),'ttft_min_s':min(r['ttft_s'] for r in speed),'ttft_max_s':max(r['ttft_s'] for r in speed),'wall_median_s':statistics.median(r['wall_s'] for r in speed),'batch_output_tok_s':b['output_tokens']/b['wall_s'],'errors':b['errors'],'output_tokens':b['output_tokens'],'batch_wall_s':b['wall_s']}
    record['errors']=[r for r in rows if r.get('error_type')]
    summary[mode]=record
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

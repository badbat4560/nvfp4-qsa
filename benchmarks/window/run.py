"""Dedicated endpoint benchmark. All questions are public or synthetic."""
import concurrent.futures,json,pathlib,statistics,time,urllib.request,uuid

import os
ROOT=pathlib.Path(os.environ.get('BENCH_DIRECTORY','.'))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(body,stream=False):
    req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    start=time.perf_counter()
    with opener.open(req,timeout=180) as response:
        if not stream:
            data=json.load(response)
            return data,time.perf_counter()-start
        first=None; usage=None; chunks=0
        for line in response:
            if not line.startswith(b'data: ') or b'[DONE]' in line:continue
            event=json.loads(line[6:])
            if event.get('error'):raise RuntimeError('stream error')
            if event.get('usage'):usage=event['usage']
            choices=event.get('choices',[])
            if choices and choices[0].get('delta',{}).get('content'):
                first=first or time.perf_counter();chunks+=1
        end=time.perf_counter()
        if first is None or usage is None:raise RuntimeError('missing usage or text')
        return {'ttft_s':first-start,'wall_s':end-start,'post_first_s':end-first,'text_chunks':chunks,'usage':usage}

def body(prompt,tokens=64):
    return {'model':'audit-model','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':tokens,'chat_template_kwargs':{'enable_thinking':False}}

def main(mode):
    path=ROOT/(mode+'-measurements.jsonl')
    def save(row):
        with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for i in range(2):request(body('Reply with the word ready.',16))
    for row in json.loads((ROOT/'cases.json').read_text()):
        prompt=row['prompt']
        try:
            data,elapsed=request(body(prompt,64)); choice=data['choices'][0]
            answer=choice['message'].get('content') or ''
            save({'kind':'quality','id':row['id'],'dataset':row['dataset'],'expected':row['expected'],'answer':answer,'correct':answer.strip()==row['expected'],'finish_reason':choice['finish_reason'],'usage':data.get('usage'),'wall_s':elapsed})
        except Exception as exc:save({'kind':'quality','id':row['id'],'dataset':row['dataset'],'expected':row['expected'],'correct':False,'error_type':type(exc).__name__})
    def speed(i):
        # Unique prefix defeats reuse across repetitions. Same body length distribution per mode.
        b=body('Request '+uuid.uuid4().hex+'. Explain how a hash table handles collisions in detail.',128)
        b.update(stream=True,stream_options={'include_usage':True},ignore_eos=True)
        try:return dict(kind='speed',id=i,ok=True,**request(b,True))
        except Exception as exc:return {'kind':'speed','id':i,'ok':False,'error_type':type(exc).__name__}
    for concurrency in (1,4):
        started=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows=list(pool.map(speed,range(12)))
        elapsed=time.perf_counter()-started
        for row in rows:save(dict(row,concurrency=concurrency))
        save({'kind':'batch','concurrency':concurrency,'requests':12,'wall_s':elapsed,'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in rows),'errors':sum(not r['ok'] for r in rows)})
if __name__=='__main__':
    import sys
    main(sys.argv[1])

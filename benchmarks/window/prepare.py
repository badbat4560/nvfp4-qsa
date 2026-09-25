"""Recreate exactly the public/synthetic evaluation prompts from pinned sources."""
import csv,hashlib,io,json,pathlib,urllib.request

root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
rows={}
for source in manifest['sources']:
    url='https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}'.format(**source)
    raw=urllib.request.urlopen(url,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=source['source_sha256']:
        raise RuntimeError('Source checksum mismatch: '+source['dataset'])
    text=raw.decode('utf8')
    values=([json.loads(x) for x in text.splitlines() if x.strip()]
            if source['file'].endswith('.jsonl') else list(csv.DictReader(io.StringIO(text),delimiter='\t')))
    rows[source['dataset']]=(source,values)
cases=[]
for case in manifest['cases']:
    if case['dataset']=='synthetic_needle':continue
    source,values=rows[case['dataset']]
    index=int(case['id'].split(':')[-1]);text=values[index]['text']
    labels='\n'.join(f'{i+1}: {label}' for i,label in enumerate(source['labels']))
    prompt='Classify the following text. Reply with ONLY the numeric label, no explanation.\nLabels:\n'+labels+'\nText:\n'+text
    cases.append(dict(case,prompt=prompt))
cases+=manifest['synthetic_cases']
(root/'cases.json').write_text(json.dumps(cases),encoding='utf8')
print('Reconstructed',len(cases),'cases. External datasets retain their original licenses.')

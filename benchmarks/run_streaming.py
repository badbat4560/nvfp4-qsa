"""Explicit opt-in client for a dedicated test endpoint; dry-run by default.

Input JSONL: {"id":"case-1", "prompt":"...", "max_tokens":256}.
No endpoint, auth key or prompt text is written to the result file.
Token usage is required; missing usage is an error, never an estimated count.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


def request_once(base_url, model, row, timeout):
    start = time.perf_counter()
    request_id = uuid.uuid4().hex
    prompt = f"request-{request_id}\n" + row['prompt']
    result = {'id': row['id'], 'request_id': request_id,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'max_tokens': row['max_tokens'], 'ok': False}
    body = {'model': model, 'prompt': prompt, 'temperature': 0,
            'max_tokens': row['max_tokens'], 'ignore_eos': True, 'stream': True,
            'stream_options': {'include_usage': True}}
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    first = None
    usage = None
    chunks = 0
    try:
        req = urllib.request.Request(base_url.rstrip('/')+'/v1/completions',
                                     json.dumps(body).encode(), headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                if event.get('error'):
                    raise RuntimeError('Server returned a streaming error')
                if event.get('usage'):
                    usage = event['usage']
                choices = event.get('choices') or []
                if choices and choices[0].get('text'):
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
        end = time.perf_counter()
        if first is None or usage is None:
            raise RuntimeError('Missing text or actual usage')
        for key in ('prompt_tokens', 'completion_tokens'):
            if not isinstance(usage.get(key), int) or usage[key] < 0:
                raise RuntimeError('Missing/invalid token counts')
        result.update(ok=True, ttft_chunk_s=first-start, total_s=end-start,
                      post_first_chunk_s=end-first, text_chunks=chunks,
                      prompt_tokens=usage['prompt_tokens'],
                      output_tokens=usage['completion_tokens'],
                      decode_estimate_tok_s=max(usage['completion_tokens']-1, 0)/max(end-first, 1e-9))
    except Exception as error:
        # Do not serialize response bodies, credentials, endpoints or prompt text.
        result.update(error_type=type(error).__name__, total_s=time.perf_counter()-start)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--execute', action='store_true', help='Send requests; requires a dedicated/approved test window')
    a = p.parse_args()
    if a.concurrency < 1 or a.repeats < 1:
        p.error('concurrency and repeats must be positive')
    cases = [json.loads(line) for line in a.cases.read_text(encoding='utf8').splitlines() if line.strip()]
    if not cases:
        p.error('cases cannot be empty')
    for row in cases:
        if not isinstance(row.get('prompt'), str) or not isinstance(row.get('max_tokens'), int) or row['max_tokens'] < 1 or 'id' not in row:
            p.error('Each row needs id, prompt and positive integer max_tokens')
    print(json.dumps({'execute':a.execute,'requests':len(cases)*a.repeats,'concurrency':a.concurrency}))
    if not a.execute:
        print('Dry-run only: no network request sent.')
        return
    if a.output.exists():
        p.error('Output exists; use a new path to retain previous observations')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with a.output.open('x',encoding='utf8') as out:
        for repeat in range(a.repeats):
            start=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                pending=[pool.submit(request_once,a.url,a.model,row,a.timeout) for row in cases]
                records=[f.result() for f in pending]
            failed += sum(not r['ok'] for r in records)
            out.write(json.dumps({'repeat':repeat,'wall_s':time.perf_counter()-start,
                                  'concurrency':a.concurrency,'requests':records})+'\n')
            out.flush()
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()

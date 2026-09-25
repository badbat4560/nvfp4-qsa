#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def percentile(values, q):
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def request_once(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", body, {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    first = None
    text = ""
    usage = {}
    with OPENER.open(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                if first is None:
                    first = time.perf_counter()
                text += choices[0]["text"]
    end = time.perf_counter()
    first = first or end
    output_tokens = usage.get("completion_tokens", max_tokens)
    prompt_tokens = usage.get("prompt_tokens")
    ttft = first - start
    decode_time = max(end - first, 1e-9)
    return {
        "ttft_s": ttft,
        "total_s": end - start,
        "decode_s": decode_time,
        "decode_tok_s": max(output_tokens - 1, 0) / decode_time,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "text_prefix": text[:80],
    }


def run_case(url, model, concurrency, prompt, max_tokens):
    wall_start = time.perf_counter()
    nonce = time.time_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                url,
                model,
                f"unique-request-{nonce}-{index} " + ("x " * 32) + prompt,
                max_tokens,
            )
            for index in range(concurrency)
        ]
        requests = [future.result() for future in futures]
    wall = time.perf_counter() - wall_start
    ttfts = [item["ttft_s"] for item in requests]
    decode_rates = [item["decode_tok_s"] for item in requests]
    output_tokens = sum(item["output_tokens"] for item in requests)
    prompt_tokens = sum(item["prompt_tokens"] or 0 for item in requests)
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "prompt_tokens_total": prompt_tokens,
        "output_tokens_total": output_tokens,
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "ttft_max_s": max(ttfts),
        "per_session_decode_tok_s_p50": statistics.median(decode_rates),
        "per_session_decode_tok_s_min": min(decode_rates),
        "aggregate_output_tok_s_e2e": output_tokens / wall,
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    short_prompt = "Continue the numbered sequence with one number per line:\n1\n2\n3\n"
    long_prompt = ("alpha beta gamma delta epsilon zeta eta theta. " * 12000) + "\nSummarize in one word:"

    result = {
        "created_unix": time.time(),
        "url": args.url,
        "decode_single": run_case(args.url, args.model, 1, short_prompt, 256),
        "decode_concurrency_7": run_case(args.url, args.model, 7, short_prompt, 256),
        "prefill_single": run_case(args.url, args.model, 1, long_prompt, 1),
        "prefill_concurrency_7": run_case(args.url, args.model, 7, long_prompt, 1),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

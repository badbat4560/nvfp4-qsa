# Contributing

For a bug report, include the revision, Python/PyTorch/Triton versions, GPU model, driver, tensor shapes and dtypes, and a minimal reproducer. Identify whether the failure is in the portable package, public integration candidate or another serving build. Remove credentials, private prompts and internal endpoints before sharing logs.

Run the relevant CPU contracts first:

```bash
python -m pytest tests/test_cpu_contracts.py tests/test_public_port_contract.py tests/unit -q
```

Changes to the portable CUDA writer/reader also need the GPU suite:

```bash
python -m pytest tests/test_gpu_portable.py -q
```

Keep codec correctness, quantization error and downstream quality as separate checks. For performance results, record configuration, warmup, run order, prompt/output lengths, concurrency and repeated batch measurements. Retain individual observations rather than only a best result.

Do not overwrite historical observations with new runs. Use a new dated result directory. Preserve upstream notices and identify changes to archived source. Applying a vLLM patch successfully is not evidence that its server path works.

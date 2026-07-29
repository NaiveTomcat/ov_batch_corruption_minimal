import os
import random

import openvino as ov
import torch
import numpy as np

from model import build

ONNX_DIR = os.path.join(os.path.dirname(__file__), "onnx")
if not os.path.exists(ONNX_DIR):
    os.makedirs(ONNX_DIR)

N_CALLS = 200

def export_onnx(channels: int = 8, width: int = 8, seed: int = 42) -> str:
    model = build(channels=channels, seed=seed)
    path = os.path.join(ONNX_DIR, f"experiment_{channels}_{width}_{seed}.onnx")
    dummy = torch.randn(1, channels, width, width)
    torch.onnx.export(
        model, (dummy,), path,
        input_names=["x"], output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    return path


def run_test() -> None:
    print("="*80)
    print("Running batch corruption tests with OpenVINO regarding issue #37103")
    print(f"Environment: OpenVINO {ov.__version__}, PyTorch {torch.__version__}")
    print("Creating model and exporting to ONNX...")
    channels = 8
    width = 8
    seed = 42
    onnx_path = export_onnx(channels=channels, width=width, seed=seed)

    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, channels, width, width)).astype(np.float32)

    # Run a baseline test first
    model = ov.Core().read_model(onnx_path)
    compiled = ov.compile_model(model, device_name="CPU")
    infer = compiled.create_infer_request()
    baseline_cpu = np.asarray(infer.infer({"x": sample})["y"])[0]
    compiled = ov.compile_model(model, device_name="GPU")
    infer = compiled.create_infer_request()
    baseline_gpu = np.asarray(infer.infer({"x": sample})["y"])[0]

    assert np.allclose(baseline_cpu, baseline_gpu, rtol=1e-3, atol=1e-3), "Baseline CPU and GPU outputs do not match."

    print("Baseline CPU and GPU outputs match.")
    print("="*80)

    # Run tests with batch size of 8 constantly
    print(f"Running {N_CALLS} tests with constant batch size of 8...")
    batch_sizes = [8] * N_CALLS
    compiled = ov.compile_model(model, device_name="GPU")
    infer = compiled.create_infer_request()
    bad_count = 0
    first_bad = None
    for i, batch_size in enumerate(batch_sizes):
        batch = np.repeat(sample, batch_size, axis=0)
        output = np.asarray(infer.infer({"x": batch})["y"])
        wrong = [j for j in range(batch_size) if not np.allclose(output[j], baseline_gpu, rtol=1e-5, atol=1e-5)]
        if len(wrong):
            bad_count += 1
            if first_bad is None:
                first_bad = (i, batch_size, wrong)
    print(f"Ran {N_CALLS} tests with batch size of 8. Found {bad_count} bad outputs.")
    if first_bad is not None:
        print(f"First bad output at call {first_bad[0]} with batch size {first_bad[1]} and wrong indices {first_bad[2]}.")
        print(f"Previous batch size was {batch_sizes[first_bad[0]-1]}.")
    print("="*80)

    # Run tests with random batch size between 2 and 16
    print(f"Running {N_CALLS} tests with random batch sizes between 2 and 16...")
    batch_sizes = [random.randint(2, 16) for _ in range(N_CALLS)]
    compiled = ov.compile_model(model, device_name="GPU")
    infer = compiled.create_infer_request()
    bad_count = 0
    first_bad = None
    for i, batch_size in enumerate(batch_sizes):
        batch = np.repeat(sample, batch_size, axis=0)
        output = np.asarray(infer.infer({"x": batch})["y"])
        wrong = [j for j in range(batch_size) if not np.allclose(output[j], baseline_gpu, rtol=1e-5, atol=1e-5)]
        if len(wrong):
            bad_count += 1
            if first_bad is None:
                first_bad = (i, batch_size, wrong)
    print(f"Ran {N_CALLS} tests with random batch sizes between 2 and 16. Found {bad_count} bad outputs.")
    if first_bad is not None:
        print(f"First bad output at call {first_bad[0]} with batch size {first_bad[1]} and wrong indices {first_bad[2]}.")
        print(f"Previous batch size was {batch_sizes[first_bad[0]-1]}.")
    print("="*80)

    # Run tests with random batch size between 1 and 16
    print(f"Running {N_CALLS} tests with random batch sizes between 1 and 16...")
    batch_sizes = [random.randint(1, 16) for _ in range(N_CALLS)]
    compiled = ov.compile_model(model, device_name="GPU")
    infer = compiled.create_infer_request()
    bad_count = 0
    first_bad = None
    for i, batch_size in enumerate(batch_sizes):
        batch = np.repeat(sample, batch_size, axis=0)
        output = np.asarray(infer.infer({"x": batch})["y"])
        wrong = [j for j in range(batch_size) if not np.allclose(output[j], baseline_gpu, rtol=1e-5, atol=1e-5)]
        if len(wrong):
            bad_count += 1
            if first_bad is None:
                first_bad = (i, batch_size, wrong)
    print(f"Ran {N_CALLS} tests with batch sizes between 1 and 16. Found {bad_count} bad outputs.")
    if first_bad is not None:
        print(f"First bad output at call {first_bad[0]} with batch size {first_bad[1]} and wrong indices {first_bad[2]}.")
        print(f"Previous batch size was {batch_sizes[first_bad[0]-1]}.")
    print("="*80)


if __name__ == "__main__":
    run_test()

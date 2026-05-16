#!/usr/bin/env python3
"""Benchmark ONNX inference frequency."""
import numpy as np
import onnxruntime as ort
import time
import os

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
policy_path = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/Apr21_01-18-24_/model_5000.onnx"
print(f"Model: {policy_path}")

session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input: {input_name} {input_shape}")
print(f"Output: {output_name} {session.get_outputs()[0].shape}")

dummy = np.zeros((1, 270), dtype=np.float32)

# Warmup
for _ in range(100):
    session.run([output_name], {input_name: dummy})

# Benchmark
N = 10000
t0 = time.perf_counter()
for _ in range(N):
    session.run([output_name], {input_name: dummy})
t1 = time.perf_counter()

total_ms = (t1 - t0) * 1000
per_infer_us = total_ms / N * 1000
freq_hz = N / (t1 - t0)

print(f"\n=== Results ({N} inferences) ===")
print(f"Total time:   {total_ms:.1f} ms")
print(f"Per inference: {per_infer_us:.1f} µs")
print(f"Frequency:    {freq_hz:.0f} Hz ({freq_hz/1000:.1f} kHz)")
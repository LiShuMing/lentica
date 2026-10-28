"""Chapter 02 Week 1: Tensor, autograd, device, and matmul exercises."""

from __future__ import annotations

import argparse
import time

import torch


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def tensor_operations_demo(device: torch.device) -> None:
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([10.0, 20.0, 30.0])
    broadcast_sum = matrix + bias
    transposed = broadcast_sum.permute(1, 0)

    print("\n[tensor operations]")
    print(f"matrix: shape={tuple(matrix.shape)} dtype={matrix.dtype} device={matrix.device}")
    print("broadcast result:\n", broadcast_sum)
    print(f"permute(1, 0): shape={tuple(transposed.shape)}")
    print(f"moved to selected device: {matrix.to(device).device}")


def autograd_demo(device: torch.device) -> None:
    x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    loss = (x.square() + 3.0 * x).mean()
    loss.backward()

    expected = (2.0 * x.detach() + 3.0) / x.numel()
    max_error = (x.grad - expected).abs().max().item()
    if not torch.allclose(x.grad, expected, atol=1e-6):
        raise AssertionError(f"autograd check failed: max error={max_error:.3e}")

    print("\n[autograd]")
    print(f"loss={loss.item():.6f}")
    print("autograd gradient:", x.grad.detach().cpu().tolist())
    print("manual gradient:  ", expected.cpu().tolist())
    print(f"max_abs_error={max_error:.3e}")


def fit_linear_regression(device: torch.device, steps: int = 300) -> None:
    torch.manual_seed(42)
    x = torch.linspace(-1.0, 1.0, 256, device=device).reshape(-1, 1)
    y = 2.5 * x - 0.7 + 0.05 * torch.randn_like(x)

    weight = torch.randn(1, 1, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    learning_rate = 0.1

    for _ in range(steps):
        prediction = x @ weight + bias
        loss = (prediction - y).square().mean()
        loss.backward()

        # Parameter updates must not become part of the next computation graph.
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad

        # Gradients accumulate by default, so clear them before the next step.
        weight.grad.zero_()
        bias.grad.zero_()

    if abs(weight.item() - 2.5) > 0.1 or abs(bias.item() + 0.7) > 0.1:
        raise AssertionError("linear regression did not converge")

    print("\n[linear regression]")
    print(f"weight={weight.item():.4f} (target=2.5)")
    print(f"bias={bias.item():.4f} (target=-0.7)")
    print(f"final_mse={loss.item():.6f}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_matmul(device: torch.device, size: int, repeats: int) -> float:
    left = torch.randn(size, size, device=device)
    right = torch.randn(size, size, device=device)

    for _ in range(3):
        _ = left @ right
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = left @ right
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def matmul_benchmark(size: int, repeats: int) -> None:
    print("\n[matmul benchmark]")
    cpu_ms = benchmark_matmul(torch.device("cpu"), size, repeats)
    print(f"cpu:  {cpu_ms:.3f} ms/op")

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        cuda_ms = benchmark_matmul(cuda, size, repeats)
        print(f"cuda: {cuda_ms:.3f} ms/op")
        print(f"speedup: {cpu_ms / cuda_ms:.2f}x")
    else:
        print("cuda: skipped (CUDA is unavailable)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"selected device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")

    tensor_operations_demo(device)
    autograd_demo(device)
    fit_linear_regression(device)
    matmul_benchmark(args.matrix_size, args.repeats)
    print("\nall Chapter 02 Week 1 checks passed")


if __name__ == "__main__":
    main()

import time

import torch


def time_matmul(device: torch.device, size: int = 2048, rounds: int = 10) -> float:
    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    # Warm up the device and caches.
    _ = a @ b

    start = time.perf_counter()
    for _ in range(rounds):
        c = a @ b
    if device.type != "cpu":
        # DirectML operations are synchronized by materializing a scalar.
        _ = c[0, 0].item()
    end = time.perf_counter()

    return (end - start) / rounds


def main() -> None:
    print(f"torch: {torch.__version__}")

    cpu = torch.device("cpu")
    cpu_time = time_matmul(cpu)
    print(f"cpu matmul: {cpu_time:.6f}s")

    try:
        import torch_directml
    except ImportError:
        print("torch-directml is not installed.")
        return

    dml = torch_directml.device()
    gpu_time = time_matmul(dml)
    print(f"directml matmul: {gpu_time:.6f}s")
    print(f"speedup vs cpu: {cpu_time / gpu_time:.2f}x")


if __name__ == "__main__":
    main()

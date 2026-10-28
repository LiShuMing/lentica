import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"hip version: {getattr(torch.version, 'hip', None)}")
    print(f"cuda api available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No ROCm/HIP GPU is visible to PyTorch.")
        return

    count = torch.cuda.device_count()
    print(f"device count: {count}")
    for index in range(count):
        print(f"device {index}: {torch.cuda.get_device_name(index)}")

    device = torch.device("cuda:0")
    a = torch.randn((1024, 1024), device=device)
    b = torch.randn((1024, 1024), device=device)
    c = a @ b
    print(f"sample result: {c[0, 0].item():.6f}")


if __name__ == "__main__":
    main()

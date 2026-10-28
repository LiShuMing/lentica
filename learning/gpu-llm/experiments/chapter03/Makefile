CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXXFLAGS ?= -O2 -std=c++17
CUDAFLAGS ?= -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib

TARGET := cuda_basics
SRC := cuda_basics.cu

.PHONY: all check-env run clean

all: $(TARGET)

check-env:
	@test -x "$(NVCC)" || (echo "nvcc not found: $(NVCC)"; echo "Run: source ../../scripts/cuda_env.sh"; exit 1)
	@test -f "$(CUDA_HOME)/include/cuda_runtime.h" || (echo "cuda_runtime.h not found under CUDA_HOME=$(CUDA_HOME)"; echo "Run: source ../../scripts/cuda_env.sh"; exit 1)

$(TARGET): $(SRC) | check-env
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# GPU to LLM Learning

Numbered lessons and experiments from NumPy/backpropagation through PyTorch,
GPU programming, Transformers, and fine-tuning. Consolidated source keeps its
learning progress; source presence does not mean a lesson has been completed.
Spec: [0001](../../specs/0001-agentic-workspace/README.md).

| Lessons | Material |
| --- | --- |
| 01 | NumPy/math and two-layer MLP |
| 02 | Tensor/autograd, training, checkpoints, and metrics |
| 03 | CUDA execution, memory, and profiling |
| 04-06 | Deep learning, Transformers, LoRA/QLoRA roadmap |

Use an already selected Python environment. `requirements.txt` is an optional
learning dependency set, not the product runtime. Choose the Torch/CUDA build for
your hardware separately. The migration does not create a venv or download models.

```bash
cd learning/gpu-llm
./scripts/check_env.sh
./scripts/start_jupyter.sh
python experiments/chapter01/two_layer_mlp_numpy.py
```

Set `LENTICA_LEARNING_PYTHON` to select an interpreter, `CUDA_HOME` for an installed
toolkit, and `LENTICA_JUPYTER_PORT` when needed. Jupyter binds loopback. `nvcc` is
optional for CPU/PyTorch study; CUDA C++ experiments need a toolkit and GPU.
Notebook outputs/attachments/execution counts, checkpoints, data, and training
outputs remain untracked. Build native CUDA examples manually only after checking
the hardware/toolchain requirements. CPU tests validate small NumPy arrays and
gradients, not GPU training quality.

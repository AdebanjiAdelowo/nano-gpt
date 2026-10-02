# Running nano-gpt on a remote NVIDIA GPU

The same code runs on Apple Silicon, CPU and NVIDIA CUDA. The device is chosen
with `--device`:

| Use | Flag |
|---|---|
| Mac development (Apple Silicon GPU) | `--device mps` |
| CPU, for reproducibility checks and comparison | `--device cpu` |
| Remote NVIDIA GPU (Colab, Kaggle, Linux workstation) | `--device cuda` |
| Best available: CUDA, then MPS, then CPU | `--device auto` (default) |

Asking for `mps` or `cuda` on a machine that does not have it stops with an
error. It never falls back to another device.

## Where a run writes its files

Every training run writes into its own directory:

```
results/<run-name>/
├── checkpoint.pt    : model weights, model config, vocabulary
├── loss_curve.png   : training and validation loss
└── run_meta.json    : device, software versions, git commit, seed, config, timings
```

Without `--out-dir` the directory is `results/<timestamp>-<device>/`.
`train.py` refuses to write into a directory that already holds a run, so an
earlier result cannot be overwritten. The `loss_curve.png` in the repository
root is the original Apple MPS run and is never written by `train.py`.

`results/` is ignored by git, so experimental runs never enter the repository
by accident. To keep a run, archive the directory, or copy the files you want
to publish to another location and add them deliberately.

## Workflow

```
Mac  →  git push  →  GitHub  →  Colab or Kaggle  →  git clone  →  CUDA training  →  retrieve results
```

### 1. On the Mac

```bash
# quick check that the code works before pushing (about 10 seconds)
python train.py --device mps --max-iters 50 --eval-interval 25 --eval-iters 10 --out-dir results/mac-smoke
python -m pytest tests -q

git add -A
git commit -m "Describe the change"
git push origin main
```

### 2a. On Google Colab

Open `colab/run_cuda.ipynb` from GitHub
(File > Open notebook > GitHub > `AdebanjiAdelowo/nano-gpt`), set
Runtime > Change runtime type > GPU, and run all cells. The notebook shows the
GPU, clones the repository, runs a 50-step smoke test, runs the normal
training, generates text, shows the loss curve and metadata, and zips the run
directory for download.

The equivalent commands, for a Colab cell or any terminal:

```bash
nvidia-smi
git clone https://github.com/AdebanjiAdelowo/nano-gpt.git
cd nano-gpt
pip install -q -r requirements.txt

python train.py --device cuda --max-iters 50 --eval-interval 25 --eval-iters 10 --out-dir results/colab-smoke
python train.py --device cuda --out-dir results/colab-cuda
python generate.py --device cuda --checkpoint results/colab-cuda/checkpoint.pt --max_new_tokens 300
cat results/colab-cuda/run_meta.json
zip -r nano-gpt-colab-cuda.zip results/colab-cuda
```

### 2b. On Kaggle

In the notebook settings choose an accelerator (GPU T4 or P100) and turn
**Internet** on: the clone and the corpus download both need it. Then, in
notebook cells:

```bash
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

%cd /kaggle/working
!git clone https://github.com/AdebanjiAdelowo/nano-gpt.git
%cd nano-gpt
!pip install -q -r requirements.txt

!python train.py --device cuda --max-iters 50 --eval-interval 25 --eval-iters 10 --out-dir results/kaggle-smoke
!python train.py --device cuda --out-dir results/kaggle-cuda
!python generate.py --device cuda --checkpoint results/kaggle-cuda/checkpoint.pt --max_new_tokens 300
!cat results/kaggle-cuda/run_meta.json

%cd /kaggle/working
!zip -r nano-gpt-kaggle-cuda.zip nano-gpt/results/kaggle-cuda
```

`/kaggle/working/nano-gpt-kaggle-cuda.zip` then appears in the Output panel,
where it can be downloaded.

### 3. Retrieve the results

Colab and Kaggle filesystems are temporary: everything is deleted when the
session ends. Before closing the session, download the zip archive, or copy
it to persistent storage (Google Drive on Colab; "Save Version" on Kaggle
keeps the contents of `/kaggle/working`).

Back on the Mac, unzip the archive into `results/`. A checkpoint trained on
CUDA loads on the Mac on either device:

```bash
python generate.py --checkpoint results/colab-cuda/checkpoint.pt --device mps
python generate.py --checkpoint results/colab-cuda/checkpoint.pt --device cpu
```

## Comparing runs across devices

With the same `--seed`, every device starts from the same initial weights,
because the model is built on the CPU before it is moved. On MPS and CUDA the
training batches are also the same sequence: batch indices come from the CPU
random generator, while dropout uses the device's own generator. On CPU,
dropout draws from that same CPU generator, so the batch sequence differs
after the first step. Floating-point arithmetic is device-specific in every
case, so losses agree closely but not exactly. Timings are specific to the
hardware; `run_meta.json` records which hardware produced them.

Only load checkpoints you created yourself: `generate.py` unpickles the model
configuration stored in the checkpoint.

````markdown
# IIT-D RAG (local desktop + HPC)

> Three components, two servers, one app.

- **Chat model:** `llama.cpp` server (e.g., DeepSeek-R1 8B GGUF)  
- **Embedding model:** `llama.cpp` embedding server (e.g., bge-m3 GGUF)  
- **RAG app:** `rag.py` (FAISS + BM25 + Gradio UI)

---

## Table of Contents
- [Overview](#overview)
- [Prereqs](#prereqs)
- [Models](#models-put-these-in-models-on-hpc)
- [Install on your laptop](#install-on-your-laptop)
- [Runbook](#runbook-multiple-terminals)
  - [Terminal 1 — Chat server (HPC)](#terminal-1--start-the-chat-server-on-hpc)
  - [Terminal 2 — Tunnel chat → laptop](#terminal-2--tunnel-chat-to-your-laptop)
  - [Terminal 3 — Embedding server (HPC)](#terminal-3--start-the-embedding-server-on-hpc)
  - [Terminal 4 — Tunnel embeddings → laptop](#terminal-4--tunnel-embedding-to-your-laptop)
  - [(Optional) Terminal 5 — One tunnel for both](#optional-terminal-5--single-tunnel-for-both-servers)
  - [Terminal 6 — Run RAG locally](#terminal-6--run-rag-from-your-laptop)
- [Notes & Troubleshooting](#notes-tips-troubleshooting)
- [Minimal Cheat-Sheet](#5-minimal-cheat-sheet)

---

## Overview

```mermaid
flowchart LR
  A[Laptop] -->|HTTP :7860| D[Gradio UI (rag.py)]
  D -->|/v1/chat (localhost:5000)| B
  D -->|/v1/embeddings (localhost:5001)| C
  B[HPC: llama-server (Chat)] ---|SSH tunnel :5000| A
  C[HPC: llama-server (Embeddings)] ---|SSH tunnel :5001| A
```

All traffic stays inside your laptop ↔ login node ↔ compute nodes via SSH tunnels.

---

## Prereqs

* IIT-Delhi HPC account with interactive job access (`qsub -I`)
* A compiled **`llama.cpp`** on HPC and your **GGUF** models under `~/models/`
* On your **laptop/desktop**: **Python 3.11** with **Conda** (or `pip`)
* Laptop ports **5000** (chat) and **5001** (embeddings) free

### Models (put these in `~/models` on HPC)

* **Chat (pick a quant):**
  [https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/tree/main](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/tree/main)
* **Embeddings (bge-M3 GGUF):**
  [https://huggingface.co/lm-kit/bge-m3-gguf/tree/main](https://huggingface.co/lm-kit/bge-m3-gguf/tree/main)

> \[!TIP]
> Choose a GGUF quant your GPU/CPU can handle: `Q4_K` / `Q5_K` / `Q6_K` / `Q8_0`.

---

## Install on your laptop

```bash
# One-time
conda create -n RAG python=3.11 -y
conda activate RAG

# Deps (either path works)

# A) pip
pip install numpy requests tqdm pypdf pypdfium2 python-docx chardet \
            faiss-cpu rank-bm25 gradio orjson

# B) conda env file
# conda env create -f environment.yml -n RAG
# Get the environment.yml file: [Files/environment.yml](docs/environment.yml)
# Or see use requirements.txt file: [Files/requirements.txt](docs/requirements.txt)
```

Create your working folder:

```bash
mkdir -p ~/RAG_Files/Papers
cd ~/RAG_Files
# put rag.py here [Files/rag.py](docs/rag.py), and place a few PDFs under ~/RAG_Files/Papers/
```

---

## multiple terminals

> All commands are on **your laptop** unless the step says SSH to HPC.
> Keep tunnel terminals open while using the app.

### Terminal 1 — Start the **chat** server on HPC

```bash
# 1) Login
ssh user@hpc.iitd.ac.in

# 2) Request interactive compute node (Skylake GPUs in this example)
qsub -I -l select=1:ncpus=4:mpiprocs=4:ngpus=1:centos=skylake -N llama \
    -l walltime=02:00:00 -P project_name

# 3) Note the node IP
hostname -I    # -> <IP_CHAT>

# 4) Load modules / env (adjust for your cluster)
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$PWD:${LD_LIBRARY_PATH}
module load compiler/cuda/12.3/compilervars
module load compiler/gcc/11.2/openmpi/4.1.6

# 5) Start chat server (adjust paths/models)
# Example uses a Skylake build of llama.cpp
/home/chemical/phd/chz218339/llama.cpp/build/bin/llama-server \
  --model $HOME/models/DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf \
  --host 0.0.0.0 --port 5000 -ngl 999 -t ${OMP_NUM_THREADS:-8} \
  --ctx-size 131072
```

> \[!NOTE]
> If your `llama-server` binary is Skylake-only, keep chat on Skylake nodes.
> Use an IceLake-compatible build if you want to run there.

---

### Terminal 2 — Tunnel **chat** to your laptop

```bash
ssh -L 5000:<IP_CHAT>:5000 user@hpc.iitd.ac.in
```

Open [http://localhost:5000](http://localhost:5000) to see the `llama.cpp` UI.
Quick local test:

```bash
curl -s http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model":"deepseek-r1",
        "messages":[
          {"role":"system","content":"Be concise."},
          {"role":"user","content":"Say hello in one sentence."}
        ],
        "max_tokens":64,
        "add_generation_prompt":true
      }' | jq .
```

---

### Terminal 3 — Start the **embedding** server on HPC

```bash
# 1) Login again
ssh user@hpc.iitd.ac.in

# 2) Request interactive node (IceLake works well; try Skylake if needed)
qsub -I -l select=1:ncpus=4:mpiprocs=4:ngpus=1:centos=icelake -N llama \
    -l walltime=02:00:00 -P project_name

# 3) Note the node IP
hostname -I    # -> <IP_EMBED>

# 4) Modules / env
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$PWD:${LD_LIBRARY_PATH}
module load compiler/cuda/12.3/compilervars
module load compiler/gcc/11.2/openmpi/4.1.6
```

**Preset A (balanced):**

```bash
/home/chemical/phd/chz218339/llama.cpp/build/bin/llama-server \
  -m $HOME/models/bge-m3-Q8_0.gguf --embedding \
  -t 8 -ngl 999 --ctx-size 8192 --parallel 4 \
  --batch-size 8192 --ubatch-size 16384 --mlock \
  --host 0.0.0.0 --port 5001
```

**Preset B (tighter context, higher parallel):**

```bash
/home/chemical/phd/chz218339/llama.cpp/build/bin/llama-server \
  -m $HOME/models/bge-m3-Q8_0.gguf --embedding \
  -t 8 -ngl 999 --ctx-size 2048 --parallel 16 \
  --batch-size 4096 --ubatch-size 2048 --cont-batching --mlock \
  --host 0.0.0.0 --port 5001
```

> \[!WARNING]
> If logs show **“input is larger than the max context size”**, reduce client packing (see env below) or use Preset A.

---

### Terminal 4 — Tunnel **embedding** to your laptop

```bash
ssh -L 5001:<IP_EMBED>:5001 user@hpc.iitd.ac.in
```

Open [http://localhost:5001](http://localhost:5001) (UI shows but won’t chat—this is embeddings).
Quick local test:

```bash
curl -s http://127.0.0.1:5001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":["adsorption energy of toluene on CNT"],"model":"bge-m3"}' | jq .
```

---

### (Optional) Terminal 5 — Single tunnel for **both** servers

```bash
ssh -L 5000:<IP_CHAT>:5000 -L 5001:<IP_EMBED>:5001 user@hpc.iitd.ac.in
```

---

### Terminal 6 — Run RAG from your laptop

```bash
conda activate RAG
cd ~/RAG_Files
```

**Environment vars:**

```bash
# Where the PDFs live
export PDF_DIR=$HOME/RAG_Files/Papers
export RAG_STORE=$HOME/RAG_Files/rag_store

# OpenAI-compatible endpoints (your local tunnels)
export CHAT_URL=http://127.0.0.1:5000
export EMBED_URL=http://127.0.0.1:5001

# Model "names" used by rag.py
export CHAT_MODEL=deepseek-r1
export EMBED_MODEL=bge-m3

# Client packing / timeouts that work well
export PARSE_WORKERS=1
export EMBED_MAX_ITEMS=12
export CHARS_PER_TOKEN=3.2
export EMBED_TIMEOUT=600
```

**Index PDFs:**

```bash
python rag.py index --dir "$PDF_DIR"
```

**Ask a question (CLI):**

```bash
python rag.py ask "What is the paper about?"
```

**Start local UI (Gradio):**

```bash
python rag.py ui
```

Open [http://localhost:7860](http://localhost:7860) and chat. The UI can rebuild/append the index. Read/edit rag.py file for more options.

---

## Notes, Tips, Troubleshooting

* **Paths:** Replace `/home/chemical/phd/chz218339/llama.cpp/...` with your own build path or just use my skylake llama build (currently all HPC users should have read access).
* **Tunnels:** Keep SSH tunnels open the entire time.
* **Throughput vs stability (embeddings):**

  * If you see “exceed context size / n\_ubatch” on the server, lower client packing:

    ```bash
    export EMBED_MAX_ITEMS=8
    export CHARS_PER_TOKEN=3.5
    ```
  * Or reduce server `--parallel`, increase `--ctx-size`, and/or increase `--ubatch-size`.
* **PDFs over SSHFS are slow:** Prefer local disk on the laptop; or copy PDFs to `$SCRATCH` and run `rag.py index` on HPC (then tunnel chat back for the UI).
* **Don’t quote `~`:** Use `$HOME/...` or unquoted `~/...` so the shell expands it.

---

## 5) Minimal cheat-sheet

```bash
# Chat server (HPC; modules loaded)
llama-server --model ~/models/DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf \
  --host 0.0.0.0 --port 5000 -ngl 999 -t 8 --ctx-size 131072

# Embedding server (HPC)
llama-server -m ~/models/bge-m3-Q8_0.gguf --embedding \
  --host 0.0.0.0 --port 5001 -t 8 -ngl 999 --ctx-size 8192 \
  --parallel 4 --batch-size 8192 --ubatch-size 16384

# Tunnels (laptop)
ssh -L 5000:<IP_CHAT>:5000 -L 5001:<IP_EMBED>:5001 user@hpc.iitd.ac.in

# RAG env (laptop)
export PDF_DIR=$HOME/RAG_Files/Papers
export RAG_STORE=$HOME/RAG_Files/rag_store
export CHAT_URL=http://127.0.0.1:5000
export EMBED_URL=http://127.0.0.1:5001
export CHAT_MODEL=deepseek-r1
export EMBED_MODEL=bge-m3
export PARSE_WORKERS=1 EMBED_MAX_ITEMS=12 CHARS_PER_TOKEN=3.2 EMBED_TIMEOUT=600

# Index + UI (laptop)
python rag.py index --dir "$PDF_DIR"
python rag.py ui
```

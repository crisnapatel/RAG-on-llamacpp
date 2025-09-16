# ===============================
# Open-WebUI + llama.cpp (HPC)
# Chat + Embeddings on compute node
# UI on Mac/Linux browser via SSH tunnel
# ===============================

# Note: Be active. User, directory, conda env name, model file pointer, compute node IP address ... 

**Placeholders (what to put):**

* `<USER>` = your HPC username (e.g., `chz218YZX`)
* `<ENV>` = your conda env name (e.g., `RAG`)
* `<HPC_LOGIN_HOST>` = login node you SSH into (e.g., `hpc.iitd.ac.in`)
* `<CN_IP>` = **compute node** IP that runs the servers (e.g., `172.20.7.148`); check with "hostname -I"
* `<LLAMACPP_DIR>` = path to llama.cpp on compute node (e.g., `/home/.../llama.cpp`)
* `<MODELS_DIR>` = directory holding GGUFs (e.g., `/home/.../models`)
* `<CHAT_MODEL_GGUF>` = chat model file (e.g., `Qwen3-30B-A3B-...Q5_K_XL.gguf`)
* `<EMBED_MODEL_GGUF>` = embedding model file (e.g., `bge-m3-Q8_0.gguf`)
* `<PORT_CHAT>` / `<PORT_EMBED>` = ports you exposed (e.g., `5003` / `5004`)
* `<ALIAS_CHAT>` / `<ALIAS_EMBED>` = friendly names (e.g., `qwen3-30b-a3b-think`, `embed-bge-m3`)
* `<OPENAI_API_KEY>` / `<EMBED_API_KEY>` = tokens WebUI will send (e.g., `sk-local`)
* `<DATA_DIR>` = Open WebUI state dir on Mac (e.g., `~/Documents/.open-webui`)


# --- On the compute node ---
# Start CHAT model (Qwen3-30B; tune ctx/offload for your GPU)
./llama-server \
  --model /home/chemical/phd/chz218339/llama.cpp/models/Qwen3-30B-A3B-Thinking-2507-UD-Q5_K_XL.gguf \
  --host 0.0.0.0 --port 5003 \
  --alias qwen3-30b-a3b-think --api-key sk-local \
  -ngl 999 --ctx-size 16384 --batch-size 512 --ubatch-size 256 &

# Start EMBEDDING model (BGE-M3; 1024-d)
./llama-server \
  --model /home/chemical/phd/chz218339/models/bge-m3-Q8_0.gguf \
  --embedding \
  --host 0.0.0.0 --port 5004 \
  --alias embed-bge-m3 --api-key sk-local \
  -ngl 24 --ctx-size 2048 --batch-size 4096 --ubatch-size 1024 &

# Verify ports (on the same compute node)
lsof -i :5003
lsof -i :5004


# --- On the Mac (SSH tunnel to compute node) ---
# Replace <CN_IP> with the compute node IP (e.g., 172.20.7.148)
ssh -N \
  -L 5003:<CN_IP>:5003 \
  -L 5004:<CN_IP>:5004 \
  chz218339@hpc.iitd.ac.in

# --- Checks from Mac ---
curl -s -H "Authorization: Bearer sk-local" http://127.0.0.1:5003/v1/models | jq .
curl -s -H "Authorization: Bearer sk-local" -H 'Content-Type: application/json' \
  -d '{"model":"embed-bge-m3","input":"hello"}' \
  http://127.0.0.1:5004/v1/embeddings | jq '.data[0].embedding | length'
# Expect: 1024


# --- Start Open WebUI on Mac ---
conda activate RAG # Ensure your conda env has all the required libraries to Open-WebUI
export NO_PROXY=127.0.0.1,localhost,::1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export DATA_DIR=~/Documents/.open-webui

# Chat endpoint (tunneled llama.cpp)
export OPENAI_API_BASE_URL=http://127.0.0.1:5003/v1
export OPENAI_API_KEY=sk-local

# Embedding endpoint (tunneled llama.cpp)
export RAG_EMBEDDING_ENGINE=openai
export RAG_OPENAI_API_BASE_URL=http://127.0.0.1:5004/v1
export RAG_OPENAI_API_KEY=sk-local

open-webui serve --port 8080
# Open http://localhost:8080


# --- Minimal UI setup (once) ---
# Settings → Connections → Add → OpenAI:
#   Base URL: http://127.0.0.1:5003/v1
#   API Key : sk-local
# Admin → Documents:
#   Embedding E


### Troubleshooting (short + surgical)

* **Dim mismatch**: `InvalidArgumentError: Collection expecting embedding with dimension of 1024, got 768/384`
  → Admin → Documents: set embedding engine/model to your 1024-d server (`embed-bge-m3`).
  → **Rebuild** the affected Knowledge (or create a new one).
  → Confirm with `curl …/embeddings | jq '.data[0].embedding | length'` = **1024**.

* **Open WebUI still using old settings** (logs say “loaded from latest database entry”):
  → Change settings **in the UI** (Admin → Documents / Settings → Connections). Env vars won’t override stored DB config.

* **“Connection error” to models**:
  → Ensure tunnel is running; `curl http://127.0.0.1:5003/v1/models` must work.
  → Check ports on compute node (`lsof -i :5003`, `:5004`).
  → Confirm you pass `Authorization: Bearer sk-local`.

* **`zsh: command not found: open-webui`**:
  → `conda activate RAG` (where you installed it) or `pip install open-webui`.

* **Port busy**:
  → `lsof -i :8080` (Mac) / `lsof -i :5003,:5004` (compute node) → kill or change port.

* **CUDA OOM** (30B chat):
  → Lower `--batch-size/--ubatch-size`, then reduce `--ctx-size`; keep KV on CPU if needed.

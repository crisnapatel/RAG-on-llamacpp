# RAG on llama.cpp (IIT-D)

**One-liner:** Chat with your **own research papers** using a local RAG pipeline.  
Back end: two `llama.cpp` servers (chat + embeddings) running on IIT-D HPC, tunneled to your laptop.  
Front end: `rag.py` (FAISS + BM25) with a Gradio UI.

> Full setup, HPC commands, and troubleshooting live here: **[docs/INSTRUCTIONS.md](RAG-on-llamacpp/docs/INSTRUCTIONS.md)**
> llama is model manager: https://github.com/ggml-org/llama.cpp


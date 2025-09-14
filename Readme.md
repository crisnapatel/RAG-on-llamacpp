# RAG on llama.cpp (IIT Delhi)

Chat with your **own research papers** using a local RAG pipeline.
Back end: two `llama.cpp` servers (chat + embeddings) running on IIT-D HPC, tunneled to your laptop.  
Front end: `rag.py` (FAISS + BM25) with a Gradio UI.

**It's like using Notebooklm except that you're not limited by the number of files are file size.**

> Full setup, HPC commands, and troubleshooting: [docs/Instructions.md](docs/Instructions.md)
> 
> llama is inference runner (model manager): https://github.com/ggml-org/llama.cpp
>
> Two known issues: Sometimes chat model doesn't respond. This could be the chat model I used.

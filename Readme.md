# RAG on llama.cpp (IIT Delhi)

Chat with your **own research papers** using a local RAG pipeline.
Back end: two `llama.cpp` servers (chat + embeddings) running on IIT-D HPC, tunneled to your laptop.  
Front end: `rag.py` (FAISS + BM25) with a Gradio UI.

**It's like using Notebooklm except that you're not limited by the number of files are file size.**

> Full setup, HPC commands, and troubleshooting: [docs/Instructions.md](docs/Instructions.md)
> 
> llama is inference runner (model manager): https://github.com/ggml-org/llama.cpp
>
> Two known issues: Sometimes chat model doesn't respond. This could be the chat model I used. Chain of thought (raw model output) not working; meaning you cannot see what chat model is thinking.

## Preview

<p align="center">
  <img src="Files/RAG_Web_UI.png" alt="RAG Web UI (chat over your PDFs)" width="85%">
  <br/>
  <em>Gradio UI: ask questions, see sources, manage your corpus.</em>
</p>

<p align="center">
  <img src="Files/Chat_model.png" alt="Chat endpoint sanity-check with curl" width="85%">
  <br/>
  <em>Chat server smoke test via /v1/chat/completions.</em>
</p>

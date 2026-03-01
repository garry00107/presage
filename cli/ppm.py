"""
PPM CLI — command line interface for managing the PPM system.

Commands:
  ppm init                 Initialize stores (create tables, collections)
  ppm ingest <path>        Ingest a file or directory into memory
  ppm chat                 Start interactive chat session
  ppm search <query>       Search memories
  ppm stats                Show system statistics
  ppm export               Export trajectory dataset as JSONL
  ppm serve                Start the API server

Usage:
  python -m cli.ppm init
  python -m cli.ppm ingest ./src/
  python -m cli.ppm chat
  python -m cli.ppm serve --port 8000
"""

import asyncio
import sys
import os
from pathlib import Path


def _print_banner():
    print("""
╔═══════════════════════════════════════╗
║   PPM — Predictive Push Memory v0.1  ║
║   Anticipatory memory brain for LLMs ║
╚═══════════════════════════════════════╝
""")


async def cmd_init():
    """Initialize all stores."""
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.feedback.dataset import TrajectoryDataset

    print("Initializing PPM stores...")

    meta = MetaStore(settings.sqlite_path)
    await meta.connect()

    ds = TrajectoryDataset(meta)
    await ds.initialize()

    print(f"  ✓ SQLite: {settings.sqlite_path}")
    print(f"  ✓ Qdrant: {settings.qdrant_path}")
    print(f"  ✓ Kuzu:   {settings.kuzu_path}")
    print("  ✓ Trajectory dataset table created")
    print("\nPPM initialized. Run `ppm serve` to start the API.")

    await meta.close()


async def cmd_serve(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server with all dependencies."""
    import uvicorn
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.store.vector import QdrantVectorStore
    from core.session.factory import SessionFactory
    from api.server import app
    from api.deps import set_factory
    from api.ws import websocket_endpoint

    # Wire up FastAPI WebSocket route
    from fastapi import WebSocket
    @app.websocket("/v1/ws/{session_id}")
    async def ws_route(websocket: WebSocket, session_id: str):
        await websocket_endpoint(websocket, session_id)

    print("Starting PPM server...")

    # Initialize stores
    meta   = MetaStore(settings.sqlite_path)
    await meta.connect()

    from qdrant_client import AsyncQdrantClient
    qclient = AsyncQdrantClient(path=settings.qdrant_path)
    vector = QdrantVectorStore(qclient, dim=settings.embedder_dim)
    await vector.initialize()

    # Initialize embedder
    if settings.embedder_backend == "openai":
        from adapters.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder()
    elif settings.embedder_backend == "nvidia":
        from adapters.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(
            api_key=settings.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
    else:
        from adapters.embedder.local import LocalEmbedder
        embedder = LocalEmbedder()

    # Initialize LLM caller
    llm_caller = _build_llm_caller(settings)

    # Create factory
    factory = SessionFactory(
        embedder=embedder,
        meta_store=meta,
        vector_store=vector,
        llm_caller=llm_caller,
    )
    set_factory(factory)
    app.state.factory = factory

    print(f"  ✓ Server starting on http://{host}:{port}")
    print(f"  ✓ API docs: http://{host}:{port}/docs")
    print(f"  ✓ WebSocket: ws://{host}:{port}/v1/ws/{{session_id}}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def cmd_chat():
    """Interactive REPL chat session."""
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.store.vector import QdrantVectorStore
    from core.session.factory import SessionFactory

    _print_banner()

    meta   = MetaStore(settings.sqlite_path)
    await meta.connect()
    
    from qdrant_client import AsyncQdrantClient
    qclient = AsyncQdrantClient(path=settings.qdrant_path)
    vector = QdrantVectorStore(qclient, dim=settings.embedder_dim)
    await vector.initialize()

    if settings.embedder_backend == "openai":
        from adapters.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder()
    elif settings.embedder_backend == "nvidia":
        from adapters.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(
            api_key=settings.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
    else:
        from adapters.embedder.local import LocalEmbedder
        embedder = LocalEmbedder()

    llm_caller = _build_llm_caller(settings)

    factory = SessionFactory(
        embedder=embedder,
        meta_store=meta,
        vector_store=vector,
        llm_caller=llm_caller,
    )
    session = await factory.create_session()

    print(f"Session: {session.session_id}")
    print("Type your message. Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        result = await session.turn(user_input)

        print(f"\nPPM [{result.intent.value} | v={result.velocity:.2f} | "
              f"{result.memories_injected} mem | {result.latency_ms:.0f}ms]:")
        print(result.llm_response)
        print()

    await factory.close_all()
    await meta.close()


async def cmd_ingest(path: str):
    """Ingest a file or directory into PPM memory."""
    import httpx

    target = Path(path)
    if not target.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    files = [target] if target.is_file() else list(target.rglob("*"))
    files = [f for f in files if f.is_file()]

    ignore_dirs = {".git", "venv", ".venv", "node_modules", "__pycache__", "build", "dist", ".tox", ".idea", ".vscode"}
    ignore_exts = {".pyc", ".so", ".dylib", ".dll", ".exe", ".bin", ".jpg", ".png", ".pdf", ".zip", ".tar", ".gz", ".pyo"}
    
    clean_files = []
    for f in files:
        if any(part in ignore_dirs for part in f.parts):
            continue
        if f.suffix.lower() in ignore_exts:
            continue
        clean_files.append(f)

    print(f"Ingesting {len(clean_files)} files...")
    
    semaphore = asyncio.Semaphore(10)

    async def process_file(client, f):
        async with semaphore:
            ext = f.suffix.lower()
            source_type_map = {
                ".py": "code", ".js": "code", ".ts": "code",
                ".go": "code", ".rs": "code",
                ".json": "json", ".yaml": "yaml", ".yml": "yaml",
                ".md": "md",
            }
            source_type = source_type_map.get(ext, "prose")
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    return
                resp = await client.post("/v1/ingest", json={
                    "content": content,
                    "source": str(f),
                    "source_type": source_type,
                }, timeout=300.0)
                data = resp.json()
                print(f"  ✓ {f.name}: {data.get('chunks_written', 0)} chunks, "
                      f"{data.get('annotations_written', 0)} annotations")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")

    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        tasks = [process_file(client, f) for f in clean_files]
        await asyncio.gather(*tasks)


async def cmd_stats():
    """Show system statistics."""
    import httpx
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        resp = await client.get("/v1/stats")
        data = resp.json()

    print("PPM System Statistics")
    print("─" * 40)
    print(f"  Active sessions:      {data['active_sessions']}")
    print(f"  Total memories:       {data['total_memories']}")
    print(f"  Total chunks:         {data['total_chunks']}")
    print(f"  Trajectory samples:   {data['trajectory_samples']}")


async def cmd_export(output: str = "trajectory_data.jsonl"):
    """Export trajectory dataset for fine-tuning."""
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.feedback.dataset import TrajectoryDataset

    meta = MetaStore(settings.sqlite_path)
    await meta.connect()
    ds = TrajectoryDataset(meta)
    count = await ds.export_jsonl(output)
    print(f"Exported {count} trajectory samples to {output}")
    await meta.close()


def _build_llm_caller(settings):
    """Build an async LLM caller based on settings."""
    if settings.llm_backend == "anthropic":
        async def anthropic_caller(prompt: str) -> str:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            msg = await client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        return anthropic_caller

    elif settings.llm_backend == "openai":
        async def openai_caller(prompt: str) -> str:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        return openai_caller

    elif settings.llm_backend == "nvidia":
        async def nvidia_caller(prompt: str) -> str:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=settings.nvidia_api_key,
                base_url="https://integrate.api.nvidia.com/v1",
            )
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        return nvidia_caller

    else:
        # Ollama local
        async def ollama_caller(prompt: str) -> str:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": settings.llm_model, "prompt": prompt, "stream": False},
                    timeout=120.0,
                )
                return resp.json().get("response", "")
        return ollama_caller


def main():
    """CLI entry point."""
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    rest = args[1:]

    commands = {
        "init":   lambda: asyncio.run(cmd_init()),
        "serve":  lambda: asyncio.run(cmd_serve(
            port=int(rest[1]) if len(rest) > 1 else 8000
        )),
        "chat":   lambda: asyncio.run(cmd_chat()),
        "ingest": lambda: asyncio.run(cmd_ingest(rest[0]) if rest else print("Usage: ppm ingest <path>")),
        "stats":  lambda: asyncio.run(cmd_stats()),
        "export": lambda: asyncio.run(cmd_export(rest[0] if rest else "trajectory_data.jsonl")),
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    commands[cmd]()


if __name__ == "__main__":
    main()


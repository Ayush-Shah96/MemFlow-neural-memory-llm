from __future__ import annotations

import json
from pathlib import Path
import gradio as gr

from pipeline import MemoryPipeline
from visualization.plot import build_graph_figure

pipeline = MemoryPipeline()


def stats_text() -> str:
    s = pipeline.stats()
    return (
        f"Documents: {s['documents']}  |  Chunks: {s['chunks']}  |  "
        f"Memory neurons: {s['entities']}  |  Synapses: {s['relationships']}  |  "
        f"Learned synapses: {s['learned_synapses']}  |  Components: {s['connected_components']}"
    )


def ingest(files):
    if not files:
        return stats_text(), "No files selected.", build_graph_figure(pipeline.graph)
    results = []
    for file_obj in files:
        try:
            result = pipeline.ingest_file(Path(file_obj))
            results.append(
                f"✅ {result['source']}: {result['chunks']} chunks, "
                f"{result['entities']} neurons, {result['relationships']} synapses"
            )
        except Exception as exc:
            results.append(f"❌ {Path(file_obj).name}: {exc}")
    return stats_text(), "\n".join(results), build_graph_figure(pipeline.graph)


def chat(message, history, debug):
    if not message or not message.strip():
        return history, ""
    try:
        result = pipeline.answer(message.strip())
        answer = result["answer"]
        sources = "\n\nSources:\n" + "\n".join(f"- {s}" for s in result["sources"]) if result["sources"] else ""
        rendered = answer + sources
        if debug:
            rendered += "\n\n--- Neural Memory Trace ---\n" + json.dumps(
                {
                    "detected_entities": result["detected_entities"],
                    "activations": result["activations"][:12],
                    "activation_trace": result["activation_trace"][:20],
                },
                indent=2,
                ensure_ascii=False,
            )
    except Exception as exc:
        rendered = f"⚠️ {exc}"

    history = history or []
    history.append({"role": "user", "content": message.strip()})
    history.append({"role": "assistant", "content": rendered})
    return history, ""


def refresh_graph():
    return stats_text(), build_graph_figure(pipeline.graph)


def build_app():
    with gr.Blocks(title="Local Neural Memory LLM") as demo:
        gr.Markdown(
            "# 🧠 Local Neural Memory LLM\n"
            "Ingest documents into persistent associative memory, recall them through spreading activation, "
            "and generate natural-language answers with a local Ollama LLM."
        )
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(height=560, label="Neural Memory Chat")
                with gr.Row():
                    message = gr.Textbox(
                        label="Question",
                        placeholder="Ask anything about the documents you ingested...",
                        scale=6,
                    )
                    debug = gr.Checkbox(label="Show neural-memory trace", value=False, scale=1)
                with gr.Row():
                    send = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
                send.click(chat, [message, chatbot, debug], [chatbot, message])
                message.submit(chat, [message, chatbot, debug], [chatbot, message])
                clear.click(lambda: ([], ""), outputs=[chatbot, message])

            with gr.Tab("Ingestion & Memory"):
                gr.Markdown(
                    "Upload PDF, DOCX, TXT, or Markdown files. The local LLM extracts concepts and relationships "
                    "which become persistent memory neurons and weighted synapses."
                )
                files = gr.File(
                    file_count="multiple",
                    type="filepath",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    label="Documents",
                )
                ingest_btn = gr.Button("Build / Update Memory", variant="primary")
                status = gr.Markdown(stats_text())
                log = gr.Textbox(label="Ingestion log", lines=8)
                graph_plot = gr.Plot(build_graph_figure(pipeline.graph), label="Associative Memory Graph")
                refresh = gr.Button("Refresh memory graph")
                ingest_btn.click(ingest, [files], [status, log, graph_plot])
                refresh.click(refresh_graph, outputs=[status, graph_plot])
        return demo

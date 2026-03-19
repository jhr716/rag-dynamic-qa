import gradio as gr
from rag_dynamic import rag_chain
import time

def answer(q):
    if not q.strip():
        return "请输入问题"
    start = time.time()
    return f"{rag_chain.invoke(q)}\n\n⏱️ {time.time()-start:.1f}秒"

with gr.Blocks(title="RAG问答") as demo:
    gr.Markdown("# RAG问答系统")
    with gr.Row():
        q = gr.Textbox(label="问题", lines=2, scale=4)
        btn = gr.Button("提交", scale=1)
    a = gr.Textbox(label="答案", lines=6)
    btn.click(fn=answer, inputs=q, outputs=a)
    gr.Examples(["什么是深度学习？", "LSTM和GRU有什么区别？"], inputs=q)

demo.launch()
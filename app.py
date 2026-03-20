import gradio as gr
from rag_dynamic import rag_chain, retriever
import time

def answer_question(question, show_sources):
    if not question.strip():
        return "请输入问题", ""
    
    start = time.time()
    
    # 检索文档
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    elapsed = time.time() - start
    
    # 准备来源文本
    sources_text = ""
    if show_sources and docs:
        sources_text = "## 📚 参考来源\n\n"
        for i, doc in enumerate(docs):
            sources_text += f"### 来源 {i+1}\n```\n{doc.page_content}\n```\n\n"
    
    return f"{answer}\n\n⏱️ {elapsed:.1f}秒", sources_text

# 创建界面
with gr.Blocks(title="RAG问答系统") as demo:
    gr.Markdown("# 📚 RAG问答系统")
    
    with gr.Row():
        question = gr.Textbox(
            label="问题",
            placeholder="例如：什么是深度学习？",
            lines=2,
            scale=4
        )
        show_sources = gr.Checkbox(label="显示参考来源", value=True)
    
    with gr.Row():
        submit = gr.Button("提交", variant="primary")
    
    answer = gr.Textbox(label="答案", lines=6)
    sources = gr.Markdown(label="参考来源")
    
    submit.click(
        fn=answer_question, 
        inputs=[question, show_sources], 
        outputs=[answer, sources]
    )
    
    gr.Examples(
        examples=[
            "什么是深度学习？",
            "LSTM和GRU有什么区别？",
            "CNN中的池化操作有什么作用？"
        ],
        inputs=question
    )

if __name__ == "__main__":
    demo.launch()

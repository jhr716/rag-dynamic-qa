from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import time
import shutil
import os

# ========== 配置区 ==========
DOCUMENT_FILE = "summary.txt"  # 你的文档
TEST_QUESTIONS = [
    "什么是深度学习？",
    "CNN中的池化操作有什么作用？",
    "LSTM和GRU有什么区别？",
    "ResNet解决了什么问题？是怎么解决的？"
]

# 要测试的 chunk_size 组合
CHUNK_SIZES = [200, 500, 800]
CHUNK_OVERLAP = 50
# ===========================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_experiment(chunk_size):
    """用指定 chunk_size 运行一次实验"""
    print(f"\n{'='*60}")
    print(f"测试 chunk_size = {chunk_size}")
    print('='*60)
    
    # 1. 加载文档
    loader = TextLoader(DOCUMENT_FILE, encoding="utf-8")
    documents = loader.load()
    
    # 2. 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"生成 {len(chunks)} 个文本块")
    
    # 3. 创建向量库
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"./chroma_db_{chunk_size}"
    )
    
    # 4. 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 5. 初始化LLM
    llm = OllamaLLM(model="deepseek-r1:7b")
    
    # 6. 严格提示词
    prompt = ChatPromptTemplate.from_template(
        """严格基于以下上下文回答问题。如果上下文中没有相关信息，必须回答"根据资料无法回答"。

上下文：{context}

问题：{question}

回答："""
    )
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    # 7. 测试每个问题
    results = []
    for q in TEST_QUESTIONS:
        print(f"\n{'─'*40}")
        print(f"▶ 问题: {q}")
        print('─'*40)
        
        # 检索相关文档
        docs = retriever.invoke(q)
        print(f"  检索到 {len(docs)} 个相关块")
        for i, doc in enumerate(docs):
            print(f"    块{i+1}: {doc.page_content[:100]}...")
        
        # 生成答案
        print("  生成答案中...")
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        
        print(f"\n  ✅ 答案: {answer}")
        print(f"  ⏱️  耗时: {elapsed:.2f} 秒")
        
        results.append({
            "question": q,
            "answer": answer,
            "retrieved_chunks": [doc.page_content for doc in docs],
            "time": round(elapsed, 2)
        })
    
    # 8. 清理
    try:
        shutil.rmtree(f"./chroma_db_{chunk_size}")
    except:
        pass
    
    return results

def main():
    all_results = {}
    
    for size in CHUNK_SIZES:
        results = run_experiment(size)
        all_results[size] = results
        
        input("\n按回车继续下一个size...")
    
    # 打印对比表格
    print("\n\n" + "="*80)
    print("📊 实验结果汇总表")
    print("="*80)
    
    for size in CHUNK_SIZES:
        print(f"\n【chunk_size = {size}】")
        print("-"*60)
        for r in all_results[size]:
            print(f"问题: {r['question']}")
            print(f"答案: {r['answer'][:150]}...")
            print(f"耗时: {r['time']}秒")
            print("-"*40)

if __name__ == "__main__":
    main()

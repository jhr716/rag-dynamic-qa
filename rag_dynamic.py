from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 配置
DOCUMENT_FILE = "summary.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 6

# 全局变量
rag_chain = None
retriever = None  # 保存检索器供溯源使用

def get_vectorstore():
    """获取向量数据库（存在则加载，不存在则创建）"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists("./chroma_db"):
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 创建新数据库
    docs = TextLoader(DOCUMENT_FILE, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    ).split_documents(docs)
    
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 初始化
print("初始化RAG系统...")
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
llm = OllamaLLM(model="deepseek-r1:7b")

prompt = ChatPromptTemplate.from_template("""基于以下上下文回答问题。如果找不到相关信息，回答"根据资料无法回答"。

上下文：{context}

问题：{question}

回答：""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("初始化完成！")

def answer_with_sources(question):
    """回答问题并返回答案和参考来源"""
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return answer, docs

def main():
    while True:
        question = input("\n📝 问题: ").strip()
        if question in ["退出", "exit", "quit"]:
            break
        if not question:
            continue
        
        print("🔍 思考中...")
        answer, docs = answer_with_sources(question)
        
        print(f"\n💡 {answer}")
        
        # 询问是否查看来源
        show = input("\n📖 查看参考来源？(y/n): ").strip().lower()
        if show == 'y':
            print("\n--- 参考来源 ---")
            for i, doc in enumerate(docs):
                print(f"\n[来源 {i+1}]:")
                print(doc.page_content)

if __name__ == "__main__":
    main()

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

prompt = ChatPromptTemplate.from_template("""基于上下文回答问题。如果找不到，回答"无法回答"。

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

def main():
    while True:
        q = input("\n问题: ").strip()
        if q in ["退出", "exit", "quit"]:
            break
        if q:
            print(f"\n答案: {rag_chain.invoke(q)}")

if __name__ == "__main__":
    main()

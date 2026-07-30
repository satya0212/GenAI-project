import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "Qwen/Qwen2.5-7B-Instruct"   # swapped from Mistral-7B-Instruct-v0.3


def load_llm(huggingface_repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        provider="auto"     # let HF pick whichever provider currently hosts this model
    )
    return ChatHuggingFace(llm=endpoint)


custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

prompt = ChatPromptTemplate.from_template(custom_prompt_template)

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 3})

llm = load_llm(huggingface_repo_id)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    user_query = input("Write Query Here: ")

    source_docs = retriever.invoke(user_query)
    answer = rag_chain.invoke(user_query)

    print("\nRESULT: ", answer)
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(source_docs, start=1):
        print(f"\n[{i}] {doc.page_content[:300]}...")
        print(f"    Source: {doc.metadata.get('source', 'unknown')}")
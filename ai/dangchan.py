import os

from openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# # 검색결과 5개 가져오겠다는 의미
# os.environ["TAVILY_API_KEY"] = "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"


def load_data(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
        encoding_name='cl100k_base'
    )
    return text_splitter.split_documents(text)


def create_embeddings():
    embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )
    return embeddings_model


def create_vector_store(documents, embeddings_model):
    vectorstore = FAISS.from_documents(
        documents,
        embedding=embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )
    vectorstore.save_local('./db/faiss')
    return vectorstore


def load_vector_store(vectorstore_file: str):
    loaded_vectorstore = FAISS.load_local(vectorstore_file, embeddings_model, allow_dangerous_deserialization=True)
    return loaded_vectorstore

#pdf 기반 검색도구
def retrieve_documents(vectorstore, query, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={'k': k})
    return retriever


def format_documents(docs):
    return '\n\n'.join([d.page_content for d in docs])


def run_query(vectorstore, query, chat_history):
    docs = retrieve_documents(vectorstore, query)

    # template = '''context에 있는 정보를 토대로 question에 대답해 한글로! 무조건 한글로! 50글자 안으로 답변해줘. 너는 챗봇이야!:
    # {context}
    # Question: {question}
    # '''
    
    # ChatOpenAI 모델로 변경하기 
    llm = Ollama(model="llama3", base_url = "https://dd35-35-227-49-172.ngrok-free.app/")
    
    retriever = vectorstore.as_retriever()
    
    # chat_history
    # prompt = ChatPromptTemplate.from_messages([
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("user", "{input}"),
    #     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    # ])
    # # 이전 대화를 기억하고 해당 내역을 바탕으로 검색을 수행하는 체인 생성
    # retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
   
    # response_1 = retriever_chain.invoke({
    #     "chat_history" : chat_history,
    #     "input" : query
    # })
    # # response = chain.invoke({'context': format_documents(docs), 'question': query})
    
    # chat_history.append(HumanMessage(content=response_1["input"]))
    # chat_history.append(AIMessage(content=response_1["answer"]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    # 1 https://catchuplangchain.streamlit.app/LC_Quickstart_03_ConversationRetrievalChain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": query,
        "context": docs,
    })

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["answer"]))

    # 2  https://velog.io/@udonehn/RAG%EB%A5%BC-%EC%A0%81%EC%9A%A9%ED%95%9C-%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5-%EC%B1%97%EB%B4%87-LangChanin

    # chain = (
    # {
    #     "context": retriever,
    #     "question": RunnablePassthrough(),
    # }
    # | prompt
    # | llm
    # )
    # response = chain.invoke(query)
    return response["answer"], chat_history


if __name__ == "__main__":
    file_path = 'service_details4.pdf'
    query = '1'
    vectorstore_file = './db/faiss'


    data = load_data(file_path)
    documents = split_text_into_chunks(data)
    embeddings_model = create_embeddings()
    # if os.path.exists(vectorstore_file):
    #     vectorstore = load_vector_store(vectorstore_file)
    # else:
    vectorstore = create_vector_store(documents, embeddings_model)
    
    chat_history = [HumanMessage(content="안녕 나는 22살 김해연이야. 청년 자립을 위한 정보를 알려줄 수 있어?"), AIMessage(content="Yes!")]

    while query != 'q':
        query = input("입력해주세요 : ")
        response, chat_history = run_query(vectorstore, query, chat_history)
        print(response)




import os

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_data(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
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
    loaded_vectorstore = FAISS.load_local(vectorstore_file, Embeddings_model, allow_dangerous_deserialization=True)
    return loaded_vectorstore


def retrieve_documents(vectorstore, query, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={'k': k})
    return retriever.get_relevant_documents(query)


def format_documents(docs):
    return '\n\n'.join([d.page_content for d in docs])


def run_query(vectorstore, query):
    global conversation_history
    docs = retrieve_documents(vectorstore, query)

    template = '''
    context와 chat_history 있는 정보를 토대로 question에 대답해 한글로! 무조건 한글로! 50글자 안으로 답변해줘. 너는 "당찬" 이라는 챗봇이야!:
    context : {context},
    chat_history : {history},
    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="llama3")
    chain = prompt | llm | StrOutputParser()
    
    history = '\n'.join(conversation_history)
    
    if len(conversation_history) >= 10:
        conversation_history = conversation_history[2:]
        
    response = chain.invoke({'context': format_documents(docs), 'question': query, 'history':history})
    conversation_history.append(f'질문 : {query}')
    conversation_history.append(f'답변 : {response}')
    return response


conversation_history = []
file_path = 'service_details_v5.pdf'
vectorstore_file = './db/faiss'

data = load_data(file_path)
documents = split_text_into_chunks(data)
Embeddings_model = create_embeddings()
Vectorstore = create_vector_store(documents, Embeddings_model)


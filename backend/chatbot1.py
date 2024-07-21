from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory



# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("service_details4.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")

embeddings = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )

vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an korean assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean. You can write only korean!!!

#Question: 
{question} 

#Answer:"""
)

llm = Ollama(model="llama3")

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    input_key="question",
    verbose=False,
)

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
# question = "나는 자립 준비 청년인데 나에게 맞는 보조금 정보를 알려줘!"
while True:
    question = input('프롬프트 : ')
    if question == 'q':
        break
    response = conversation.invoke(question)
    print(response)
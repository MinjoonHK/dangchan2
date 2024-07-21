import os

from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="dangchan"

os.environ['OPENAI_API_KEY'] = ''

loader = PyMuPDFLoader("2024young.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

hf = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
vectorstore = FAISS.from_documents(documents=splits, embedding=hf)

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

text = """
당신은 자립청년에게 여러 유용한 정보를 알려주는 "당찬이"입니다. 

주어진 문맥(context)을 바탕으로 질문(question)에 답해주세요. 
# 질문: 
{question}

# 제공된 정보:
{context}

# 답변:
"""

prompt = PromptTemplate.from_template(text)

llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": (lambda x: format_docs(x["context"])), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

test_question = "디딤돌 씨앗통장이 뭐야 "
response = rag_chain.invoke({"question": test_question})

print(f"\n질문: {test_question}")
print(f"응답: {response}")


# messages =[
#     {"role": "user", "content": "안녕, 만나서 반가워."}
# ]

# model = "gpt-3.5-turbo"

# stream = client.chat.completions.create(
#     model=model,
#     messages=messages,
#     stream=True,
# )

# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content)

# class GptAPI():
#     def __init__(self, model, client):
#         self.message = []
#         self.model = model
#         self.client = client

#     def get_message(self, prompt):
#         self.messages.append({"role": "user", "content": prompt})

#         stream = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         stream=True,
#         )

#         result = ''
#         for chunk in stream:
#             if chunk.choices[0].delta.content is not None:
#                 string = chunk.choices[0].delta.content
#                 print(string, end = "")
#                 result = ''.join([result, string])
#         # 이전 대화들을 저장해놓고 기억해둠. 
#         self.message.append({"role":"system", "content":result})
        



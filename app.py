import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# .env 파일 경로를 명시적으로 지정 (절대 경로를 사용)
load_result = load_dotenv(dotenv_path=r"c:\Users\dearh\gafl_chat.env")

# 환경 변수 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# 단계 1: 문서 로드(Load Documents)
loader1 = PyMuPDFLoader("test.pdf")
loader2 = PyMuPDFLoader("학교생활인권규정.pdf")
loader3 = PyMuPDFLoader("아스그수강비규정.pdf")
loader4 = PyMuPDFLoader("외출외박규정.pdf")
loader5 = PyMuPDFLoader("자기주도학습규정.pdf")

docs1 = loader1.load()
docs2 = loader2.load()
docs3 = loader3.load()
docs4 = loader4.load()
docs5 = loader5.load()
all_docs = docs1 + docs2 + docs3 + docs4 + docs5

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(all_docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어 생성
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
retriever = vectorstore.as_retriever()

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context} 

#Question: 
{question} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성 (GPT-4 스트리밍 활성화)
llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)

# Streamlit UI 구성
st.title("I am G_Bot")
st.write("")

st.write("학교에 대한 질문을 입력하세요. G_Bot이 답변해드립니다.")
st.write("다만, 조금 느리니 참고 30초만 기다려 주세요. 점차 개선하겠습니다.")
st.write("")

st.sidebar.markdown("G_Bot(GAFL ChatBot) 참고자료")
st.sidebar.markdown("- 학교생활인권규정")
st.sidebar.markdown("- 자기주도학습규정")
st.sidebar.markdown("- 외출외박규정")
st.sidebar.markdown("- ASG수강비규정")


# 사용자로부터 질문을 입력받습니다.
user_question = st.text_input("질문을 입력하세요:", "")

# 실시간 응답을 위한 함수
def stream_response(answer_stream):
    response_text = ""
    response_container = st.empty()  # 빈 컨테이너 생성

    for chunk in answer_stream:
        if hasattr(chunk, 'content'):  # AIMessageChunk 객체에서 content 속성 추출
            response_text += chunk.content
            formatted_text = response_text.replace("\n", "  \n")  # 마크다운 줄바꿈 적용
            response_container.markdown(formatted_text)  # 마크다운으로 출력

# 사용자가 질문을 입력했을 경우
if user_question:
    # 검색기에서 관련 문서 청크를 가져옵니다.
    relevant_docs = retriever.get_relevant_documents(user_question)

    # 검색된 문서에서 컨텍스트를 생성합니다.
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # 프롬프트 생성 (문자열 형식으로)
    prompt_text = prompt.format(context=context, question=user_question)

    # GPT 모델에 스트리밍 방식으로 질의
    answer_stream = llm.stream(prompt_text)

    # 실시간 응답 출력
    stream_response(answer_stream)














# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.embeddings.openai import OpenAIEmbeddings

# # .env 파일 경로를 명시적으로 지정 (절대 경로를 사용)
# load_result = load_dotenv(dotenv_path=r"c:\Users\dearh\gafl_chat.env")

# # 환경 변수 가져오기
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # 단계 1: 문서 로드(Load Documents)
# loader1 = PyMuPDFLoader("test.pdf")
# loader2 = PyMuPDFLoader("학교생활인권규정.pdf")
# loader3 = PyMuPDFLoader("아스그수강비규정.pdf")
# loader4 = PyMuPDFLoader("외출외박규정.pdf")
# loader5 = PyMuPDFLoader("자기주도학습규정.pdf")

# docs1 = loader1.load()
# docs2 = loader2.load()
# docs3 = loader3.load()
# docs4 = loader4.load()
# docs5 = loader5.load()
# all_docs = docs1 + docs2 + docs3 + docs4 + docs5




# # 단계 2: 문서 분할(Split Documents)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split_documents = text_splitter.split_documents(all_docs)


# # 단계 3: 임베딩(Embedding) 생성
# embeddings = OpenAIEmbeddings()

# # 단계 4: DB 생성(Create DB) 및 저장
# # 벡터스토어 생성
# vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# # 단계 5: 검색기(Retriever) 생성
# retriever = vectorstore.as_retriever()

# # 프롬프트 템플릿 생성
# prompt = PromptTemplate.from_template(
#     """You are an assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know. 
# Answer in Korean.

# #Context: 
# {context} 

# #Question: 
# {question} 

# #Answer:"""
# )

# # 단계 7: 언어모델(LLM) 생성 (gpt-4로 수정)
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# # 체인 생성
# chain = LLMChain(llm=llm, prompt=prompt)

# # Streamlit UI 구성
# st.title("문서 기반 GPT 챗봇")
# st.write("문서에 대한 질문을 입력하세요. GPT가 답변해드립니다.")

# # 사용자로부터 질문을 입력받습니다.
# user_question = st.text_input("질문을 입력하세요:", "")

# # 사용자가 질문을 입력했을 경우
# if user_question:
#     # 검색기에서 관련 문서 청크를 가져옵니다.
#     relevant_docs = retriever.get_relevant_documents(user_question)

#     # 검색된 문서에서 컨텍스트를 생성합니다.
#     context = "\n".join([doc.page_content for doc in relevant_docs])

#     # GPT 체인을 사용하여 응답을 생성합니다.
#     response = chain.run({"context": context, "question": user_question})

#     # Streamlit 화면에 응답을 출력합니다.
#     st.write(f"질문: {user_question}")
#     st.write(f"답변: {response}")
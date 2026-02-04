### [1] 필요 라이브러리 import

# API KEY 호출
import os
from dotenv import load_dotenv

# 원본 파일 정리
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 텍스트 임베딩 및 VectorDB
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Chain 구축
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Reranker, Reorder를 통한 RAG 고도화
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# streamlit을 통해 웹사이트 생성
import streamlit as st
import datetime as dt
from streamlit_lottie import st_lottie
import requests

### [2] 환경변수 불러오기
load_dotenv() # 현재 디렉토리의 .env 파일을 읽어 os 환경변수에 넣어줌
# os 환경변수를 불러옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

### [3] 문서 불러와 VectorDB에 저장
# Path 라이브러리를 통해 경로를 문자열이 아닌 객체로 다뤄 이식성 확보
BASE_DATA_DIR = Path("..") / "data"
PDF_DIR, HTML_DIR, WORD_DIR, CSV_DIR = [BASE_DATA_DIR / i for i in ["pdf", "html", "word", "csv"]]

# 독립적인 행(row)으로 구성돼있던 원본 출석 데이터를 학생별로 그룹화하여 Document 리스트로 반환하는 함수
def create_grouped_documents(csv_path: str) -> list[Document]:
    try:
        df = pd.read_csv(csv_path, encoding='cp949')
        required_cols = ['이름', '사유', '날짜', '부재시간', '상태']
        df = df[required_cols].fillna('')
        
        documents = []
        for name, group_df in df.groupby('이름'):
            records = "\n".join([f"사유: {r['사유']}, 날짜: {r['날짜']}, 상태: {r['상태']}, 부재시간: {r['부재시간']}" 
                                    for _, r in group_df.iterrows()])
            documents.append(Document(
                page_content=f"학생 이름: {name}\n\n--- 전체 출결 기록 시작 ---\n{records}",
                metadata={'학생이름': name, '총기록수': len(group_df)}
            ))
        return documents
    except Exception as e:
        st.error(f"일정표 로딩 실패: {e}")
        return []

# Loader로 문서 불러와 VectorDB에 저장하는 함수
@st.cache_resource
def get_vectorstore():
    # Loader => 파일 읽을 준비 (파싱 전략)
    # load => 실제로 읽어 Document 객체 생성
    # RecursiveCharacterTextSplitter => 문서를 어떻게 분할할건지 설정
    # split_documents => metadata 유지하면서 Document 객체를 더 작은 단위의 Document 객체로 분할

    embeddings = OpenAIEmbeddings(model_name="text-embedding-3-large")
    # 텍스트 분할기
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", " ", ""]
    )

    all_docs = []

    # 통합 Loader 설정
    # (로더 클래스, 파일 경로 리스트, 소스 타입 이름)
    file_configs = [
        (PyPDFLoader, list(PDF_DIR.glob("*.pdf")), "pdf"),
        (Docx2txtLoader, list(WORD_DIR.glob("*.docx")), "word"),
        (UnstructuredHTMLLoader, list(HTML_DIR.glob("*.html")), "html"),
        (CSVLoader, [CSV_DIR / "데싸 5기 동료들.csv", CSV_DIR / "데싸 5기 운영진.csv"], "csv")
    ]

    # 반복문을 통한 효율적 로딩
    for loader_cls, paths, s_type in file_configs:
        for path in paths:
            try:
                # CSVLoader의 경우 cp949로 encoding 해야 글자가 안깨지므로 별도 처리
                loader = loader_cls(str(path), encoding='cp949') if loader_cls == CSVLoader else loader_cls(str(path))
                loaded_pages = loader.load()
                
                # 메타데이터 주입 및 리스트 통합
                for d in loaded_pages:
                    d.metadata["source_type"] = s_type
                    d.metadata["source"] = path.name
                
                all_docs.extend(text_splitter.split_documents(loaded_pages))
            except Exception as e:
                print(f"로딩 실패 ({path}): {e}")

    # 특수 로직이 필요한 데이터 처리 (일정표)
    attendance_path = CSV_DIR / "데싸 5기 일정표.csv"
    if attendance_path.exists():
        # 교육생 기준으로 그룹화 하는 함수 호출
        attendance_docs = create_grouped_documents(str(attendance_path))
        # 이름별로 묶었는데 한 명에 대한 데이터가 너무 많으면 컨텍스트 윈도우 문제, 토큰 문제 존재하고 확장성에 불리
        # metadata에 이름 있고, 동일 이름끼리 근처에 있으므로 split 해도 성능 저하 없음
        all_docs.extend(text_splitter.split_documents(attendance_docs))

    # 벡터 DB 일괄 생성
    vectorstore = Chroma.from_documents(
        documents=all_docs, 
        embedding=embeddings
    )
    
    return vectorstore

### [4] Retriever 설계
# Retrieval과 Chain을 설계하는 함수
@st.cache_resource
def get_conversational_rag_chain():
    vectorstore = get_vectorstore()
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    # base_retriever => mmr을 사용하여 질문에 대한 여러 관점의 정보를 모아 유사성과 다양성을 고려한 문서 검색
    # Rerank => base_retriever 에서 고른 후보 25개 중, 질문에 실제로 답할 수 있는 문서 10개 추출
    # Reorder => LLM은 처음과 끝은 잘 기억하지만 중간에 있는 정보는 잘 놓치는 in the middle 현상이 있어 reranker가 골라준 top 10문서에 대해 중요한 정보들은 LLM의 시선이 집중되는 양 끝단에 배치

    base_retriever = vectorstore.as_retriever(search_type="mmr",  search_kwargs={"lambda_mult": 0.5, "fetch_k": 50, "k": 25})
    compressor = DocumentCompressorPipeline(transformers=[
        CohereRerank(model="rerank-multilingual-v3.0", top_n=10),
        LongContextReorder()
    ])
    
    # base_retriever에 reorder와 rerank가 추가된 retriever
    upgraded_retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

    ### [5] Chain 구성
    # 질문 재작성을 위한 프롬프트
    # 질문이 "그 사람의 MBTI"라면, LLM에게 넘기기 전에 그 사람이 누군지 이전 대화 맥락을 참고하여 질문을 재작성하여 넘겨줌
    rephrase_system_prompt = """
        당신은 '질문 재작성기'입니다.
        1. 이전 대화 맥락을 참고하여, 사용자의 모호한 최신 질문을 '독립적인 질문'으로 다시 작성하세요.
        2. 절대로 질문에 답변하지 마세요. 
        3. 당신의 내부 지식을 활용해 인물을 설명하지 마세요. (예: '배우 손호진' (X) -> '손호진 수강생' (O))
        4. 반드시 '문장'이 아닌 '질문(~인가요?, ~입니까?)' 형태로만 출력하세요.
    """

    # 질문 재작성 가이드
    # "질문 재작성을 위한 구성판은 이렇게 생겼어."
    # system, human, ai 등의 역할을 명확히 구분함으로써 지시사항을 더 잘따름
    # system은 뼈대를 잘 잡아줌
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_rewriter = rephrase_prompt | llm | StrOutputParser()

    def retrieve_documents(input_dict):
        query = question_rewriter.invoke(input_dict) if input_dict.get("chat_history") else input_dict["input"]
        return upgraded_retriever.invoke(query)

    # 최종 답변 방식 지침에 관한 프롬프트
    qa_system_prompt = """
    당신은 '모두의연구소(모두연)' 수강생들의 비서입니다.

    현재 시간은 {today} (KST)입니다. 사용자의 '어제, 내일' 등의 표현은 {today}를 기준으로 파악하세요.
    오늘의 날짜/요일은 {today_ko} / {weekday_ko} 입니다. 날짜 및 요일 관련 질문에는 추론하지 말고 반드시 이 값을 그대로 사용하세요.

    제공된 문서 내용만을 근거로 답하세요. 근거가 없으면 '정보가 명확하지 않습니다. 운영매니저님이나 퍼실님께 문의해주세요.'라고만 대답하세요.
    사용자 입력에 포함된 사실은 근거로 사용하지 마세요.

    훈련장려금의 경우 주어진 단위 기간 일수의 80%이상을 출석해야만 금액이 지급됨을 명심하세요. 
    최대 3문장으로 짧게 답변하세요.

    {context}
    """
    
    # 답변 작성 가이드
    # LLM에게 넘겨줄 프롬프트는 답변 방식 지침과 이전 대화 문맥, 새로 들어온 질문을 모두 결합한 증강된 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = (qa_prompt | llm | StrOutputParser())
    rag_core_chain = RunnablePassthrough.assign(context=retrieve_documents) | qa_chain

    # 대화 기록 관리 결합
     # 단기 기억 상실증이 있는 AI에게 기억력을 달아주는 단계로, SESSION_ID를 통해 아 이 사람이 아까 ~질문을 물어본 그 사람이구나라고 기억하게 만듦
    return RunnableWithMessageHistory(
        rag_core_chain,
        lambda session_id: st.session_state.lc_store.setdefault(session_id, ChatMessageHistory()),
        input_messages_key="input",
        history_messages_key="chat_history"
    )


### [5] lottie animation 불러오기
# lottie animation url을 입력받아 json 불러오는 함수
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()

### [6] streamlit을 통해 배포한 웹사이트에서의 동작
def run_app():
    # 데이터 및 체인 준비
    rag_chain = get_conversational_rag_chain()

    # 세션 초기화
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "default"
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]
    if "lc_store" not in st.session_state:
        st.session_state["lc_store"] = {}
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = dt.date.today()

    with st.sidebar:
        st.title("🍀모두의 연구소")
        st.markdown("---")
        
        # key를 고정하고, value를 session_state에서 가져오도록 강제하여 streamlit이 재실행되어도 이 위젯을 새로운 것으로 착각하지 않음.
        selected_date = st.date_input(
            "원하는 날짜를 선택하세요:",
            value=st.session_state.selected_date,
            key="unique_sidebar_date_final"
        )
        
        # 선택된 값을 세션에 저장
        st.session_state.selected_date = selected_date
        st.markdown("---")
        st.info(f"오늘은: **{selected_date}** 입니다.")

        # 학습 관련 사이트
        st.markdown("---")
        st.header("🔗관련 사이트")
        st.link_button("모두의 연구소 홈페이지", "https://modulabs.co.kr")
        st.link_button("데싸 5기 노션 워크스페이스", "https://www.notion.so/New-5-25-07-07-26-01-08-New-23f2d25db62480828becc399aaa41877")
        st.link_button("데싸 5기 ZEP", "https://zep.us/play/8l5Vdo")
        st.link_button("LMS 홈페이지", "https://lms.aiffel.io/")

        # 첨부파일
        st.markdown("---")
        st.header("📄첨부파일")
        try:
            with open(r"..\data\word\휴가신청서(데싸_5기).docx", 'rb') as file:
                st.download_button(
                    label='휴가신청서 다운로드',
                    data=file,
                    file_name='휴가신청서.docx',
                    mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
        except FileNotFoundError:
            st.warning(r"첨부파일 경로를 확인하세요: ..\data\word\휴가신청서(데싸_5기).docx")

    # lottie animation 추가
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json"
    lottie_animation = load_lottie_url(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, reverse=False, loop=True, quality="high", height=500, width=800, key="animation")

    # 안내 문구 추가
    st.markdown(
        """
    <div style="text-align: center;">
        <p style="font-size:25px;">
            안녕하세요! 저는 모두봇입니다.<br>즐거운 모두연 생활을 위한 정보를 제공합니다.😊
        </p>
    </div>
    """,
        unsafe_allow_html=True
    )

    # 대화 내용 출력
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # chat loop
    # 변수에 값 할당과 동시에 할당된 값을 반환하는 (:=) 연산자 사용하여 가독성 확보
    if question := st.chat_input("질문을 입력해주세요 :)"):
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                # 날짜 정보 생성
                date_info = {
                    "input": question,
                    "today": f"{st.session_state.selected_date} 00:00:00", # 사이드바에서 고른 날짜를 today로 동적 주입 (KST 00:00:00로 고정)
                    "today_ko": st.session_state.selected_date.strftime("%Y년 %m월 %d일"),
                    "weekday_ko": ["월","화","수","목","금","토","일"][st.session_state.selected_date.weekday()] + "요일"
                }
                
                # stream 기능을 사용해 한글자씩 출력하여 사람이 읽는 것처럼 자연스럽게 출력
                stream_gen = rag_chain.stream(date_info, config={"configurable": {"session_id": st.session_state["session_id"]}})
                full_response = st.write_stream(stream_gen)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# 실행 보호 구문
if __name__ == "__main__":
    run_app()
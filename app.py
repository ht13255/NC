# app.py - BitNet Streamlit 소설 생성기 (토큰 청크 분할 & 계속 생성 지원)
import os
import subprocess
import streamlit as st
import multiprocessing
from threading import Thread

# 최대 CPU 스레드 사용
NUM_THREADS = multiprocessing.cpu_count()

st.set_page_config(page_title="BitNet Novel Generator", layout="wide")
st.title("🚀 BitNet 기반 소설 생성기")

# 초기 환경 설정: bitnet.cpp 빌드
if not os.path.isdir("bitnet.cpp"):
    with st.spinner("BitNet 환경 설정 중..."):
        subprocess.run(["bash", "setup_env.sh"], check=True)

# 사용자 입력 영역
prompt = st.text_area("소설 시작 문장 입력", height=150)
length = st.selectbox("출력 길이", ["단편(128)", "중편(512)", "장편(1024)"])
length_map = {"단편(128)": 128, "중편(512)": 512, "장편(1024)": 1024}
total_tokens = length_map[length]
# 청크 크기 설정 (사이드바)
chunk_size = st.sidebar.number_input("청크 크기(토큰)", value=128, step=64, min_value=32)

# 세션 상태 초기화
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'tokens_generated' not in st.session_state:
    st.session_state.tokens_generated = 0

# 출력 컨테이너
output = st.empty().container()

# 청크 단위 생성 함수
def run_chunk(n_tokens, input_text):
    cmd = [
        "python", "bitnet.cpp/run_inference.py",
        "-m", "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        "-p", input_text,
        "-n", str(n_tokens),
        "-t", str(NUM_THREADS),
        "-temp", "0.8"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = ""
    for line in proc.stdout:
        out += line
    proc.wait()
    return out

# "생성 시작" 버튼: 초기 청크 생성
if st.button("생성 시작"):
    st.session_state.generated_text = ""
    st.session_state.tokens_generated = 0
    n = min(chunk_size, total_tokens)
    result = run_chunk(n, prompt)
    st.session_state.generated_text += result
    st.session_state.tokens_generated += n

# "계속 생성" 버튼: 남은 토큰 생성
if st.session_state.tokens_generated < total_tokens:
    if st.button("계속 생성"):
        remaining = total_tokens - st.session_state.tokens_generated
        n = min(chunk_size, remaining)
        result = run_chunk(n, st.session_state.generated_text)
        st.session_state.generated_text += result
        st.session_state.tokens_generated += n

# 결과 표시
with output:
    st.markdown("### 생성된 소설")
    st.write(st.session_state.generated_text)

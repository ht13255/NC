# app.py - BitNet 스타일 소설 파인튜닝 & 생성 Streamlit 앱
import os
import subprocess
import torch
import multiprocessing
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# 최대 CPU 스레드 설정
NUM_THREADS = multiprocessing.cpu_count()
torch.set_num_threads(NUM_THREADS)

# Streamlit 페이지 설정
st.set_page_config(page_title="BitNet 스타일 소설 생성기", layout="wide")
st.title("📖 BitNet 스타일 소설 파인튜닝 & 생성기")

# --- 1️⃣ 소설 스타일 학습 영역 ---
st.header("1️⃣ 소설 스타일 학습")
# 파일 업로더 (메인 영역)
uploaded = st.file_uploader("학습용 소설(.txt) 업로드", type=["txt"])
# 파인튜닝 파라미터
model_name = st.text_input("기본 모델명 (Hugging Face)", value="mistralai/Mistral-7B-Instruct")
epochs = st.number_input("Epochs", min_value=1, max_value=5, value=3)
batch_size = st.number_input("Batch size", min_value=1, max_value=4, value=1)
output_dir = st.text_input("Fine-tuned 모델 저장 경로", value="./fine_tuned")
convert = st.checkbox("BitNet용 GGUF로 변환", value=True)

# 학습 및 변환 함수 정의
def fine_tune_and_convert():
    train_path = os.path.join(".", "train.txt")
    with open(train_path, "wb") as f:
        f.write(uploaded.getbuffer())
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False, device_map={"": "cpu"})
    model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=512)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=batch_size,
                              num_train_epochs=epochs, logging_steps=50, save_total_limit=1,
                              fp16=False, dataloader_num_workers=NUM_THREADS)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    st.success("✅ 학습 완료: 모델이 저장되었습니다.")
    if convert:
        gguf_path = os.path.join("models", "finetuned.gguf")
        os.makedirs(os.path.dirname(gguf_path), exist_ok=True)
        cmd = ["python", "bitnet.cpp/3rdparty/llama.cpp/convert_hf_to_gguf.py",
               "--model-dir", output_dir, "--outfile", gguf_path]
        subprocess.run(cmd, check=True)
        st.success(f"✅ GGUF 변환 완료: {gguf_path}")

# 학습 트리거 버튼
if uploaded and st.button("학습 시작"):  
    with st.spinner("모델 학습 중입니다... 잠시만 기다려주세요"):  
        fine_tune_and_convert()
        st.balloons()

# --- 2️⃣ 소설 생성 영역 ---
st.header("2️⃣ 소설 생성")
outline = st.text_area("줄거리 또는 전체 내용 요약 입력", height=120)
length = st.selectbox("생성 길이 (토큰)", [128, 512, 1024])
chunk = st.number_input("청크 크기 (토큰)", value=128, step=64, min_value=32)

# 세션 상태 초기화
if 'gen_text' not in st.session_state:
    st.session_state.gen_text = ""
if 'gen_tokens' not in st.session_state:
    st.session_state.gen_tokens = 0

# 추론 함수
def infer_chunk(prompt, n):
    gguf_path = "models/finetuned.gguf" if os.path.exists("models/finetuned.gguf") else None
    model_arg = ["-m", gguf_path] if gguf_path else ["-m", "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"]
    cmd = ["python", "bitnet.cpp/run_inference.py"] + model_arg + ["-p", prompt, "-n", str(n), "-t", str(NUM_THREADS), "-temp", "0.8"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = ""
    for line in proc.stdout:
        out += line
    proc.wait()
    return out

# 생성 시작 버튼
if st.button("생성 시작"):
    st.session_state.gen_text = outline
    st.session_state.gen_tokens = 0
    first_n = min(chunk, length)
    res = infer_chunk(st.session_state.gen_text, first_n)
    st.session_state.gen_text += res
    st.session_state.gen_tokens += first_n

# 계속 생성 버튼
if st.session_state.gen_tokens < length:
    if st.button("계속 생성"):
        remaining = length - st.session_state.gen_tokens
        next_n = min(chunk, remaining)
        res = infer_chunk(st.session_state.gen_text, next_n)
        st.session_state.gen_text += res
        st.session_state.gen_tokens += next_n

# 결과 표시
if st.session_state.gen_text:
    st.markdown("### 생성된 소설")
    st.write(st.session_state.gen_text)


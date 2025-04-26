# app.py - 사용자 업로드 소설 학습 & 스타일 재생성 Streamlit 앱
import os
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

# 최대 CPU 스레드 사용
NUM_THREADS = multiprocessing.cpu_count()
torch.set_num_threads(NUM_THREADS)

def fine_tune(model_name, train_file, output_dir, epochs, batch_size):
    # LoRA를 이용한 미세조정
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,  # CPU에서는 False
        device_map={"": "cpu"}
    )
    model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=512
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        fp16=False,
        dataloader_num_workers=NUM_THREADS
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def generate_chunk(model_dir, prompt, n_tokens, temperature=0.8):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={"": "cpu"})
    inputs = tokenizer(prompt, return_tensors='pt')
    generated = model.generate(
        **inputs,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        top_k=50
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Streamlit 레이아웃
st.set_page_config(page_title="Fine-Tune Novel Generator", layout="wide")
st.title("📚 스타일 기반 소설 생성기")

# 1) 학습 파트
st.sidebar.header("1. 소설 학습")
uploaded = st.sidebar.file_uploader("학습할 소설(.txt) 업로드", type="txt")
model_name = st.sidebar.text_input("기본 모델명", value="mistralai/Mistral-7B-Instruct")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=5, value=3)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=4, value=1)
output_dir = st.sidebar.text_input("출력 디렉토리", value="./fine-tuned-model")
if uploaded:
    train_path = os.path.join(".", "train.txt")
    with open(train_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success("업로드 완료")
    if st.sidebar.button("학습 시작"):
        with st.spinner("모델 학습 중... (시간이 오래 걸릴 수 있습니다) 🌱"):
            fine_tune(model_name, train_path, output_dir, epochs, batch_size)
        st.sidebar.success("학습 완료! 🎉")
        st.balloons()

# 2) 생성 파트
st.header("2. 소설 생성")
prompt = st.text_area("소설 시작 문장 또는 줄거리 입력", height=150)
length_opt = st.selectbox("원하는 길이", ["단편(128)", "중편(512)", "장편(1024)"])
total_tokens = int(length_opt.split("(")[1].strip(")"))
chunk_size = st.number_input("청크 크기(토큰)", value=128, step=64, min_value=32)

# 세션 상태 초기화
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'tokens_generated' not in st.session_state:
    st.session_state.tokens_generated = 0

output_box = st.empty()

# 생성 시작
if st.button("생성 시작"):
    st.session_state.generated_text = ""
    st.session_state.tokens_generated = 0
    # 사용자가 학습했다면 output_dir로, 아니면 기본 모델
    model_dir = output_dir if os.path.isdir(output_dir) else model_name
    # 첫 청크
    n = min(chunk_size, total_tokens)
    result = generate_chunk(model_dir, prompt, n)
    st.session_state.generated_text = result
    st.session_state.tokens_generated = n

# 계속 생성
if st.session_state.tokens_generated < total_tokens:
    if st.button("계속 생성"):
        model_dir = output_dir if os.path.isdir(output_dir) else model_name
        remaining = total_tokens - st.session_state.tokens_generated
        n = min(chunk_size, remaining)
        result = generate_chunk(model_dir, st.session_state.generated_text, n)
        st.session_state.generated_text += result
        st.session_state.tokens_generated += n

# 결과 표시
if st.session_state.generated_text:
    output_box.markdown("### 생성된 소설")
    output_box.write(st.session_state.generated_text)


#!/usr/bin/env bash
set -e
# 1) bitnet.cpp 클론 & 빌드
git clone --recursive https://github.com/microsoft/BitNet.git bitnet.cpp
cd bitnet.cpp

# Python 의존성 설치
pip install -r requirements.txt

# Hugging Face에서 GGUF 모델 다운로드 (모델 크기에 따라 시간이 걸릴 수 있음)
# -- 모델 디렉토리: ../models/BitNet-b1.58-2B-4T
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir ../models/BitNet-b1.58-2B-4T

# bitnet.cpp 환경 설정: i2_s (2-bit) quantization 커널 사용
python setup_env.py \
  --hf-repo microsoft/BitNet-b1.58-2B-4T-gguf \
  --model-dir ../models/BitNet-b1.58-2B-4T \
  --quant-type i2_s

cd ..
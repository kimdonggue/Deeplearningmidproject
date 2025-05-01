#!/bin/bash

# 모델 리스트
models=(efficientnet_b0 efficientnet_b1 efficientnet_b2 efficientnet_b3 efficientnet_b4 efficientnet_b5 efficientnet_b6 efficientnet_b7)

# 경로 설정
REPO_PATH="/home/dk0219/git/AUE8088-PA1"
CONFIG_FILE="$REPO_PATH/src/config.py"
TRAIN_FILE="$REPO_PATH/train.py"

for model_name in "${models[@]}"
do
    echo "======================================="
    echo "Training with model: $model_name"
    echo "======================================="

    # config.py에서 MODEL_NAME 수정
    sed -i "s/^MODEL_NAME\s*=.*/MODEL_NAME          = '$model_name'/" "$CONFIG_FILE"

    # train.py 실행
    python3 "$TRAIN_FILE"

    echo ""
done

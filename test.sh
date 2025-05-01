#!/bin/bash

# 모델과 체크포인트 리스트 (순서를 보장하기 위해 배열 사용)
model_names=(
    "resnet18"
    "resnet18"
    "alexnet"
    "Modified_AlexNet"
    "efficientnet_b0"
    "efficientnet_b1"
    "efficientnet_b2"
    "efficientnet_b3"
    "efficientnet_b4"
    "efficientnet_b5"
)

checkpoint_paths=(
    "/home/dk0219/git/AUE8088-PA1/wandb/aue8088-pa1/2we9rbbf/checkpoints/epoch=32-step=6468.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/xq0gbhb5/checkpoints/epoch=35-step=7056.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/e10uzcwr/checkpoints/epoch=39-step=7840.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/39vcqbsu/checkpoints/epoch=36-step=7252.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/3sfrrhnk/checkpoints/epoch=37-step=7448.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/na2cmfzy/checkpoints/epoch=37-step=7448.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/9snzapb1/checkpoints/epoch=37-step=7448.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/d7wr57np/checkpoints/epoch=38-step=7644.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/b8o50glr/checkpoints/epoch=37-step=7448.ckpt"
    "/home/dk0219/git/AUE8088-PA1/wandb/debug/152a9gu1/checkpoints/epoch=39-step=7840.ckpt"
)

# 경로 설정
REPO_PATH="/home/dk0219/git/AUE8088-PA1"
CONFIG_FILE="$REPO_PATH/src/config.py"
TEST_FILE="$REPO_PATH/test.py"

# 루프 수행
for i in "${!model_names[@]}"; do
    model="${model_names[$i]}"
    ckpt="${checkpoint_paths[$i]}"

    echo "======================================="
    echo "Testing model: $model"
    echo "Checkpoint: $ckpt"
    echo "======================================="

    # config.py 내 MODEL_NAME 수정
    sed -i "s/^MODEL_NAME\s*=.*/MODEL_NAME          = '$model'/" "$CONFIG_FILE"

    # test.py 실행
    python3 "$TEST_FILE" --ckpt_file "$ckpt"

    echo ""
done

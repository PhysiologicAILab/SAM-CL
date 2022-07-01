#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../


DATA_ROOT=$3
SCRATCH_ROOT=$4
ASSET_ROOT=${DATA_ROOT}

DATA_DIR="${DATA_ROOT}/pascalcontext"
SAVE_DIR="${SCRATCH_ROOT}/seg_results/pascal_context/"
BACKBONE="deepbase_resnet101_dilated8"

CONFIGS="configs/pascal_context/R_101_D_8.json"
CONFIGS_TEST="configs/pascal_context/R_101_D_8_TEST.json"

MODEL_NAME="fcnet"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_"$2
LOG_FILE="./log/pascal_context/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained_model/resnet101-imagenet.pth"
MAX_ITERS=30000


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --nbb_mult 10 \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --nbb_mult 10 \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 1 2 3 \
                       --resume_continue y \
                       --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS_TEST} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 1 2 3 \
                       --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  cd lib/metrics
  ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS_TEST} \
                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
                                   --gt_dir ${DATA_DIR}/val/label  


elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/pascal_context/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi


else
  echo "$1"x" is invalid..."
fi

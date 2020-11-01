set -e
#!/usr/bin/env bash

declare -a arr=("arc-standard")
declare -a languages=("English")

for l in "${languages[@]}"
do
  for i in "${arr[@]}"
  do
    enc=$i
    lang=$l
    DATA_DIR=./models/$enc/gold/$lang/
    MODEL_DIR=./models/$enc/models/$lang/
    OUTPUT=./models/$enc/models/$lang/
    PATH_GOLD=./gold/$lang/dev.conllu
    PATH_PREDICTED=./gold/$lang/dev.conllu
    LOG=./log/
    
    TRANSFORMER_MODEL="bert_model"
    BILSTMS=False
    BERT_BASE_MULTILINGUAL_MODEL=bert-base-multilingual-cased
    BERT_BASE_MODEL=bert-base-cased
    BERT_CHINESE=bert-base-chinese
    BERT_FINNISH=bert-base-finnish-cased-v1

    
    if [ $l = "Russian" ]; then
        SEQ_LENGTH=510
    else
        SEQ_LENGTH=400
    fi

    echo $SEQ_LENGTH

    if [ $l = "English" ]; then
        MODEL_TYPE="bert_base"
    elif [ $l = "Chinese" ]; then
        MODEL_TYPE="bert_chinese"
    elif [ $l = "Finnish" ]; then
        MODEL_TYPE="bert_finnish"
    else
        MODEL_TYPE="bert_multilingual"
        echo "BERT multilingual"
    fi

    echo $l
    echo $i
    
    DO="--do_train --do_eval --num_train_epochs 45 --train_batch_size 8 "

    if [ $BILSTMS == true ]; then
      USE_BILSTMS="--use_bilstms"
      LEARNING_RATE=1e-5 #if used with bilstms
    else
      USE_BILSTMS=""
    fi

    if [ $MODEL_TYPE  == "bert_base" ]; then
        BERT_MODEL=$BERT_BASE_MODEL
        LEARNING_RATE=1e-5
    elif [ $MODEL_TYPE == "bert_large" ]; then
        BERT_MODEL=$BERT_LARGE_MODEL
        LEARNING_RATE=1e-5
    elif [ $MODEL_TYPE == "bert_multilingual" ]; then
        BERT_MODEL=$BERT_BASE_MULTILINGUAL_MODEL
        LEARNING_RATE=1e-5

    elif [ $MODEL_TYPE == "distilbert_base" ]; then
        BERT_MODEL=$DISTILBERT_BASE_MODEL
        LEARNING_RATE=5e-5
    elif [ $MODEL_TYPE == "distilbert_multilingual" ]; then
        BERT_MODEL=$DISTILBERT_BASE_MULTILINGUAL_MODEL
        LEARNING_RATE=5e-6
    elif [ $MODEL_TYPE == "bert_chinese" ]; then
        BERT_MODEL=$BERT_CHINESE
        LEARNING_RATE=1e-5
    elif [ $MODEL_TYPE == "bert_finnish" ]; then
        BERT_MODEL=$BERT_FINNISH
        LEARNING_RATE=1e-5
    fi

    MODEL_NAME=$BERT_MODEL.proof

    python run_token_classifier.py \
    --status train \
    --data_dir $DATA_DIR \
    --transformer_model $TRANSFORMER_MODEL \
    --transformer_pretrained_model $BERT_MODEL \
    --task_name sl_tsv \
    --model_dir $MODEL_DIR/$MODEL_NAME \
    --output_dir $OUTPUT/$MODEL_NAME.output \
    --path_gold_conllu $PATH_GOLD\
    --path_predicted_conllu $PATH_GOLD \
    --label_split_char {} \
    --log $LOG/$MODEL_NAME \
    --learning_rate $LEARNING_RATE \
    --encoding $enc \
    --max_seq_length $SEQ_LENGTH $DO $USE_BILSTMS | tee -a $MODEL_DIR/training.log  
  done
done

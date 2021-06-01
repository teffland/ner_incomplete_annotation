# Run the jie experiments
mkdir -p runs

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

# Run the conll languages
for lang in eng deu esp ned
do
    echo
    echo DO $lang

    model_name="bert-base-multilingual-cased"
    if [ $lang == "eng" ]
    then
        model_name="roberta-base"
    fi 

    # Don't do full here

    # NNS
    echo "$lang NNS  runs/21-05-28_run_${lang}_r0.5_p0.9.log"
    echo "$(timestamp): start"
    nohup python adapted_main.py\
        --device "cuda:0"\
        --train-file ../data/conll2003/${lang}/entity.train-docs_r0.5_p0.9.jsonl\
        --dev-file ../data/conll2003/${lang}/entity.dev-docs.jsonl\
        --test-file ../data/conll2003/${lang}/entity.test-docs.jsonl\
        --learning_rate "2e-5"\
        --l2 0.0\
        --num_epochs 10\
        --model_folder runs/${lang}\
        --bert-model $model_name\
        --batch_size 8\
        &> runs/21-05-28_run_${lang}_r0.5_p0.9.log
    echo "$(timestamp): stop"

    # # EE
    # echo "$lang EE  runs/21-05-28_run_${lang}_P-1000.log"
    # echo "$(timestamp): start"
    # nohup python adapted_main.py\
    #     --device "cuda:0"\
    #     --train-file ../data/conll2003/${lang}/entity.train-docs_P-1000.jsonl\
    #     --dev-file ../data/conll2003/${lang}/entity.dev-docs.jsonl\
    #     --test-file ../data/conll2003/${lang}/entity.test-docs.jsonl\
    #     --learning_rate "2e-5"\
    #     --l2 0.0\
    #     --num_epochs 10\
    #     --model_folder runs/${lang}\
    #     --bert-model $model_name\
    #     &> runs/21-05-28_run_${lang}_P-1000.log
    # echo "$(timestamp): stop"
done


# Run the ontonotes languages
for lang in chinese arabic english
do
    echo
    echo DO $lang
    model_name="bert-base-multilingual-cased"
    if [ $lang == "english" ]
    then
        model_name="roberta-base"
    fi 

    # Dont do full

    # NNS
    echo "$lang NNS  runs/21-05-28_run_${lang}_r0.5_p0.9.log"
    echo "$(timestamp): start"
    nohup python adapted_main.py\
        --device "cuda:0"\
        --train-file ../data/ontonotes5/processed_docs/${lang}/train_r0.5_p0.9.jsonl\
        --dev-file ../data/ontonotes5/processed_docs/${lang}/dev.jsonl\
        --test-file ../data/ontonotes5/processed_docs/${lang}/test.jsonl\
        --learning_rate "2e-5"\
        --l2 0.0\
        --num_epochs 10\
        --model_folder runs/${lang}\
        --bert-model $model_name\
        --batch_size 8\
        &> runs/21-05-28_run_${lang}_r0.5_p0.9.log
    echo "$(timestamp): stop"

    # # EE
    # echo "$lang EE  runs/21-05-24_run_${lang}_P-1000.log"
    # echo "$(timestamp): start"
    # nohup python adapted_main.py\
    #     --device "cuda:0"\
    #     --train-file ../data/ontonotes5/processed_docs/${lang}/train_P-1000.jsonl\
    #     --dev-file ../data/ontonotes5/processed_docs/${lang}/dev.jsonl\
    #     --test-file ../data/ontonotes5/processed_docs/${lang}/test.jsonl\
    #     --learning_rate "2e-5"\
    #     --l2 0.0\
    #     --num_epochs 10\
    #     --model_folder runs/${lang}\
    #     --bert-model $model_name\
    #     &> runs/21-05-24_run_${lang}_P-1000.log
    # echo "$(timestamp): stop"
done
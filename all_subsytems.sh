subs=(all)
SEED=42
EPOCHES=200
beam_search=64
beam_search_type=tasar

for sub in "${subs[@]}"; do
    echo "Running $sub with seed=$SEED"
    RESULTS_DIRPATH='./environ_correct/advantage_learning/tasar_64'
    python ./main.py \
        --subsystem $sub \
        --device "cuda:0" \
        --seed $SEED \
        --epoches $EPOCHES \
        --results $RESULTS_DIRPATH \
        --beam_search $beam_search
done

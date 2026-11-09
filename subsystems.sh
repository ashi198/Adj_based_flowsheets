subs=(ethanol_water acetone_chloroform water_pyridine)
SEED=42
EPOCHES=100
beam_search=64

for sub in "${subs[@]}"; do
    echo "Running $sub with seed=$SEED"
    RESULTS_DIRPATH='./replay_random_new_CF_with_lit_imp_policy_restrictive_tasar_64'
    python ./main.py \
        --subsystem $sub \
        --device "cuda:1" \
        --seed $SEED \
        --epoches $EPOCHES \
        --results $RESULTS_DIRPATH \
        --beam_search $beam_search
done
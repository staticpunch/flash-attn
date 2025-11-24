# Simple benchmarking with NSight
python benchmark.py \
    --config large \
    --num-steps 20 \
    --num-warmups 10 \
    --batch-size 1 \
    --sequence-length 128 \
    --mode grad \
    --compile \
    --memory \
    --dtype fp16 / bf16 / fp32


# Annotated profiling with NSight. Following scripts are quite comprehensive,
# as they annotate forward, backward, optimizer step and also the attention operation.
MODES=(forward grad train)
CONFIGS=(small medium large)

for mode in "${MODES[@]}"; do
    for config in "${CONFIGS[@]}"; do
        echo "Running: model_config=$config, mode=$mode"
        nsys profile -o "results/nsys/gpt_${config}_${mode}" --force-overwrite true \
        python nsys_profile.py \
            --config "$config" \
            --num-steps 20 \
            --num-warmups 10 \
            --batch-size 1 \
            --sequence-length 128 \
            --annotate_attention \
            --mode "$mode"
    done
done

nsys profile -o nsys_results/gpt_large_forward --force-overwrite true \
python nsys_profile.py \
	--config large \
	--num-steps 20 \
	--num-warmups 10 \
	--batch-size 1 \
	--sequence-length 128 \
    --annotate_attention \
	--mode forward


# Benchmarking attention implementations
## naive attention
HEAD_DIMS=(16 32 64 128)
SEQ_LENGTHS=(256 1024 4096 8192 16384)
CSV_FILE="results/naive_attention.csv"

rm  "$CSV_FILE"
echo "head_dim,seq_length,mean_ms,std_ms,min_ms,max_ms" > "$CSV_FILE"

for head_dim in "${HEAD_DIMS[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        echo "Running: head_dim=$head_dim, seq_len=$seq_len"
        python benchmark_attn.py \
            --n-queries "$seq_len" \
            --n-keys "$seq_len" \
            --head-dim "$head_dim" \
            --csv "$CSV_FILE"
    done
done

HEAD_DIMS=(16 32 64 128)
SEQ_LENGTHS=(256 1024 4096 8192 16384)
CSV_FILE="results/compiled_attention.csv"

### compiled version
rm  "$CSV_FILE"
echo "head_dim,seq_length,mean_ms,std_ms,min_ms,max_ms" > "$CSV_FILE"

for head_dim in "${HEAD_DIMS[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        echo "Running: head_dim=$head_dim, seq_len=$seq_len"
        python benchmark_attn.py \
            --n-queries "$seq_len" \
            --n-keys "$seq_len" \
            --head-dim "$head_dim" \
            --csv "$CSV_FILE" \
            --compile
    done
done

# Profiling attention implementations
nsys profile -o results/attention python benchmark_attn.py \
    --n-queries 1024 \
    --n-keys 1024 \
    --head-dim 128 \
    --mode grad \
    --dtype fp32 \
    --memory \
    --compile

# Testing attention implementaion
uv run pytest tests/test_attention.py -k test_flash_forward_pass_pytorch

# ---------------------------- #
# Test attention implementations
uv run pytest tests/test_flash.py                  # run all tests
uv run python -m tests.test_flash.py               # run manual tests with logging
uv run pytest tests/test_flash.py -k False / True  # run tests with is_causal = False / True
uv run pytest tests/test_flash.py::test_name       # run a specific test

python3 alq_llama.py meta-llama/Llama-3.1-8B-Instruct c4-new \
    --new-eval \
    --wbits 4 \
    --nsamples 128 \
    --true-sequential --act-order \
    --percdamp 0.1 \
    --exponent 4 \
    --save_path llama3.1.8b.4bit.safetensors

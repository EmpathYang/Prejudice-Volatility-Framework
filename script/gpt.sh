PARENT_DIR=$(dirname $(pwd))

python "$PARENT_DIR/main.py" \
--model_class "CausalLM" \
--model_name_or_path "gpt2" \
--output_dir "$PARENT_DIR/output" \
--T "Generic" \
--X "occupation" \
--Y="race" \
--batch_size=4
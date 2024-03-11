PARENT_DIR=$(dirname $(pwd))

python "$PARENT_DIR/main.py" \
--model_class "MaskedLM" \
--model_name_or_path "google-bert/bert-base-uncased" \
--output_dir "$PARENT_DIR/output" \
--T "Gender" \
--X "occupation" \
--Y="gender" \
--batch_size=4
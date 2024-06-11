python "main.py" \
--model_class "MaskedLM" \
--model_name_or_path "google-bert/bert-base-uncased" \
--output_dir "output" \
--T "Gender" \
--X "occupation" \
--Y="gender" \
--batch_size=4
import argparse
from utils import CURRENT_DIR, DATA_DIR, TEMPLATE_DIR

def parse_args_for_context_template_collection():
    parser = argparse.ArgumentParser(description="Collect Context Templates")
    
    parser.add_argument("--nlp_model", type=str, required=False, default="en_core_web_lg", help="Spacy NLP model type")
    parser.add_argument("--data_dir", type=str, required=False, default=DATA_DIR, help="Input data directory")
    parser.add_argument("--input_file", type=str, required=False, default="articles.jsonl", help="Input file name")
    parser.add_argument("--template_dir", type=str, required=False, default=TEMPLATE_DIR, help="Output template directory")
    parser.add_argument("--template_file", type=str, required=True, help="Output file name")
    
    args = parser.parse_args()
    return args

class PVFArgumentsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Arguments pertaining to model evaluation.')
        self.add_arguments()

    def add_arguments(self):
        # Model Arguments
        self.parser.add_argument('--model_class', type=str, choices=["MaskedLM", "CausalLM"], default="MaskedLM",
                                 help='The evaluated model class.')
        self.parser.add_argument('--model_name_or_path', type=str, default=None,
                                 help='The model checkpoint for weights initialization.')
        self.parser.add_argument('--model_type', type=str, default=None,
                                 help='Initialize model weights with the default model for the given type.')
        self.parser.add_argument('--config_name', type=str, default=None,
                                 help='Pretrained config name or path if not the same as model_name')
        self.parser.add_argument('--tokenizer_name', type=str, default=None,
                                 help='Pretrained tokenizer name or path if not the same as model_name')
        self.parser.add_argument('--cache_dir', type=str, default=None,
                                 help='Where do you want to store the pretrained models downloaded from huggingface.co')
        self.parser.add_argument('--use_fast_tokenizer', action='store_true',
                                 help='Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.')
        self.parser.add_argument('--model_revision', type=str, default='main',
                                 help='The specific model version to use (can be a branch name, tag name or commit id).')
        self.parser.add_argument('--token', type=str, default=None,
                                 help='The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).')
        self.parser.add_argument('--trust_remote_code', action='store_true',
                                 help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.')
        self.parser.add_argument('--low_cpu_mem_usage', action='store_true',
                                 help='It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. Set True will benefit LLM loading time and RAM consumption.')
        self.parser.add_argument('--random_seed', type=int, default=65,
                                 help='Random seed for initializing model.')
        

        # Data Arguments
        self.parser.add_argument('--T', type=str, default="Gender",
                                 help='Select a set of template.')
        self.parser.add_argument('--X', type=str, default="occupation",
                                 help='A social division which will augment the templates and serve as the evidence.')
        self.parser.add_argument('--Y', type=str, default="gender",
                                 help='A social division towards which we will evaluate the model\' attitude.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Set the batch size for loading the templates.')

        # Other Arguments
        self.parser.add_argument('--output_dir', type=str, required=True,
                                 help='The output directory where the model predictions and checkpoints will be written.')

    def parse_args(self):
        model_args = self.parser.parse_args()
        if model_args.model_name_or_path == None and model_args.model_type == None:
            raise ValueError("You should assign a value to model_name_or_path or model_type.")
        if (not model_args.model_name_or_path == None) and (not model_args.model_type == None):
            raise ValueError("You can only assign either of model_name_or_path or model_type to avoid conflicts.")
        return model_args
import torch
from transformers import (
    set_seed,
    AutoConfig,
    CONFIG_MAPPING,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os

from arguments import PVFArgumentsParser
from dataset import DATASET_MAPPING
from compute_risk import compute_risk

AUTO_MODEL_MAPPING = {
    "MaskedLM": AutoModelForMaskedLM,
    "CausalLM": AutoModelForCausalLM,
}

def main():

    parser = PVFArgumentsParser()
    args = parser.parse_args()
    
    set_seed(args.random_seed)

    # Load model and tokenizer.
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()

    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AUTO_MODEL_MAPPING[args.model_class].from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        model = AUTO_MODEL_MAPPING[args.model_class].from_config(config, trust_remote_code=args.trust_remote_code)

    # TODO: Load Ts, Xs, and Ys, where you have to realize the Dataset class (in dataset.py under the same directory) with the following functions:
    #   1. Basic Dataset class functions: __init__() and __getitem__();
    #   2. get_class_indices() that find the model-specific Ys' indices for aggregating the results later;
    #   3. data_collator() that takes batch features and outputs tokenized_features, X_idx, T_idx, with the later 2 intended for locating the computed probabilities;
    #   4. compute_probability() that compute the batched probabilties for each Y class.
    dataset = DATASET_MAPPING[model.config.model_type](args=args, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.data_collator)

    # Compute the P_matrix.
    P_matrix = np.zeros(dataset.get_P_matrix_shape())
    with torch.no_grad():
        for batch in dataloader:
            tokenized_features, X_idx, T_idx = batch
            outputs = model(**tokenized_features)
            logits = outputs.logits
            probs = dataset.compute_probability(logits=logits)
            P_matrix[X_idx, T_idx] = probs
    P_matrix = P_matrix / np.sum(P_matrix, axis=-1, keepdims=True)

    # Compute risks and save them.
    r_X, r_X_prejudice, r_X_volatility, R, R_prejudice, R_volatility = compute_risk(P_matrix=P_matrix, T_distribution=dataset.get_T_dist(), X_distribution=dataset.get_X_dist())
    risk_dict = {
        "r_X": r_X,
        "r_X_prejudice": r_X_prejudice,
        "r_X_volatility": r_X_volatility,
        "R": R,
        "R_prejudice": R_prejudice,
        "R_volatility": R_volatility
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir+f'/[M]{model.config.model_type}[T]{args.T}[X]{args.X}[Y]{args.Y}.pkl'), 'wb') as wf:
        pickle.dump(risk_dict, wf)

if __name__ == "__main__":
    main()
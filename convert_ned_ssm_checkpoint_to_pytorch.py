"""This script can be used to convert checkpoints provided in the `mamba2_ssm` library into the format provided in HuggingFace `transformers` for the NED model. It depends on the `mamba2_ssm` package to be installed."""

import argparse
import json
from functools import partial
from os import path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_model

from transformers import GPTNeoXTokenizerFast, LlamaTokenizerFast
from modeling_ned import NedConfig, NedForCausalLM

def load_state_dict_from_safetensors(ned_checkpoint_path: str, ckpt_name: str) -> dict[str, torch.Tensor]:
    # Load weights and config from paths
    original_state_dict = {}
    with safe_open(path.join(ned_checkpoint_path, ckpt_name), framework="pt") as f:
        for k in f.keys():
            newk = k.removeprefix("model.")
            original_state_dict[newk] = f.get_tensor(k).clone()
    return original_state_dict


def load_state_dict_from_torch(ned_checkpoint_path: str, ckpt_name: str) -> dict[str, torch.Tensor]:
    return torch.load(path.join(ned_checkpoint_path, ckpt_name), map_location="cpu", weights_only=True)


def convert_ssm_config_to_hf_config(config_ssm: dict, ned_model_dict: dict) -> NedConfig:
    hf_config = NedConfig()

    hf_config.hidden_size = config_ssm[ned_model_dict["hidden_size"]]
    hf_config.num_hidden_layers = config_ssm[ned_model_dict["num_hidden_layers"]]
    hf_config.state_size = config_ssm.get("state_size", 128)
    hf_config.intermediate_size = hf_config.hidden_size * 4  # Default expansion
    hf_config.vocab_size = config_ssm["vocab_size"]
    hf_config.tie_word_embeddings = config_ssm.get("tie_embeddings", False)
    hf_config.bos_token_id = 0
    hf_config.pad_token_id = 0
    hf_config.eos_token_id = 0

    return hf_config


def load_and_save_tokenizer(
    ned_model_type: str,
    output_dir: str,
    tokenizer_model_path: Optional[str] = None,
) -> None:
    tokenizer = None

    # Load tokenizer
    if ned_model_type == "mamba_ssm": # Only mamba_ssm conversion is supported for NED
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("state-spaces/mamba-130m-hf", padding_side="left")

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


_NED_MODELS_DICT = {
    "mamba_ssm": {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layer",
        "n_groups": "ngroups",
        "bos_token_id": 0,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "config_name": "config.json",
        "load_state_dict": partial(load_state_dict_from_torch, ckpt_name="pytorch_model.bin"),
        "load_and_save_tokenizer": partial(load_and_save_tokenizer, "mamba_ssm"),
    },
}


def convert_ned_checkpoint_file_to_huggingface_model_file(
    ned_checkpoint_path: str,
    ned_model_type: str,
    precision: str,
    output_dir: str,
    tokenizer_model_path: Optional[str] = None,
) -> None:
    ned_model_dict = _NED_MODELS_DICT[ned_model_type]

    # Load and save config based on name
    config_path = path.join(ned_checkpoint_path, ned_model_dict["config_name"])
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)
    hf_config = convert_ssm_config_to_hf_config(config_ssm=config, ned_model_dict=ned_model_dict)
    hf_config.save_pretrained(output_dir)

    # Load state dict of the original model and transfer to hf model
    original_state_dict = ned_model_dict["load_state_dict"](ned_checkpoint_path=ned_checkpoint_path)
    hf_model = NedForCausalLM(hf_config)
    hf_model.load_state_dict(original_state_dict)

    # Save new model to pytorch_dump_path
    dtype = torch.float32 if precision == "fp32" else (torch.bfloat16 if precision == "bf16" else torch.float16)
    save_model(hf_model.to(dtype), path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})

    # Load and save tokenizer
    ned_model_dict["load_and_save_tokenizer"](output_dir=output_dir, tokenizer_model_path=tokenizer_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ned_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` or `.safetensors` mamba2_ssm checkpoint file to be converted for NED.",
    )
    parser.add_argument(
        "-m",
        "--ned_model_type",
        type=str,
        default="mamba_ssm",
        const="mamba_ssm",
        required=True,
        choices=("mamba_ssm",), # Only mamba_ssm conversion is supported for NED
        help="The model type the conversion will be performed on. Currently only `mamba_ssm` is supported for NED.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp16",
        const="fp16",
        required=True,
        choices=("fp32", "fp16", "bf16"),
        help="The precision the model will be saved in. Select from fp32, fp16 or bf16.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output NED model to."
    )
    parser.add_argument(
        "-t",
        "--tokenizer_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to a tokenizer file for NED model.",
    )
    args = parser.parse_args()

    convert_ned_checkpoint_file_to_huggingface_model_file(
        args.ned_checkpoint_directory,
        args.ned_model_type,
        args.precision,
        args.output_dir,
        args.tokenizer_model_path,
    )

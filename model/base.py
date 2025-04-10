import os
import json
from pathlib import Path
from typing import Optional, Union, Dict
import torch
from dataclasses import dataclass

from safetensors.torch import save_file
from huggingface_hub import snapshot_download, HfApi
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from model.conf import ModelConfig


def _get_model_config_by_hf_config(hf_config: dict, dataclass_type: dataclass) -> dict:
    mapping = {
        "hidden_size": "n_embed",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "n_mlp",
        "rms_norm_eps": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "tie_word_embeddings": "tie_word_embeddings",
    }
    config = {
        mapping.get(key, key): value for key, value in hf_config.items()
    }
    llm_config =  {k: v for k, v in config.items() if k in dataclass_type.__annotations__}
    return ModelConfig(**llm_config)

class BaseModel:
    @classmethod
    def from_pretrained(
        cls,
        repo_id: Union[str, Path],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        **kwargs,
    ):
        if os.path.isdir(repo_id):
            model_path = Path(repo_id)
        else:
            model_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            )

        with open(model_path / "config.json", "r") as f:
            config_data = json.load(f)

        model_config: ModelConfig = _get_model_config_by_hf_config(config_data, ModelConfig)
        with init_empty_weights():
            model = cls(model_config)
        model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map=device_map,
            dtype=torch.bfloat16,
            no_split_module_classes=["Block"],
        )

        return model

    def save_pretrained(
            self,
            save_directory: Union[str, Path],
            config: Optional[ModelConfig] = None,
            push_to_hub: bool = False,
            repo_id: Optional[str] = None,
            token: Optional[str] = None,
    ):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self._save_pretrained(save_directory)

        config = config or getattr(self, "config", None)
        if config is not None:
            (save_directory / "config.json").write_text(
                json.dumps(config.__dict__, indent=2)
            )

        if push_to_hub and repo_id:
            HfApi(token=token).upload_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    def _save_pretrained(self, save_directory: Path):
        state_dict = self.state_dict()
        split_size = 3 * 1024 * 1024 * 1024  # 3GB
        tensors, current_size, file_index = {}, 0, 1

        def write_chunk():
            nonlocal tensors, current_size, file_index
            save_path = save_directory / f"model-{file_index:05d}-of-xxxx.safetensors"
            save_file(tensors, save_path)
            file_index += 1
            return {}, 0

        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > split_size:
                tensors, current_size = write_chunk()
            tensors[key] = tensor
            current_size += tensor_size

        if tensors:
            tensors, current_size = write_chunk()

        total_files = file_index - 1
        for idx in range(1, total_files + 1):
            old_name = save_directory / f"model-{idx:05d}-of-xxxx.safetensors"
            new_name = save_directory / f"model-{idx:05d}-of-{total_files:05d}.safetensors"
            old_name.rename(new_name)
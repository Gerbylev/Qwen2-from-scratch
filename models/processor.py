import torch
from typing import List, Union,Optional
from typing import Callable
from transformers import AutoTokenizer

from models.model import Qwen2

def wrap_messages(messages: List[dict]) -> str:
    conversation = ""
    for message in messages:
        conversation += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return conversation


class Processor:
    def __init__(self, repo_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = Qwen2.from_pretrained(repo_id)

    def __call__(
        self,
        inputs: str,
        device: Optional[Union[str, torch.device]] = None,
        custom_callback: Callable[[int], bool] = None,
        use_cache: bool = True,
        max_new_tokens: int = 200
    ) -> str:
        prompt = wrap_messages(inputs)
        device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = self.encode(prompt, device)
        return self.generate(input_ids, custom_callback, use_cache, max_new_tokens)

    def generate(self, input_ids, custom_callback: Callable[[int],bool] = None, use_cache: bool = True, max_new_tokens: int = 200):
        def callback(token):
            return token == 151643 or token == 151645
        inpt_len = input_ids.size(1) + 3
        output = self.model.generate(input_ids, callback if custom_callback is None else custom_callback, max_new_tokens = max_new_tokens, use_cache=use_cache)
        return self.decode(output[: , inpt_len:])

    def decode(self, inputs_ids):
        res = []
        for ids in inputs_ids:
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            res.append(self.tokenizer.decode(ids, skip_special_tokens=False))
        return res

    def encode(self, inputs: List[str],
        device: Optional[Union[str, torch.device]] = None):
        input_ids = []
        input_ids.extend(self.tokenizer.encode(inputs))
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        return input_ids

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Расскажи про Олега Гербылева?"}
    ]
    processor = Processor("Qwen/Qwen2-0.5B-Instruct")
    result = processor(messages, use_cache=False)
    print(f"Запрос: {messages[0]["content"]}")
    print(f"Ответ модели: {result[0]}")


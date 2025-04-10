import torch
from typing import List, Union,Optional
from typing import Callable
from transformers import AutoTokenizer

from model.model import Qwen2


def wrap_messages(messages: List[dict]) -> str:
    return ''.join(
        f"<|im_start|>{message['content']}<|im_end|>\n"
        for message in messages
    )


class Processor:
    def __init__(self, repo_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = Qwen2.from_pretrained(repo_id)

    def __call__(
            self,
            inputs: List[dict],
            device: Optional[Union[str, torch.device]] = None,
            custom_callback: Optional[Callable[[int], bool]] = None,
            use_cache: bool = True,
            max_new_tokens: int = 200
    ) -> str:
        prompt = wrap_messages(inputs)
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_ids = self.encode(prompt, device)
        with torch.no_grad():
            return self.generate(input_ids, custom_callback, use_cache, max_new_tokens)

    def generate(
            self,
            input_ids: torch.LongTensor,
            custom_callback: Optional[Callable[[int], bool]] = None,
            use_cache: bool = True,
            max_new_tokens: int = 200
    ) -> str:
        stop_callback = custom_callback or (lambda token: token in {151643, 151645})
        generated_tokens = []
        current_input_ids = input_ids
        past_cache = None

        for _ in range(max_new_tokens):
            logits, _, past_cache = self.model.forward(current_input_ids, use_cache=use_cache, past_cache=past_cache)
            last_token_logits = logits[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            if stop_callback(token_id):
                break
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def decode(self, input_ids: torch.LongTensor) -> str:
        return self.tokenizer.decode(input_ids.squeeze(0).tolist(), skip_special_tokens=False)

    def encode(self, inputs: str, device: Union[str, torch.device]) -> torch.LongTensor:
        input_ids = self.tokenizer.encode(inputs, return_tensors='pt')
        return input_ids.to(device)

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Расскажи про Олега Гербылева?"}
    ]
    processor = Processor("Qwen/Qwen2-0.5B-Instruct")
    result = processor(messages, use_cache=True, max_new_tokens=500)
    print(f"Запрос: {messages[0]["content"]}")
    print(f"Ответ модели: {result}")


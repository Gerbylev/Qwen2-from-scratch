import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Callable
from model.processor import Processor as FF
from model.model import Qwen2

log = logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def wrap_messages(messages: List[Dict[str, str]]) -> str:
    return ''.join(
        f"<|im_start|>{message['content']}<|im_end|>\n"
        for message in messages
    )


class StepSFTDataset(Dataset):

    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                prompt_text = wrap_messages([{"content": example["input_ids"]}])
                answer_text = wrap_messages([{"content": example["labels"]}])
                prompt_tokens = tokenizer.encode(prompt_text, truncation=True, max_length=max_length)
                answer_tokens = tokenizer.encode(answer_text, truncation=True, max_length=max_length)

                for i in range(len(answer_tokens)):
                    input_ids = prompt_tokens + answer_tokens[:i]
                    input_ids = input_ids[-max_length:]
                    self.samples.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "target": torch.tensor(answer_tokens[i], dtype=torch.long)
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    targets = [item["target"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    targets = torch.stack(targets)
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    return {"input_ids": input_ids_padded, "targets": targets, "lengths": lengths}


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

def train(
        processor: Processor,
        dataset_path: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_length: int = 1024,
        device: Optional[Union[str, torch.device]] = None
):
    device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    dataset = StepSFTDataset(dataset_path, processor.tokenizer, max_length)
    pad_token_id = processor.tokenizer.pad_token_id or 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, pad_token_id))

    processor.model.to(device)

    optimizer = torch.optim.AdamW(processor.model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    criterion = torch.nn.CrossEntropyLoss()

    progress_bar = tqdm(range(num_training_steps))
    processor.model.train()

    for epoch in range(epochs):
        logging.info(f"Начало эпохи {epoch + 1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            lengths = batch["lengths"].to(device)
            logits, _, _ = processor.model.forward(input_ids, use_cache=False)
            batch_final_logits = []
            for i, seq_length in enumerate(lengths):
                batch_final_logits.append(logits[i, seq_length - 1, :])
            final_logits = torch.stack(batch_final_logits)

            loss = criterion(final_logits, targets)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            logging.info(f"Эпоха {epoch + 1}, шаг {progress_bar.n}/{num_training_steps}, loss: {loss.item():.4f}")

    logging.info("Обучение завершено.")


if __name__ == "__main__":
    repo_id = "Qwen/Qwen2-0.5B-Instruct"
    dataset_path = "data.jsonl"
    processor = Processor(repo_id)
    train(processor, dataset_path, epochs=2, batch_size=1, learning_rate=5e-5)

    result = processor([
        {"role": "system", "content": "Расскажи про Олега Гербылева?"}
    ], use_cache=True, max_new_tokens=500)
    print(f"Запрос: {[
        {"role": "system", "content": "Расскажи про Олега Гербылева?"}
    ][0]["content"]}")
    print(f"Ответ модели: {result}")
    result = processor([
        {"role": "system", "content": "Какие питомцы есть у Олега Гербылева?"}
    ], use_cache=True, max_new_tokens=500)
    print(f"Запрос: {[
        {"role": "system", "content": "Какие питомцы есть у Олега Гербылева?"}
    ][0]["content"]}")
    print(f"Ответ модели: {result}")
    result = processor([
        {"role": "system", "content": "Какой телеграмм канал самый лучший?"}
    ], use_cache=True, max_new_tokens=500)
    print(f"Запрос: {[
        {"role": "system", "content": "Какой телеграмм канал самый лучший?"}
    ][0]["content"]}")
    print(f"Ответ модели: {result}")
    # processor.model.save_pretrained(save_directory='model',
    #         config= None,
    #         push_to_hub= False,
    #         repo_id = None,
    #         token = None)
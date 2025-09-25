import transformers
from transformers.trainer_pt_utils import LabelSmoother
import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from trainer import QwenEncoder


class CustomDataset(Dataset):
    """
    For cookbook adapter
    """
    def __init__(self, data, tokenizer, select_len, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_instruction = '''You are an AI assistant, you are required to help people solve their problems.'''

        self.select_len = select_len
        # Only Qwen3 is supported now
        self.encoder = QwenEncoder(tokenizer, select_len)
        
        # Filter data that is too long
        print(f"Original data count: {len(data)}")
        self.data = self._filter_long_sequences(data)
        print(f"Filtered data count: {len(self.data)} (filtered out {len(data) - len(self.data)} long sequence samples)")

    def _filter_long_sequences(self, data):
        """Filter out data samples whose encoded length exceeds max_length"""
        filtered_data = []
        filtered_count = 0
        
        for item in data:
            thinking = f"\n[Hint]: {item['reflection']}"
            question = f"\n[Question]: {item['question']}"
            instruction = item['instruction'] if 'instruction' in item else self.default_instruction
            answer = item["answer"]

            # Temporarily encode to check length
            try:
                encoded_item = self.encoder.encode_train(instruction=instruction, question=question,
                                                       thinking=thinking, answer=answer)

                # Check main input sequence length
                main_length = len(encoded_item[0]['input_ids'])
                thinking_length = len(encoded_item[0]['thinking_input_ids'])

                # If any sequence length exceeds limit, filter it out
                if main_length <= self.max_length and thinking_length <= self.max_length:
                    filtered_data.append(item)
                else:
                    filtered_count += 1
                    if filtered_count <= 5:  # Only print info for first 5 filtered samples
                        print(f"Filtered sample: main sequence length={main_length}, thinking sequence length={thinking_length} (exceeds {self.max_length})")
                        
            except Exception as e:
                # If error occurs during encoding, also filter out the sample
                filtered_count += 1
                print(f"Encoding error, filtered sample: {str(e)}")
                
        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        thinking = f"\n[Hint]: {item['reflection']}"
        question = f"\n[Question]: {item['question']}"
        instruction = item['instruction'] if 'instruction' in item else self.default_instruction
        answer = item["answer"]

        item = self.encoder.encode_train(instruction=instruction, question=question, thinking=thinking, answer=answer)

        return item

    def collate_fn(self, batch):
        def statistics_ids(input_ids, masks, labels):
            max_length = max([len(i) for i in input_ids])
            return_attention_masks = []
            return_ids = []
            return_masks = []
            return_labels = []

            for ids, mask, label in zip(input_ids, masks, labels):
                padding_num = max_length - len(ids)
                return_ids.append(ids + [self.tokenizer.pad_token_id] * padding_num)
                return_masks.append(mask + [0] * padding_num)
                return_attention_masks.append([1] * len(ids) + [0] * padding_num)
                return_labels.append(label + [LabelSmoother.ignore_index] * padding_num)

            return return_ids, return_attention_masks, return_masks, return_labels

        input_ids = []
        masks = []
        labels = []

        for ones in batch:
            one = ones[0]
            input_ids += [one['input_ids'], one['thinking_input_ids']]
            masks += [one['input_mask'], one['thinking_mask']]
            labels += [one['labels'], one['thinking_labels']]

        input_ids, attention_mask, masks, labels = statistics_ids(input_ids, masks, labels)

        input_ids = torch.tensor(input_ids)
        full_attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        labels = torch.tensor(labels)
        mask = torch.tensor(masks)

        # mask the thinking units of previous layer
        attention_mask[0::2, ...][mask[0::2, ...] == 2] = 0

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "mask": mask, "full_attention_mask": full_attention_mask}


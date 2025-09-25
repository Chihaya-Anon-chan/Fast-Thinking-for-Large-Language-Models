import transformers
from transformers.trainer_pt_utils import LabelSmoother
import torch

class Encoder():
    def __init__(self, tokenizer, select_len):
        self.tokenizer = tokenizer
        self.select_len = select_len

    
    def encode_train(self, instruction, question, thinking, answer):
        """
        Encode for training phase
        """
        pass

    def encode_cot_train(self, instruction, question, cot_thinking, answer):
        """
        Encode for CoT training phase
        """
        pass

    def encode_inference(self, instruction, question):
        """
        Encode for inference phase
        """
        pass


# LlamaEncoder removed - only Qwen3 is supported


class QwenEncoder(Encoder):
    def __init__(self, tokenizer, select_len):
        super().__init__(tokenizer, select_len)

    def encode_inference(self, instruction, question):
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START, add_special_tokens=False).input_ids
        im_end = self.tokenizer(IM_END, add_special_tokens=False).input_ids
        nl_tokens = self.tokenizer(NL, add_special_tokens=False).input_ids
        _system = self.tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
        _user = self.tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
        _assistant = self.tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens
        pad_tokens = [self.tokenizer.pad_token_id]

        input_ids = []

        # mask
        input_mask = []
        attention_mask = []
        full_attention_mask = []

        # add instruction
        input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens     

        input_mask += [0] * len(input_ids)
        attention_mask += [1] * len(input_ids)
        full_attention_mask += [1] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += im_start + _user + question_enc

        input_mask += [0] * len(im_start + _user) + [1] * len(question_enc)
        attention_mask += [1] * len(im_start + _user + question_enc)
        full_attention_mask += [1] * len(im_start + _user + question_enc)

        # add thinking
        input_ids += self.select_len * pad_tokens + im_end + nl_tokens

        input_mask += [2] * self.select_len + [0] * len(im_end + nl_tokens)
        attention_mask += [0] * self.select_len + [1] * len(im_end + nl_tokens)
        full_attention_mask += [1] * (self.select_len + len(im_end + nl_tokens))

        # add ASSISTANT
        input_ids += im_start + _assistant

        input_mask += [0] * len(im_start + _assistant)
        attention_mask += [1] * len(im_start + _assistant)
        full_attention_mask += [1] * len(im_start + _assistant)

        assert len(input_ids) == len(input_mask) == len(attention_mask) == len(full_attention_mask)

        return [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'attention_mask': attention_mask,
            'full_attention_mask': full_attention_mask
        }]


    def encode_train(self, instruction, question, thinking, answer,):
        IGNORE_TOKEN_ID = [LabelSmoother.ignore_index]
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START, add_special_tokens=False).input_ids
        im_end = self.tokenizer(IM_END, add_special_tokens=False).input_ids
        nl_tokens = self.tokenizer(NL, add_special_tokens=False).input_ids
        _system = self.tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
        _user = self.tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
        _assistant = self.tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens
        pad_tokens = [self.tokenizer.pad_token_id]

        input_ids = []
        thinking_input_ids = []

        labels = []
        thinking_labels = []

        # input & thinking mask
        input_mask = []
        thinking_mask = []

        # add instruction
        input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens
        thinking_input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens

        labels += IGNORE_TOKEN_ID * len(input_ids)
        thinking_labels += IGNORE_TOKEN_ID * len(thinking_input_ids)

        input_mask += [0] * len(input_ids)
        thinking_mask += [0] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += im_start + _user + question_enc
        thinking_input_ids += im_start + _user + question_enc

        labels += IGNORE_TOKEN_ID * len(im_start + _user + question_enc)
        thinking_labels += IGNORE_TOKEN_ID * len(im_start + _user + question_enc)

        input_mask += [0] * len(im_start + _user) + [1] * len(question_enc)
        thinking_mask += [0] * len(im_start + _user + question_enc)

        # add thinking
        thinking_enc = self.tokenizer(thinking, add_special_tokens=False).input_ids

        input_ids += self.select_len * pad_tokens + im_end + nl_tokens
        thinking_input_ids += thinking_enc + im_end + nl_tokens

        labels += IGNORE_TOKEN_ID * (self.select_len + len(im_end + nl_tokens))
        thinking_labels += IGNORE_TOKEN_ID * len(thinking_enc) + im_end + nl_tokens

        # 2 in question mask denote for thinking, 1 in question mask denote for question
        input_mask += [2] * self.select_len + [0] * len(im_end + nl_tokens)
        thinking_mask += [1] * len(thinking_enc) + [0] * len(im_end + nl_tokens)

        # add answer
        answer_enc = self.tokenizer(answer, add_special_tokens=False).input_ids

        input_ids += im_start + _assistant + answer_enc + im_end + nl_tokens
        labels += len(im_start + _assistant) * IGNORE_TOKEN_ID + answer_enc + im_end + nl_tokens

        input_mask += [0] * len(im_start + _assistant + answer_enc + im_end + nl_tokens)

        # TODO adding answer into thinking 

        assert len(input_ids) == len(labels) == len(input_mask)
        assert len(thinking_input_ids) == len(thinking_labels) == len(thinking_mask)

        return [{
            'input_ids': input_ids,
            'thinking_input_ids': thinking_input_ids,
            'labels': labels,
            'thinking_labels': thinking_labels,
            'input_mask': input_mask,
            'thinking_mask': thinking_mask
        }]

    def encode_cot_train(self, instruction, question, cot_thinking, answer):
        """
        Encode for CoT training phase - similar to encode_train but for CoT thinking
        """
        IGNORE_TOKEN_ID = [LabelSmoother.ignore_index]
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START, add_special_tokens=False).input_ids
        im_end = self.tokenizer(IM_END, add_special_tokens=False).input_ids
        nl_tokens = self.tokenizer(NL, add_special_tokens=False).input_ids
        _system = self.tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
        _user = self.tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
        _assistant = self.tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens
        pad_tokens = [self.tokenizer.pad_token_id]

        input_ids = []
        cot_input_ids = []

        labels = []
        cot_labels = []

        # input & cot mask
        input_mask = []
        cot_mask = []

        # add instruction
        input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens   
        cot_input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens

        labels += IGNORE_TOKEN_ID * len(input_ids)
        cot_labels += IGNORE_TOKEN_ID * len(cot_input_ids)

        input_mask += [0] * len(input_ids)
        cot_mask += [0] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += im_start + _user + question_enc
        cot_input_ids += im_start + _user + question_enc

        labels += IGNORE_TOKEN_ID * len(im_start + _user + question_enc)
        cot_labels += IGNORE_TOKEN_ID * len(im_start + _user + question_enc)

        input_mask += [0] * len(im_start + _user) + [1] * len(question_enc)
        cot_mask += [0] * len(im_start + _user + question_enc)

        # add CoT thinking
        cot_enc = self.tokenizer(cot_thinking, add_special_tokens=False).input_ids

        input_ids += self.select_len * pad_tokens + im_end + nl_tokens
        cot_input_ids += cot_enc + im_end + nl_tokens

        labels += IGNORE_TOKEN_ID * (self.select_len + len(im_end + nl_tokens))
        cot_labels += IGNORE_TOKEN_ID * len(cot_enc) + im_end + nl_tokens

        # 2 in question mask denote for CoT thinking, 1 in question mask denote for question
        input_mask += [2] * self.select_len + [0] * len(im_end + nl_tokens)
        cot_mask += [1] * len(cot_enc) + [0] * len(im_end + nl_tokens)

        # add answer
        answer_enc = self.tokenizer(answer, add_special_tokens=False).input_ids

        input_ids += im_start + _assistant + answer_enc + im_end + nl_tokens
        labels += len(im_start + _assistant) * IGNORE_TOKEN_ID + answer_enc + im_end + nl_tokens

        input_mask += [0] * len(im_start + _assistant + answer_enc + im_end + nl_tokens)

        assert len(input_ids) == len(labels) == len(input_mask) 
        assert len(cot_input_ids) == len(cot_labels) == len(cot_mask)

        return [{
            'input_ids': input_ids,
            'cot_input_ids': cot_input_ids,
            'labels': labels,
            'cot_labels': cot_labels,
            'input_mask': input_mask,
            'cot_mask': cot_mask
        }]


if __name__ == '__main__':

    # Example usage with Qwen3
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
 
    encoder = QwenEncoder(tokenizer, 16)
    
    encoder.encode_train(
        instruction="You are an AI assistant",
        question="\n1+1=?",
        thinking="\nFor this question, you need to first.",
        answer="The answer is 2."
    )


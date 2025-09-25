from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.utils import PeftType
import os
import json

@dataclass
class CodebookConfig(PeftConfig):
    codebook_size: int = field(default=None, metadata={"help": "Size of codebook"})
    inserted_layer: int = field(default=None, metadata={"help": "Position of inserted layer"})
    select_len: int = field(default=None, metadata={"help": "Length of soft prompt to insert"})

    def __post_init__(self):
        self.peft_type = PeftType.ADAPTION_PROMPT

    @classmethod
    def from_pretrained(cls, path_name: str):
        config_file = os.path.join(path_name, 'adapter_config.json')
        with open(config_file, 'r') as f_c:
            data = json.load(f_c)

        return cls(
            codebook_size=data['codebook_size'],
            inserted_layer=data['inserted_layer'],
            select_len=data['select_len'],
        )

    @property
    def is_adaption_prompt(self) -> bool:
        return True
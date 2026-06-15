from typing import Dict, Optional, List, Union
from transformers import AutoTokenizer

from src.exceptions import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)

class TokenizerWrapper:
    """Wraps a HuggingFace tokenizer to provide a consistent interface for the RIFD project."""
    def __init__(self,pretrained_name:str="bert-base-uncased", max_length:int=512):
        logger.info(f"Loading tokenizer: '{pretrained_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.vocab_size = self.tokenizer.vocab_size

        #ensure pad token exists
        if self.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token":"[PAD]"})
            self.pad_token_id = self.tokenizer.pad_token_id

        
    def encode(self, text:str, max_length:Optional[int]=None, padding:bool=False, truncation:bool=True, 
               add_special_tokens:bool=True)->Dict[str,List[int]]:
        """
        Tokenize a single piece of text and return input_ids and attention_mask.
        """
        if max_length is None:
            max_length = self.max_length
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=None,
            return_attention_mask=True,
            return_token_type_ids=False
        )
    def decode(self, ids:List[int], skip_special_tokens:bool=True)->str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self)->int:
        return self.vocab_size

def load_tokenizer(config:Dict)->TokenizerWrapper:
    """Factory: create a TokenizerWrapper from the project config."""
    tk_cfg = config.get("tokenizer", {})
    pretrained_name = tk_cfg.get("pretrained_name", "bert-base-uncased")
    max_length = tk_cfg.get("max_length", 512)
    return TokenizerWrapper(pretrained_name=pretrained_name, max_length=max_length)
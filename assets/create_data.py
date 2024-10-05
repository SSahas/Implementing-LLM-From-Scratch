from regex import Regex
from datasets import load_dataset

import regex
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece

from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from transformers import AutoTokenizer
from tokenizers import normalizers, pre_tokenizers, processors, decoders

import datasets 

from datasets import load_dataset

import numpy as np

import re


def deduplicate(ds):
    
    unique_text = set()

    def dedup_func(x):
        """Use this function to remove duplicate entries"""
        if x['text'] in unique_text:
            return False
        else:
            unique_text.add(x['text'])
            return True


    ds = ds.filter(dedup_func, load_from_cache_file=False, num_proc=1)
    unique_text.clear()

    print("remove duplicate", ds)
    
    print("dedupplucatio complete")
    return ds





def clean_text(text):
      text = text.replace("@-@", "")
      text = text.replace("@.@", ".")
      text = text.replace("@,@", ",")
      text = text.replace("", "[UNK]")
      text = text.replace("\n", "")
      text = text.replace("\'", "")
      text = text.replace("\\", "")
      text = text.replace(" '", "'")



      return text










def Tokenizer_train(ds):

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(r"[\p{Other}&&[^\n\t\r]]", ""),
            normalizers.Replace(r"[\s]", " "),
            #normalizers.Lowercase(),
            normalizers.NFD(), normalizers.StripAccents()]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["[UNK]", "<|BOS|>", "<|EOS|>"]
    trainer = WordPieceTrainer(vocab_size=40000,  special_tokens=special_tokens)


    for split in ds.keys():
        tokenizer.train_from_iterator(ds[split]['text'], trainer=trainer)


    eos_token_id = tokenizer.token_to_id("<|EOS|>")

    tokenizer.post_processor = processors.TemplateProcessing(

        single="<|EOS|> $A <|EOS|>",

        special_tokens=[ ("<|EOS|>", eos_token_id)],

        )

    #tokenizer.save("tokenizer.json")

    return tokenizer


def tokenization(ds, tokenizer):

    #tokenizer = Tokenizer.from_file("tokenizer.json")

    def tokenization_func(example):



        tokens = tokenizer.encode(example["text"])

        token_ids = tokens.ids

        example["input_ids"] = token_ids

        example["num_tokens"] = len(token_ids)

        return example
    
    ds = ds.map(tokenization_func, load_from_cache_file=False)

    return ds 

def create_datafile(ds):

    train_input_ids = np.concatenate(ds['train']["input_ids"])
    print(len(train_input_ids))


    #input_ids_list = train_input_ids.tolist()
    packaged_pretrain_dataset = datasets.Dataset.from_dict(
        {"input_ids": train_input_ids}
    )
    print(packaged_pretrain_dataset)


    packaged_pretrain_dataset.to_parquet("pretraining_data.parquet")





def main():
    
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    ds = ds.filter(lambda example: len(example['text']) > 0)
    ds = deduplicate(ds)
    ds = ds.map(lambda example: {'text': clean_text(example['text'])})
    print("clean text complete")
    tokenizer = Tokenizer_train(ds)
    print("tokenizer training complete")
    ds = tokenization(ds, tokenizer=tokenizer)
    print("tokenization complete")
    create_datafile(ds)
    print("data file created")
    tokenizer.save("tokenizer.json")
    print("tokenizer saved")
    
    


if __name__ == "__main__":
    main()









    

     










    
# Implementing Decoder only Model (GPT style) from scratch with PyTorch
- Pretraining a LLM model for Text generation, used Salesforce/wikitext for training. The model was trained for 30000 iterations with a batch size of 8 for ~2.5 hours on Tesla P100 (Kaggle Free gpu support). The training loss is around 3.5. Used adam optimizer with a learning rate of 5e-4. After training, the model is producing little reasonable english, can be trained for more time with bigger n_embd and block size for better generation.

## Model Details
```
n_embd = 512
vocab_size = 28144
n_layers = 6
n_heads = 8
block_size = 512 # number to previous tokens to attend to perform attention
batch_size = 8
learning rate = 5e-4
```

## Load Dataset for training

- To create data file and the tokenizer for traning the model, run [Tokenizer_Training notebook](https://github.com/SSahas/Implementing-LLM-From-Scratch/blob/main/Tokenizer_Training.ipynb). Added a special token "<|EOS|>" as bos and eos token.Used [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) for pretrainig the model and training the tokenizer.

### Load datafile  from huggingface 

```
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SSahas/llm_tokenizer")


ds = load_dataset("SSahas/llm_pretrain_dataset")                # load data file from huggingface
print(dataset)
data = torch.tensor(dataset['train']['input_ids'], dtype = torch.long, device = device)
```

### Tokenizer Training
```
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace(r"[\p{Other}&&[^\n\t\r]]", ""),
        normalizers.Replace(r"[\s]", " "),
        #normalizers.Lowercase(),
        normalizers.NFD(), normalizers.StripAccents()]
)

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

special_tokens = ["[UNK]", "<|EOS|>"]
trainer = WordPieceTrainer(vocab_size=40000,  special_tokens=special_tokens)


for split in ds.keys():
    tokenizer.train_from_iterator(ds[split]['text'], trainer=trainer)

eos_token_id = tokenizer.token_to_id("<|EOS|>")

tokenizer.post_processor = processors.TemplateProcessing(

    single="<|EOS|> $A <|EOS|>",

    special_tokens=[ ("<|EOS|>", eos_token_id)],

)
```

## Model Traning

- Used a basic for loop for training the model. After creating the data file and tokenizer run [Model_Training.ipynb](https://github.com/SSahas/Implementing-LLM-From-Scratch/blob/main/Tokenizer_Training.ipynb) for trainig the LLM model.
```
for iter in range(max_iters):

    xb, yb = get_batch()

    logits, loss = model(batch = xb, targets = yb)
    

    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```


# Loss Curve 
![cross_entropy_loss_curve](https://github.com/user-attachments/assets/70396741-6fab-4ca0-96b6-a1e32ca49826)

# References 
- [Andrej karpathy-nanoGPT](https://github.com/karpathy/nanoGPT)
- [t5-pytorch](https://github.com/conceptofmind/t5-pytorch)
- [nonoT5](https://github.com/PiotrNawrot/nanoT5)


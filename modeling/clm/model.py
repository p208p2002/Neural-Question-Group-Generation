import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
args = get_args()

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids,attention_mask,labels=None):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        loss = outputs['loss']
        self.log('dev_loss',loss)
    
    def test_step(self, batch, batch_idx):
        input_ids = batch[0]
        input_ids_len = input_ids.shape[-1]
        sample_outputs = self.model.generate(
            input_ids = input_ids,
            do_sample=True, 
            max_length=1024,
            top_k=20, 
            top_p=0.85, 
            num_return_sequences=1,
            eos_token_id=self.tokenizer.pad_token_id
        )

        print("Output:\n" + 100 * '-')
        for i, sample_output in enumerate(sample_outputs):
            decode_questions = self.tokenizer.decode(sample_output[input_ids_len:], skip_special_tokens=False)
            decode_questions = re.sub(re.escape(self.tokenizer.pad_token),'',decode_questions).split(self.tokenizer.sep_token)
            for j,q in enumerate(decode_questions):
                print(i,j,q)
            print()
            
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)

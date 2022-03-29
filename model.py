from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.nn as nn

class GPT2FewshotClassifier(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        # self.gpt2 = GPT2LMHeadModel.from_pretrained(dir_path, from_tf=True)
        # self.gpt2 = GPT2LMHeadModel.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name_or_path)      

    def forward(self, x):
        # import pdb; pdb.set_trace()
        outputs = self.gpt2(x)[0][:, -1, :]
        
        return outputs
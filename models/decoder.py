
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from enum import Enum
# from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from typing import Tuple, Optional, Union
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Please install the 'peft' library to use this module.")

def get_decoder(name: str):
    if name == "Decoder":
        return DecoderModel
    else:
        raise Exception('The decoder model {} is incorrect or not supported'.format(name))
    
def downsample(x):
    if x.shape[1] == 32:
        return x
    clip_latent = x[:,0,:].unsqueeze(1)
    pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
    x = torch.concat((clip_latent,pooled),axis=1)
    return x

class Downsampler(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        # self.fc = nn.Linear(din, dout)

    def forward(self, x):
        # x = self.fc(x)
        clip_latent = x[:,0,:].unsqueeze(1)
        # downsample the timesteps by 4. Downsample only the frame-level info
        pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
        # add clip-level latent back to audio
        x = torch.concat((clip_latent,pooled),axis=1)
        return x

class DecoderModel(nn.Module):
    def __init__(self, text_decoder: str, prefix_length: int, freeze_decoder_weights: bool = True,):
        super(DecoderModel, self).__init__()
        self.prefix_length = prefix_length
        self.text_decoder = text_decoder.lower()
        # self.gpt = GPT2LMHeadModel.from_pretrained(text_decoder)
        self.lm = AutoModelForCausalLM.from_pretrained(text_decoder)
        if "gpt2" in self.text_decoder:
            self.lm_embedding_size = self.lm.transformer.wte.weight.shape[1]
        elif "smollm2" in self.text_decoder:
            self.lm_embedding_size = self.lm.model.embed_tokens.weight.shape[1]
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")

        # Configure and apply LoRA
        self.lora = False
        if self.lora:
            lora_config = LoraConfig(
                # r=8,                # Rank of LoRA
                # lora_alpha=16,      # Scaling factor
                r=256,
                lora_alpha=256,
                target_modules=["q_proj", "v_proj"],  # Specify modules to adapt
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.lm = get_peft_model(self.lm, lora_config)
            self.lm.print_trainable_parameters()  # Debug: Verify trainable parameters
        
        if freeze_decoder_weights:
            for p in self.lm.parameters():
                p.requires_grad = False

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    def generate_prefix_inference(self, daudio1, daudio2, texts_enc):
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()

        # separate token between two audios'
        if "gpt" in self.text_decoder:
            dtext = self.lm.transformer.wte(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            sep_token = torch.tensor([50256]).to(dtext.device)
            sep_embed = self.lm.transformer.wte(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        elif "smollm2" in self.text_decoder:
            if self.lora:
                dtext = self.lm.base_model.model.model.embed_tokens(texts_enc["input_ids"])
            else:
                dtext = self.lm.model.embed_tokens(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            sep_token = torch.tensor([0]).to(dtext.device)
            if self.lora:
                sep_embed = self.lm.base_model.model.model.embed_tokens(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
            else:
                sep_embed = self.lm.model.embed_tokens(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        
        prefix = torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)
        return prefix

    def forward(self, daudio1: torch.Tensor, daudio2: torch.Tensor, texts_enc: torch.Tensor, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        if "gpt2" in self.text_decoder:
            # input prompt
            dtext = self.lm.transformer.wte(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            # output labels
            embedding_text = self.lm.transformer.wte(tokens['input_ids'])
            # separator
            sep_token = torch.tensor([50256]).to(dtext.device)
            sep_embed = self.lm.transformer.wte(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        elif "smollm2" in self.text_decoder:
            # input prompt
            if self.lora:
                dtext = self.lm.base_model.model.model.embed_tokens(texts_enc["input_ids"])
            else:
                dtext = self.lm.model.embed_tokens(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            # output labels
            if self.lora:
                embedding_text = self.lm.base_model.model.model.embed_tokens(tokens["input_ids"])
            else:
                embedding_text = self.lm.model.embed_tokens(tokens['input_ids'])
            # separator
            sep_token = torch.tensor([0]).to(dtext.device)
            if self.lora:
                sep_embed = self.lm.base_model.model.model.embed_tokens(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
            else:
                sep_embed = self.lm.model.embed_tokens(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()
        
        prefix = torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens['input_ids'].shape[0], tokens['input_ids'].device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.lm(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out
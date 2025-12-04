import sys
sys.path.append('')
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
import os

from models.audio import get_audio_encoder
from models.decoder import get_decoder

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        """Initialize a Batchnorm layer. """
        m.bias.data.fill_(0.)
        m.weight.data.fill_(1.)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, 
                 audioenc_name:str, 
                 d_in: int, d_out: int, 
                 use_pretrained_audioencoder: bool, 
                 freeze_audio_encoder_weights: bool,
                 pretrained_audioencoder_path: str) -> None:
        super().__init__()

        audio_encoder, pretrained_emb_size = get_audio_encoder(audioenc_name)

        if use_pretrained_audioencoder:
            d_in = pretrained_emb_size

        self.base = audio_encoder()
        
        if use_pretrained_audioencoder:
            if audioenc_name == 'HTSAT':
                # Load pretrained weights for HTSAT
                pretrained_model_path = os.path.join(pretrained_audioencoder_path, 'HTSAT_AudioSet_Saved_1.ckpt')
                ckpt = torch.load(pretrained_model_path, map_location="cpu")
                new_ckpt = {}
                for key in ckpt["state_dict"]:
                    new_ckpt[key[10:]] = ckpt['state_dict'][key]
                self.base.htsat.load_state_dict(new_ckpt)
            elif audioenc_name == 'Cnn14':
                # Load pretrained weights for Cnn14
                pretrained_model_path = os.path.join(pretrained_audioencoder_path, 'Cnn14_mAP=0.431.pth')
                ckpt = torch.load(pretrained_model_path, map_location="cpu")["model"]
                self.base.cnn14.load_state_dict(ckpt)
            else:
                Exception('Add loading audio encoder weights code for {}'.format(audioenc_name))

        self.projection = Projection(pretrained_emb_size if use_pretrained_audioencoder else d_in, d_out)

        if freeze_audio_encoder_weights:
            if audioenc_name == 'HTSAT':
                for p in self.base.htsat.parameters():
                    p.requires_grad = False
            elif audioenc_name == 'Cnn14':
                for p in self.base.cnn14.parameters():
                    p.requires_grad = False

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output, out_dict

class Mellow(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                d_in: int,
                # text decoder
                text_decoder: str,
                prefix_length: int,
                freeze_text_decoder_weights: bool,
                # common
                d_out: int,
                use_pretrained_audioencoder: bool,
                freeze_audio_encoder_weights: bool,
                pretrained_audioencoder_path: str = None,
                ):
        super().__init__()        
        self.audio_encoder = AudioEncoder(
            audioenc_name, d_in, d_out,
            use_pretrained_audioencoder, freeze_audio_encoder_weights,
            pretrained_audioencoder_path)

        self.caption_decoder = get_decoder('Decoder')(
            text_decoder, prefix_length, freeze_text_decoder_weights,
        )

    def forward(self, input_dict):
        audio1 = input_dict['audio1']
        audio2 = input_dict['audio2']
        texts_enc = input_dict['input']
        texts_dec = input_dict['answer']

        audio_embed1, _, _ = self.audio_encoder(audio1)
        audio_embed2, _, _ = self.audio_encoder(audio2)
        out = self.caption_decoder(audio_embed1, audio_embed2, texts_enc, texts_dec)
        return out
    
    def generate_prefix_inference(self, input_dict):
        audio1 = input_dict['audio1']
        audio2 = input_dict['audio2']
        texts_enc = input_dict['input']

        audio_embed1, _, od1 = self.audio_encoder(audio1)
        audio_embed2, _, od2 = self.audio_encoder(audio2)
        prefix = self.caption_decoder.generate_prefix_inference(audio_embed1, audio_embed2, texts_enc)
        return prefix, od1, od2
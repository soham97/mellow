import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import os
import random
from tqdm import tqdm
from scipy.signal import fftconvolve
import numpy as np
from transformers import AutoTokenizer
from data.template import DETAIL, WORD, BOTH, FIRST, SECOND, LONGLINE, BOTH_SPK, FIRST_SPK, SECOND_SPK
from data.template import DETAILONLY, WORDONLY, LONGLINEONLY

class AudioTextEvalDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        data_path="",
        datafiles=[''],
        sampling_rate=44100, 
        max_clip_len=10,
        tokenizer_type="gpt2",
        ip_text_len=40,
        op_text_len=300,
    ):
        self.collect_data_jsons(data_path, datafiles)
        self.ip_text_len = ip_text_len
        self.op_text_len = op_text_len

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate
        self.tokenizer_type = tokenizer_type
        self.tokenizer = self._create_tokenizer(tokenizer_type)
    
    def _create_tokenizer(self, tokenizer_type):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        # if 'gpt' in tokenizer_type:
        tokenizer.add_special_tokens({'pad_token': '!'})
        return tokenizer

    def _tokenize_text(self, sentence, max_len):
        # if 'gpt' in self.tokenizer_type:
        sentence = sentence + ' <|endoftext|>'
        return self.tokenizer.encode_plus(text=sentence, add_special_tokens=True, truncation=True,
                                          max_length=max_len, pad_to_max_length=True, return_tensors="pt")

    def collect_data_jsons(self, data_path, datafiles):
        # collect per class data json
        all_data_json = []
        self.data_path = data_path
        for i, d in enumerate(datafiles):
            with open(d, 'r') as fp:
                data_json = json.load(fp)
                # Only print from rank 0 to avoid duplicate messages
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
                if local_rank == 0:
                    print(f"Dataset: {d.split(os.path.sep)[-1]}, \t Examples: {str(len(data_json))}")
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json
        self.exists_filepaths1 = list(set([self.all_data_json[i]["filepath1"] for i in range(len(self.all_data_json)) if self.all_data_json[i]["filepath1"] != '']))

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        if waveform.shape[0] > 1:
            waveform = (waveform[0] + waveform[1]) / 2
            waveform = waveform.unsqueeze(0)
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _read_audio(self, index):
        file_path1, file_path2 = self.all_data_json[index]["filepath1"], self.all_data_json[index]["filepath2"]
        if self.all_data_json[index]["filepath2"] == '':
            file_path2 = random.choice(self.exists_filepaths1)
        
        file_path1 = os.path.join(self.data_path, file_path1)
        file_path1 = file_path1.replace("/",os.path.sep).replace("\\",os.path.sep)
        file_path2 = os.path.join(self.data_path, file_path2)
        file_path2 = file_path2.replace("/",os.path.sep).replace("\\",os.path.sep)
        try:
            audio_data1, audio_rate1 = torchaudio.load(file_path1, channels_first=True)
            audio_data2, audio_rate2 = torchaudio.load(file_path2, channels_first=True)

            return audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {file_path1} or {file_path2}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)
        
    def _create_answer_input(self, index):
        input = self.all_data_json[index]['input']
        if input == "explain the difference in few words":
            input = "Explain the difference between the two audios in few words."
            answer = self.all_data_json[index]['answer']
        elif input == "explain the difference in a sentence":
            input = "Explain the difference between the two audios in one extended sentence."
            answer = self.all_data_json[index]['answer']
        elif input == "explain the difference in detail":
            input = "Explain the difference between the two audios in detail."
            answer = self.all_data_json[index]['answer']
        elif input == "caption first audio":
            input = "caption the audio"
            answer = self.all_data_json[index]['caption1']
        else:
            input = self.all_data_json[index]["input"]
            answer = self.all_data_json[index]['answer']
            
        return answer.lower(), input.lower()
    
    def _read_text(self, index):
        answer, input = self._create_answer_input(index)

        tok_answer = self._tokenize_text(answer, self.op_text_len)
        tok_input = self._tokenize_text(input, self.ip_text_len)
        return tok_answer, tok_input, answer, input

    def __getitem__(self, index):
        # create a audio tensor  
        audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2 = self._read_audio(index)
        tok_answer, tok_input, answer_text, input_text = self._read_text(index)
        
        # resample audio clip
        try:
            if audio_rate1 != self.sampling_rate:
                audio_data1 = torchaudio.functional.resample(audio_data1, orig_freq=audio_rate1, new_freq=self.sampling_rate)
            if audio_rate2 != self.sampling_rate:
                audio_data2 = torchaudio.functional.resample(audio_data2, orig_freq=audio_rate2, new_freq=self.sampling_rate)
        except Exception as e:
            print(f'Error resampling: {e} occurs, when loading {file_path1} or {file_path2}')
        
        # audio_data1 = audio_data1.unsqueeze(0)
        # audio_data2 = audio_data2.unsqueeze(0)
        
        audio_data1 = self._cut_or_randomcrop(audio_data1)
        audio_data2 = self._cut_or_randomcrop(audio_data2)

        data_dict = {
            'waveform1': audio_data1,
            'waveform2': audio_data2,
            'answer': tok_answer,
            'answer_text': answer_text,
            'input_text': input_text,
            'input': tok_input,
            'file_path1': file_path1,
            'file_path2': file_path2,
        }

        return data_dict
    

def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {
                'text': 'a sound of dog',
                'waveform': (1, samples),
                'modality': 'audio_text'
            }
            ...
            ]
    Returns:
        data_dict: e.g. 
            'audio_text': {
                'text': ['a sound of dog', ...]
                'waveform': (batch_size, 1, samples)
        }
    """
    at_data_dict = {}
    
    if len(list_data_dict) > 0:
        for key in list_data_dict[0].keys():
            at_data_dict[key] = [at_data_dict[key] for at_data_dict in list_data_dict]
            if key == 'waveform1' or key == "waveform2":
                at_data_dict[key] = torch.stack(at_data_dict[key]).squeeze(1)
            elif key == 'answer' or key == "input":
                stack = {k:[] for k in at_data_dict[key][0].keys()}
                for entry in at_data_dict[key]:
                    for k in entry.keys():
                        stack[k].append(entry[k])
                at_data_dict[key] = {k:torch.stack(stack[k]).squeeze(1) for k in stack.keys()}                       
            elif key == 'file_path1' or key == "file_path2" or key == "answer_text" or key == "input_text":
                at_data_dict[key] = [text for text in at_data_dict[key]]
    
    return at_data_dict

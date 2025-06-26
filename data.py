import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import torchaudio
import torch
import pandas as pd
import string
from tqdm import tqdm
import logging


phone_dict = ['a', 'aɪ', 'aʊ', 'b', 'd', 'dʒ', 'e', 'ee', 'eɪ', 'eː', 'f', 'h', 'i', 'iː', 'j', 
              'k', 'l', 'm', 'n', 'o', 'oʊ', 'oː', 'p', 'r', 's', 'ss', 't', 'ts', 'tʃ', 'tː', 
              'u', 'uː', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑː', 'ɑːɹ', 'ɑ̃', 
              'ɔ', 'ɔː', 'ɔ̃', 'ə', 'ɚ', 'ɛ', 'ɛ̃', 'ɜ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʁ', 
              'ʃ', 'ʊ', 'ʊɹ', 'ʌ', 'ʒ', 'ʝ', 'β', 'θ', ' '] # 70 + unk = 71 in total 72, epsilon id is 71
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}
# space: 69
# unknown: 70
# epsilon: 71

class LyricsAlignDataset(Dataset):
    def __init__(self, dataset, partition, sr, input_sample, sepa_dir):
        '''
        :param dataset:       a list of song dictionaries with vocal_path and annotation
        :param sr:            sampling rate
        :param input_sample:  input and output length in samples
        :param sepa_dir:      separated files directory
        '''
        super(LyricsAlignDataset, self).__init__()

        self.dataset = dataset[partition]
        self.sr = sr
        self.input_sample = input_sample
        self.hop = input_sample // 2

        # Precompute number of samples available per song
        self.song_lengths = []
        for song in tqdm(self.dataset):
            song["vocal_path"] = song["vocal_path"].replace("/import/c4dm-datasets/sepa_DALI/audio_umx_16000/", sepa_dir)
            info = torchaudio.info(song["vocal_path"])
            total_length = np.int(np.floor(info.num_frames/info.sample_rate*self.sr))
            segments = ((total_length - input_sample) // self.hop) + 1
            self.song_lengths.append(segments)

        self.start_pos = SortedList(np.cumsum(self.song_lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):
        while True:
            song_idx = self.start_pos.bisect_right(index)
            if song_idx > 0:
                index = index - self.start_pos[song_idx - 1]

            song = self.dataset[song_idx]
            start_sample = index * self.hop
            end_sample = start_sample + self.input_sample

            # Load audio slice and pad if necessary
            waveform, sr = torchaudio.load(song["vocal_path"])
            start_sec, end_sec = start_sample / self.sr, end_sample / self.sr
            start_sample, end_sample = int(start_sec * sr), int(end_sec * sr)
            waveform = waveform[:, start_sample:end_sample]
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            # waveform = waveform[:, start_sample:end_sample] #
            if waveform.shape[-1] < self.input_sample:
                waveform = torch.nn.functional.pad(waveform, (0, self.input_sample - waveform.shape[-1]))
            if waveform.shape[-1] > self.input_sample:
                waveform = waveform[:, :self.input_sample]

            # Find target words within window
            words_idx = [i for i, w in enumerate(song["words"]) if w["time"][1] > start_sec and w["time"][0] < end_sec]
            words = [song["words"][i] for i in words_idx]
            if not words:
                index = np.random.randint(self.length)
                continue

            targets = " ".join([w["text"] for w in words]).strip()
            if len(targets) > 120:
                index = np.random.randint(self.length)
                continue

            # Convert to phoneme sequence
            phoneme_lists = song["phonemizer"]
            phonemes = [phoneme_lists[i] for i in words_idx if i < len(phoneme_lists)]
            phone_seq = self.phone2seq(self.convert_phone_list(phonemes))
            
            # Extract melody (pitch and time) for notes inside window
            notes = song.get("notes", [])
            pitch_times = [(n["pitch"], n["time"]) for n in notes if n["time"][1] > start_sec and n["time"][0] < end_sec]
            if pitch_times:
                pitches, times = zip(*pitch_times)
                notes_data = (np.array(pitches).reshape(-1, 1) - 38, 
                              np.array(times) - start_sec)
            else:
                notes_data = (np.empty((0, 1), dtype=np.short), np.empty((0, 2)))
             
            if len(notes_data[0]) > 0 and (np.min(notes_data[0]) < 0 or np.max(notes_data[0]) > (83 - 38)):
                index = np.random.randint(self.length)
                continue

            seq = self.text2seq(targets)

            break

        return waveform[0], targets, seq, phone_seq, notes_data

    def text2seq(self, text):
        seq = []
        for c in text.lower():
            idx = string.ascii_lowercase.find(c)
            if idx == -1:
                if c == "'":
                    idx = 26
                elif c == " ":
                    idx = 27
                else:
                    continue # remove unknown characters
            seq.append(idx)
        return np.array(seq)

    def phone2seq(self, text):
        seq = []
        for c in text:
            if c in phone2int:
                idx = phone2int[c]
            else:
                idx = 70 # unknown phoneme
            seq.append(idx)
        return np.array(seq)

    def convert_phone_list(self, phonemes):
        ret = []
        for l in phonemes:
            l_decode = [' '] + l
            ret += l_decode
        if len(ret) > 1:
            return ret[1:]
        else:
            return []
    
    def seq2phone(self, seq):
        ret = []
        for idx in seq:
            if idx < len(phone_dict):
                ret.append(phone_dict[idx])
            else:
                ret.append('unk')
        return ret

    def __len__(self):
        return self.length

class JamendoLyricsDataset(Dataset):
    def __init__(self, sr, audio_dir, annot_dir, ext=".wav"):
        super(JamendoLyricsDataset, self).__init__()

        self.sr = sr
        self.annot_dir = annot_dir
        self.ext = ext
        self.audio_list = [file.replace(".csv", "") for file in os.listdir(self.annot_dir) if file.endswith('.csv')]
        self.audio_dir = audio_dir

    def __getitem__(self, index):

        file = self.audio_list[index]
        audio_path = os.path.join(self.audio_dir, file + self.ext)
        
        # load audio
        waveform, sr = torchaudio.load(audio_path)

        # the bdr model was trained on 22050 Hz
        waveform_bdr = torchaudio.functional.resample(waveform, sr, self.sr)

        # the ac model was trained on 16000 Hz -> resampled to 22050 Hz
        # to simulate the same condition, we resample to 16000 Hz and then 22050 Hz
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = torchaudio.functional.resample(waveform, 16000, self.sr)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            waveform_bdr = waveform_bdr.mean(dim=0, keepdim=True)
        audio_length = waveform.shape[-1]
        if audio_length < waveform_bdr.shape[-1]:
            waveform_bdr = waveform_bdr[:, :audio_length]
        elif audio_length > waveform_bdr.shape[-1]:
            waveform_bdr = torch.nn.functional.pad(waveform_bdr, (0, audio_length - waveform_bdr.shape[-1]))
        
        # read timestamps and lyrics
        timestamps = pd.read_csv(os.path.join(self.annot_dir, file + ".csv"))
        timestamps["phone_idx"] = timestamps["phone_idx"].apply(lambda x: eval(x) if pd.notna(x) else None)
        timestamps["line_idx"] = timestamps["line_idx"].apply(lambda x: eval(x) if pd.notna(x) else None)
        word_idx = timestamps["phone_idx"].to_list()
        line_idx = timestamps["line_idx"].dropna().to_list()
        lyrics = "; ;".join(timestamps["phonemizer"].to_list()).split(";")

        return [waveform[0]], (word_idx, line_idx), (lyrics, file, audio_length), [waveform_bdr[0]]
        

    def __len__(self):
        return len(self.audio_list)

import os
import soundfile
import torch
import numpy as np
import librosa
import string
import warnings

phone_dict = ['a', 'aɪ', 'aʊ', 'b', 'd', 'dʒ', 'e', 'ee', 'eɪ', 'eː', 'f', 'h', 'i', 'iː', 'j', 
              'k', 'l', 'm', 'n', 'o', 'oʊ', 'oː', 'p', 'r', 's', 'ss', 't', 'ts', 'tʃ', 'tː', 
              'u', 'uː', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑː', 'ɑːɹ', 'ɑ̃', 
              'ɔ', 'ɔː', 'ɔ̃', 'ə', 'ɚ', 'ɛ', 'ɛ̃', 'ɜ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʁ', 
              'ʃ', 'ʊ', 'ʊɹ', 'ʌ', 'ʒ', 'ʝ', 'β', 'θ', ' '] # 70 + unk = 71 in total 72, epsilon id is 71
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}

def my_collate(batch):
    audio, targets, seqs, audio_bdr = zip(*batch)
    audio = np.array(audio[0][0])
    audio_bdr = np.array(audio_bdr[0][0])
    print(audio.shape, audio_bdr.shape)
    targets = list(targets)
    seqs = list(seqs)
    return audio, targets, seqs, audio_bdr

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")

def gen_phone_gt(words, raw_lines):

    # helper function
    def getsubidx(x, y):  # find y in x
        l1, l2 = len(x), len(y)
        for i in range(l1 - l2 + 1):
            if x[i:i + l2] == y:
                return i
    words_p = []
    lyrics_p = []
    for word in words:
        out = g2p(word)
        out = [phone if phone[-1] not in string.digits else phone[:-1] for phone in out]
        words_p.append(out)
        if len(lyrics_p) > 0:
            lyrics_p.append(' ')
        lyrics_p += out

    len_words_p = [len(phones) for phones in words_p]
    idx_in_full_p = []
    s1 = 0
    s2 = s1
    for l in len_words_p:
        s2 = s1 + l
        idx_in_full_p.append([s1, s2])
        s1 = s2 + 1

        # beginning of a line
        idx_line_p = []
        last_end = 0
        for i in range(len(raw_lines)):
            line = []
            line_phone = [g2p(word) for word in raw_lines[i].split()]
            for l in line_phone:
                line += l + [' ']
            line = line[:-1]
            line = [phone if phone[-1] not in string.digits else phone[:-1] for phone in line]
            offset = getsubidx(lyrics_p[last_end:], line)
            assert (offset >= 0)
            assert (line == lyrics_p[last_end + offset:last_end + offset + len(line)])
            idx_line_p.append([last_end + offset, last_end + offset + len(line)])
            last_end += offset + len(line)

    return lyrics_p, words_p, idx_in_full_p, idx_line_p

class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,
    }, path)

def load_model(model, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        state = {"step": 0,
                 "worse_epochs": 0,
                 "epochs": checkpoint['epoch'],
                 "best_loss": np.Inf}

    return state


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def alignment(song_pred, lyrics, idx):
    audio_length, num_class = song_pred.shape
    lyrics_int = phone2seq(lyrics)
    lyrics_length = len(lyrics_int)
    # print(lyrics_int, lyrics, lyrics_length)

    s = np.zeros((audio_length, 2*lyrics_length+1)) - np.Inf
    opt = np.zeros((audio_length, 2*lyrics_length+1))

    blank = 71

    # init
    s[0][0] = song_pred[0][blank]
    # insert eps
    for i in np.arange(1, audio_length):
        s[i][0] = s[i-1][0] + song_pred[i][blank]

    for j in np.arange(lyrics_length):
        if j == 0:
            s[j+1][2*j+1] = s[j][2*j] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 1  # 45 degree
        else:
            s[j+1][2*j+1] = s[j][2*j-1] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 2 # 28 degree

        s[j+2][2*j+2] = s[j+1][2*j+1] + song_pred[j+2][blank]
        opt[j+2][2*j+2] = 1  # 45 degree


    for audio_pos in np.arange(2, audio_length):

        for ch_pos in np.arange(1, 2*lyrics_length+1):

            if ch_pos % 2 == 1 and (ch_pos+1)/2 >= audio_pos:
                break
            if ch_pos % 2 == 0 and ch_pos/2 + 1 >= audio_pos:
                break

            if ch_pos % 2 == 1: # ch
                ch_idx = int((ch_pos-1)/2)
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # last ch -> ch
                b = s[audio_pos-1][ch_pos-2] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][lyrics_int[ch_idx]]
                if a > b and a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                elif b >= a and b >= c:
                    s[audio_pos][ch_pos] = b
                    opt[audio_pos][ch_pos] = 2
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

            if ch_pos % 2 == 0: # eps
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][blank]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][blank]
                if a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

    score = s[audio_length-1][2*lyrics_length]

    # retrive optimal path
    path = []
    x = audio_length-1
    y = 2*lyrics_length
    path.append([x, y])
    while x > 0 or y > 0:
        if opt[x][y] == 1:
            x -= 1
            y -= 1
        elif opt[x][y] == 2:
            x -= 1
            y -= 2
        else:
            x -= 1
        path.append([x, y])

    path = list(reversed(path))
    word_align = []
    path_i = 0

    word_i = 0
    while word_i < len(idx):
        # e.g. "happy day"
        # find the first time "h" appears
        if path[path_i][1] == 2*idx[word_i][0]+1:
            st = path[path_i][0]
            # find the first time " " appears after "h"
            while  path_i < len(path)-1 and (path[path_i][1] != 2*idx[word_i][1]+1):
                path_i += 1
            ed = path[path_i][0]
            # append
            word_align.append([st, ed])
            # move to next word
            word_i += 1
        else:
            # move to next audio frame
            path_i += 1

    return word_align, score

def alignment_bdr(song_pred, lyrics, idx, bdr_pred, line_start):
    audio_length, num_class = song_pred.shape
    lyrics_int = phone2seq(lyrics)
    lyrics_length = len(lyrics_int)

    s = np.zeros((audio_length, 2*lyrics_length+1)) - np.Inf
    opt = np.zeros((audio_length, 2*lyrics_length+1))

    blank = 71

    # init
    s[0][0] = song_pred[0][blank]
    # insert eps
    for i in np.arange(1, audio_length):
        s[i][0] = s[i-1][0] + song_pred[i][blank]

    for j in np.arange(lyrics_length):
        if j == 0:
            s[j+1][2*j+1] = s[j][2*j] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 1  # 45 degree
        else:
            s[j+1][2*j+1] = s[j][2*j-1] + song_pred[j+1][lyrics_int[j]]
            opt[j+1][2*j+1] = 2 # 28 degree
        if j in line_start:
            s[j + 1][2 * j + 1] += bdr_pred[j+1]

        s[j+2][2*j+2] = s[j+1][2*j+1] + song_pred[j+2][blank]
        opt[j+2][2*j+2] = 1  # 45 degree

    for audio_pos in np.arange(2, audio_length):

        for ch_pos in np.arange(1, 2*lyrics_length+1):

            if ch_pos % 2 == 1 and (ch_pos+1)/2 >= audio_pos:
                break
            if ch_pos % 2 == 0 and ch_pos/2 + 1 >= audio_pos:
                break

            if ch_pos % 2 == 1: # ch
                ch_idx = int((ch_pos-1)/2)
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # last ch -> ch
                b = s[audio_pos-1][ch_pos-2] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][lyrics_int[ch_idx]]
                if a > b and a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                elif b >= a and b >= c:
                    s[audio_pos][ch_pos] = b
                    opt[audio_pos][ch_pos] = 2
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

                if ch_idx in line_start:
                    s[audio_pos][ch_pos] += bdr_pred[audio_pos]

            if ch_pos % 2 == 0: # eps
                # cur ch -> ch
                a = s[audio_pos-1][ch_pos] + song_pred[audio_pos][blank]
                # eps -> ch
                c = s[audio_pos-1][ch_pos-1] + song_pred[audio_pos][blank]
                if a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

    score = s[audio_length-1][2*lyrics_length]

    # retrive optimal path
    path = []
    x = audio_length-1
    y = 2*lyrics_length
    path.append([x, y])
    while x > 0 or y > 0:
        if opt[x][y] == 1:
            x -= 1
            y -= 1
        elif opt[x][y] == 2:
            x -= 1
            y -= 2
        else:
            x -= 1
        path.append([x, y])

    path = list(reversed(path))
    word_align = []
    path_i = 0

    word_i = 0
    while word_i < len(idx):
        # e.g. "happy day"
        # find the first time "h" appears
        if path[path_i][1] == 2*idx[word_i][0]+1:
            st = path[path_i][0]
            # find the first time " " appears after "h"
            while  path_i < len(path)-1 and (path[path_i][1] != 2*idx[word_i][1]+1):
                path_i += 1
            ed = path[path_i][0]
            # append
            word_align.append([st, ed])
            # move to next word
            word_i += 1
        else:
            # move to next audio frame
            path_i += 1

    return word_align, score

def phone2seq(text):
    seq = []
    for c in text:
        if c in phone_dict:
            idx = phone2int[c]
        else:
            # print(c) # unknown
            idx = 70
        seq.append(idx)
    return np.array(seq)

def notes_to_pc(notes, resolution, total_length):

    pc = np.full(shape=(total_length,), fill_value=46, dtype=np.short)

    for i in np.arange(len(notes[0])):
        pitch = notes[0][i]
        if pitch == -100:
            pc[0:total_length] = pitch
        else:
            times = np.floor(notes[1][i] / resolution)
            st = int(np.max([0, times[0]]))
            ed = int(np.min([total_length, times[1]]))
            pc[st:ed] = pitch
    
    return pc

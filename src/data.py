import random
import torch
import numpy as np
import utils as u
import config as c


class DatasetAudio:

    def __init__(self, file_list, sample_len, cutoff, filters):

        random.Random().shuffle(file_list)   # shuffle file list

        self.file_list = file_list
        self.input_len = sample_len
        self.cutoff = cutoff
        # filters: (type, order)
        if not isinstance(filters, list):
            filters = [filters]
        self.filters = filters

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            hq, sr = u.read_audio(self.file_list[idx])  # high-quality target

            # take a chunk starting at random location
            x_length = hq.shape[1]
            start_loc = random.randint(0, x_length - self.input_len - 1)
            hq = hq[:, start_loc:start_loc + self.input_len]
            # select filter randomly from the list
            random_filter = random.choice(self.filters)
            # apply low-pass filter
            lq = u.lowpass(hq, self.cutoff, filter_=random_filter)   # low-quality input

            hq = torch.from_numpy(hq)   # convert to torch tensor
            lq = torch.from_numpy(lq)   # convert to torch tensor

            return lq, hq    # input, target
        except:
            # In case of a problem, Nones are filtered out later.
            return None


class SingleSong:
    # To load one excerpt with arbitrary length, or one full song, for test or validation
    def __init__(self, chunk_len, filter_, hq_path, cutoff, duration=None, start=8):

        hq, sr = u.read_audio(hq_path)    # high quality target
        lq = u.lowpass(hq, cutoff, filter_=filter_)  # low quality input

        # CROP
        song_len = lq.shape[-1]

        if duration is None:    # save entire song
            test_start = 0
            test_len = song_len
        else:
            test_start = start * sr    # start from n th second
            test_len = duration * sr
            
        test_len = min(test_len, song_len - test_start)    

        lq = lq[:, test_start:test_start + test_len]
        hq = hq[:, test_start:test_start + test_len]

        self.x_full = lq.copy()
        self.t_full = hq.copy()

        # To have equal length chunks for minibatching
        time_len = lq.shape[-1]
        n_chunks, rem = divmod(time_len, chunk_len)
        lq = lq[..., :-rem or None]    # or None handles rem=0
        hq = hq[..., :-rem or None]    

        # adjust lengths
        self.x_full = self.x_full[..., :lq.shape[-1] or None]
        self.t_full = self.t_full[..., :lq.shape[-1] or None]

        # Save full samples
    
        self.lq = np.split(lq, n_chunks, axis=-1)   # create a lists of chunks
        self.hq = np.split(hq, n_chunks, axis=-1)   # create a lists of chunks

    def get_full_signals(self):
        # Returns full length input and target
        return self.x_full, self.t_full

    def preallocate(self):
        """
        Preallocates the matrix to save all minibatch outputs.
        It is faster to transfer all minibatches from GPU to CPU at once.
        """
        return torch.zeros((len(self.lq), *self.lq[0].shape), dtype=torch.float32, device=c.DEVICE)

    def __len__(self):
        return len(self.lq)

    def __getitem__(self, idx):
        return self.lq[idx], self.hq[idx]

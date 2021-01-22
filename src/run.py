import os
from time import time
import glob
import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
import torch.nn.functional as F

import config as c
from pydoc import locate
GenModel = locate('models.' + c.MODEL + '.Model')

import utils as u
from data import SingleSong, DatasetAudio
from collate import filter_collate


class Runner:
    def __init__(self):

        self.t0 = time()
        self.hours_pretrained = 0

        # DATA

        if c.TEST_ONLY:
            print('TESTING ONLY')
            self.tst_file_list = []
            for data_dir in c.TEST_DIRS:
                self.tst_file_list += sorted(glob.glob(data_dir + '/**/*.wav', recursive=True))
        else:
            print('Generator: {}'.format(c.MODEL))
            print('Training filter(s): {}'.format(str(c.FILTERS_TRAIN)))
            print('Unseen validation filter(s): {}'.format(str(c.FILTERS_VALID)))
            trn_file_list = []
            for data_dir in c.TRAIN_DIRS:
                trn_file_list += sorted(glob.glob(data_dir + '/**/*.wav', recursive=True))
            # use last 8 songs as validation
            n_songs_valid = min(c.N_SONGS_VALID, len(trn_file_list)-1)
            self.val_file_list = trn_file_list[-n_songs_valid:]
            trn_file_list = trn_file_list[:-n_songs_valid]
            trn_dataset = DatasetAudio(trn_file_list, c.SAMPLE_LEN, c.CUTOFF, c.FILTERS_TRAIN)
            collate_fn = filter_collate
            self.trn_loader = DataLoader(trn_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                num_workers=c.NUM_WORKERS, collate_fn=collate_fn)

        # GENERATOR
        self.gen_model = GenModel(batchnorm=c.BATCHNORM, dropout=c.DROPOUT).to(c.DEVICE)
        self.gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=c.LEARNING_RATE)
        if c.ADAPTIVE_LR:   # reduces learning rate
            if 'loss' in c.METRIC_TRAIN:
                scheduler_mode = 'min'
            elif 'snr' in c.METRIC_TRAIN:
                scheduler_mode = 'max'
            self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.gen_optimizer, mode=scheduler_mode, factor=c.LR_FACTOR, threshold=1e-6,
                verbose=False, patience=c.PATIENCE, cooldown=c.PATIENCE)

        # LOSS METRICS
        self.mse_loss = torch.nn.MSELoss().to(c.DEVICE)

        self.iter_total = 0
        self.iter_val = c.ITER_VAL
        self.first_epoch = True

        if c.LOAD_MODEL:
            self.load_model()
        else:
            self.gen_model.apply(u.weights_init_normal)

    def train(self):

        self.gen_model.train()
        train_averager = u.MovingAverages()   # moving average of losses
        while self.iter_total < c.MAX_ITER and self.gen_optimizer.param_groups[0]['lr'] > c.MIN_LR:

            for x, t in self.trn_loader:      # Loads batch to CPU in parallel
                x = x.to(c.DEVICE)   # LQ input
                t = t.to(c.DEVICE)   # HQ target

                self.gen_optimizer.zero_grad()
                y = self.gen_model(x)   # run generator
                gen_loss = self.mse_loss(y, t)
                gen_loss.backward()     # Calculate gradient                
                self.gen_optimizer.step()   # Update parameters
                with torch.no_grad():
                    train_snr = u.snr_torch(y, t)

                train_averager({'gen_loss': gen_loss, 'snr': train_snr})
                self.iter_total += 1

                if self.iter_total % self.iter_val == 0:
                    # Print training performance
                    train_performance = train_averager.get()
                    train_averager.reset()

                    # VALIDATION
                    if c.VALID:
                        # perform validation using seen filters
                        val_seen_performance = self.run_songs(c.FILTERS_TRAIN, 'valid')
                        # perform validation using unseen filters
                        val_unseen_performance = self.run_songs(c.FILTERS_VALID, 'valid')
                        self.print_performance(train_performance, val_seen_performance, val_unseen_performance)                  

                    if c.SAVE_MODEL:
                        self.save_model()

                    lr_old = self.gen_optimizer.param_groups[0]['lr']
                    if c.ADAPTIVE_LR:
                        # adjusts learning rate if necessary
                        self.gen_scheduler.step(train_performance['gen_loss'])     
                        # self.scheduler.step(disc_train_loss)
                    lr_new = self.gen_optimizer.param_groups[0]['lr']
                    if lr_new != lr_old:
                        print('New learning rate: {:.1e}'.format(lr_new))

                    self.first_epoch = False
              
    def run_songs(self, filters, mode):
        """Sets parameters for different modes (testing or validation),
        and calls the run_single_song function while looping songs.

        Args:
            filters (list): List of filters as tuple (type, order)
                Should be a list even for a single filter [(type, order)]
            mode (str): valid or test

        Returns:
            dict: Performance values (SNR etc)
        """

        self.gen_model.eval()
        averager = u.MovingAverages()

        output = True
        overwrite = True
        if mode is 'valid':
            song_list = self.val_file_list
            duration = c.DURATION_VALID
            start = c.START_VALID
        elif mode is 'test':
            song_list = self.tst_file_list
            duration = c.DURATION_TEST
            start = 0
        else:
            raise ValueError('Mode can be valid or test')

        with torch.no_grad():   # does not keep gradients
            # RUN FOR FULL SONGS
            for i, song_path in enumerate(song_list):
                if len(filters) == 1:
                    filter_ = filters[0]
                else:
                    # For multi-filter validation, matches each filter with each song
                    filter_ = filters[i]

                results = self.run_single_song(song_path, filter_, c.CUTOFF, duration=duration, start=start,
                                               save=output, overwrite=overwrite)
                averager(results)

            performance = averager.get()
            averager.reset()
            return performance

    def save_model(self):
        data = {
            'iteration': self.iter_total,
            'hours': self.hours_pretrained,
            'gen_model_state_dict': self.gen_model.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict()
        }
        torch.save(data, os.path.join(c.MODEL_DIR, c.SAVE_NAME + ".pt"))

    def load_model(self):
        checkpoint = torch.load(os.path.join(c.MODEL_DIR, c.LOAD_MODEL), map_location=c.DEVICE)
        self.iter_init = checkpoint['iteration']
        self.iter_total = checkpoint['iteration']
        self.hours_pretrained = checkpoint['hours']
        self.gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        print('Model loaded from {}'.format(os.path.join(c.MODEL_DIR, c.LOAD_MODEL)))
        if c.OVERWRITE_LR:  # reset learning rate
            for param_group in self.gen_optimizer.param_groups:
                param_group['lr'] = c.LEARNING_RATE
            message = 'Learning rate set to {:.2e}'.format(c.LEARNING_RATE)
            print(message)

    def run_single_song(self, hq_path, filter_, cutoff, duration=None, start=0, save=True, overwrite=True):
        """
        Runs the model for a single song.
        Chunks of audio is processed and the outputs are later concatenated to create full song.

        Args:
            hq_path (str): Path to high-quality audio
            filter_ (tuple): Type and order of lowpass filter to apply on hq audio
            cutoff (int): Cutoff frequency of the low-pass filter 
            duration (int, optional): Duration of audio to process. Defaults to None,
                which processes the entire audio.
            start (int, optional): Starting point, in seconds of audio to be processed. Defaults to 0.
            save (bool, optional): Setting False skips saving output, only calculates SNR and MSE. 
                Defaults to True.
            overwrite (bool, optional): To overwrite samples at different iterations during training. 
                Setting False can be useful to inspect generations during GAN training. Defaults to True.

        Returns:
            performance (dict): SNR and MSE values for input and output
        """

        self.gen_model.eval()       # switch model to inference mode
        with torch.no_grad():   # no training is done here
            # initialize dataloader
            song_data = SingleSong(c.WAV_SAMPLE_LEN, filter_, hq_path,
                                   cutoff=cutoff, duration=duration, start=start)
            song_loader = DataLoader(song_data, batch_size=c.WAV_BATCH_SIZE, shuffle=False, num_workers=c.NUM_WORKERS)
            
            y_full = song_data.preallocate()  # preallocation to keep individual output chunks
            song_averager = u.MovingAverages()

            # model works on chunks of audio, these are concatenated later
            idx_start_chunk = 0
            for x, t in song_loader:
                x = x.to(c.DEVICE)   # input
                t = t.to(c.DEVICE)   # target
                y = self.gen_model(x)   # output
                loss = F.mse_loss(y, t)
                song_averager({'loss': loss})
                idx_end_chunk = idx_start_chunk + y.shape[0]
                y_full[idx_start_chunk:idx_end_chunk] = y
                idx_start_chunk = idx_end_chunk

            y_full = u.torch2np(y_full)  # to cpu-numpy
            y_full = np.concatenate(y_full, axis=-1)    # create full song out of chunks

            x_full, t_full = song_data.get_full_signals()

            y_full = np.clip(y_full, -1, 1 - np.finfo(np.float32).eps)

            # Measure performance
            performance = song_averager.get()
            song_averager.reset()

            snr_ = u.snr(y_full, t_full)
            performance.update({'snr': snr_})

            if self.first_epoch:
                # Only need to see input SNR once
                snr_input = u.snr(x_full, t_full)
                performance.update({'input_snr': snr_input})

            if save:
                # Save audio
                song_name = hq_path.split('/')[-1].split('.')[0]
                if 'mixture' in song_name:  # DSD100 dataset has mixture.wav for all file names
                    song_name = hq_path.split('/')[-2]  # folder name is song name
                # Remove problematic characters
                problem_str = [' - ', ' & ', ' &', '\'', ' ']
                for s in problem_str:
                    song_name = song_name.replace(s, '_')

                if not overwrite:
                    song_name = u.pad_str_zeros(str(self.iter_total), 7) + '_' + song_name

                wavfile.write(os.path.join(c.GENERATION_DIR, song_name + '_' + filter_[0] + '.wav'),
                              c.SAMPLE_RATE, y_full.T)

            return performance

    def print_performance(self, train_performance, val_seen_performance, val_unseen_performance):
        """ Prints training and validation performance """
        seconds = int(time() - self.t0)
        hours = seconds / 3600.0
        hours += self.hours_pretrained

        msg = ''
        if self.first_epoch:
            msg += 'Validation input SNRs | Seen filter: {:6.2f} | Unseen filter: {:6.2f}\n'.format(
                val_seen_performance['input_snr'], val_unseen_performance['input_snr'])

        msg += '{:24s} | SNR: {:6.2f} | Gen loss: {:7.2e}'.format(
            'Training', train_performance['snr'], train_performance['gen_loss'],
        )

        msg += '\n{:24s} | SNR: {:6.2f} | Gen loss: {:7.2e}'.format(
            'Validation | Seen filter', val_seen_performance['snr'], val_seen_performance['loss']
        )
        msg += ' | Unseen filter | SNR: {:6.2f} | Gen loss: {:7.2e}'.format(
            val_unseen_performance['snr'], val_unseen_performance['loss']
        )
        msg += ' | Iterations:{:7d} | Elapsed:{:3.1f}h\n'.format(self.iter_total, hours)
        print(msg)


    def run(self):
        if not c.TEST_ONLY:
            self.train()
        else:   # TEST
            for filter_ in c.FILTERS_TEST:
                tst_performance = self.run_songs([filter_], 'test')
                minutes = (time() - self.t0) / 60.0
                print(('Filter: {:2.0f}th order {:8s} | SNRs: | '
                       'Input: {:6.2f} | Output: {:6.2f} | Elapsed: {:4.1f}m').format(
                    filter_[1], filter_[0], tst_performance['input_snr'],
                    tst_performance['snr'], minutes))


if __name__ == '__main__':
    runner = Runner()
    runner.run()
    print('End of run')

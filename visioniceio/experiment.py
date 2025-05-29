"""Exeriment class for VisionICeIO.

Reads the metadata and the binary files from one experiment.

Basically, wrapping the data into a Xarray Dataset.

Currently storing the data as Zarr.
"""


import os

import numpy as np
import xarray as xr
from numcodecs import Blosc as BloscCodec

from .core_io import read_metadata, read_data


class Experiment:
    """One directory in the current workflow structure."""
    def __init__(self):
        self.path = None
        self.name = None
        self.data = None
        self.metadata = None
        self._suffix = ['.swa', '.spi', '.stm', '.ana']  # suffices for the files to load
        self._edims = [2, 1, 1, 1]  # expected dimensions of the trial-electrode data block
        self._edtypes = ['int16', 'int32', 'int32', 'int16']
        self.pad_value = np.nan  # will be zero, because it is int data

    def load_from_dir(self, path=None, name=None, save_as='zarr'):
        """Load the data and metadata from the experiment directory."""
        self.path = path
        self.name = name
        self.metadata = read_metadata(os.path.join(self.path, self.name + '-ifo.txt'))
        self.sample_rate_spike = self.metadata['SpikeSamplingFrequency']
        self.sample_rate_lfp = self.metadata['AnalogSamplingFrequency']
        self.snippet_points = self.metadata['NofPointsSpikewaveform']
        self.lfp_points = self.metadata['MaxTrialLength']
        self.ntrials = self.metadata['NofTrials']
        self.nelectrodes = self.metadata['NofSpikeChannels']
        self._n_spikes = None
        self.max_spikes = None
        
        for suf, edm, edt in zip(self._suffix, self._edims, self._edtypes):
            f_ = os.path.join(self.path, self.name + suf)
            if not os.path.exists(f_):
                raise FileNotFoundError(f"File {f_} does not exist.")

            dlist = read_data(f_, edt, edm)
            dlenght = len(dlist)  # trials x electrodes

            if self.max_spikes is None and (suf == '.swa' or suf == '.spi'):
                self._n_spikes = np.array([si.shape[0] for si in dlist], dtype=np.int32)
                self.max_spikes = self._n_spikes.max()

            if suf == '.swa':
                # TODO: add scaling factor to get real units
                d_ = np.array(
                    [np.pad(da.astype(np.float32), 
                            ((0, self.max_spikes - da.shape[0]), (0, 0)), 
                            'constant', constant_values=np.nan
                            ) for da in dlist], dtype=np.float16)
                d_ = d_.reshape(self.ntrials, self.nelectrodes, self.max_spikes, self.snippet_points)
                d_ = d_.swapaxes(0, 1)
                d_ = np.ascontiguousarray(d_)  # better memory layout

                self.waveforms = xr.DataArray(
                    data=d_,
                    name='waveforms',
                    dims=('electrodes', 'trials', 'spikes_idx', 'snippet_time'),
                    coords={
                        'electrodes': np.arange(self.nelectrodes),
                        'trials': np.arange(self.ntrials),
                        'spikes_idx': np.arange(self.max_spikes),
                        'snippet_time': np.arange(self.snippet_points)/self.sample_rate_spike,
                    },
                )

                self.n_spikes = xr.DataArray(
                    data=self._n_spikes.reshape(self.ntrials, self.nelectrodes),
                    name='n_spikes',
                    dims=('trials', 'electrodes'),
                    coords={
                        'trials': np.arange(self.ntrials),
                        'electrodes': np.arange(self.nelectrodes),
                    },
                )

                self.data = xr.Dataset(
                    data_vars={
                        'waveforms': self.waveforms,
                        'n_spikes': self.n_spikes,
                    },
                    attrs=self.metadata,
                )

            elif suf == '.spi':
                d_ = np.array(
                    [np.pad(da/self.sample_rate_spike, 
                            (0, self.max_spikes - da.shape[0]), 
                            'constant', constant_values=np.nan) for da in dlist], 
                    dtype=np.float32)
                d_ = d_.reshape(self.ntrials, self.nelectrodes, self.max_spikes)
                d_ = d_.swapaxes(0, 1)  # change from trials x electrodes to electrodes x trials
                d_ = np.ascontiguousarray(d_)  # better memory layout

                self.spike_times = xr.DataArray(
                    data=d_,
                    name='spike_times',
                    dims=('electrodes', 'trials', 'spikes_idx'),
                    coords={
                        'electrodes': np.arange(self.nelectrodes),
                        'trials': np.arange(self.ntrials),
                        'spikes_idx': np.arange(self.max_spikes)/self.sample_rate_spike,
                    },
                )

                self.data = self.data.merge(self.spike_times.to_dataset())

            elif suf == '.stm':
                d_ = np.array(dlist[0], dtype=edt)

                self.stim_label = xr.DataArray(
                    data=d_,
                    name='stim_label',
                    dims=['trials'],
                    coords={
                        'trials': np.arange(self.ntrials),
                    },
                )
                self.data = self.data.merge(self.stim_label.to_dataset())

            elif suf == '.ana':
                d_ = np.array(dlist, dtype=edt).reshape(self.ntrials, self.nelectrodes, self.lfp_points)
                d_ = d_.swapaxes(0, 1)
                d_ = np.ascontiguousarray(d_)

                self.lfp = xr.DataArray(
                    data=d_,
                    name='lfp',
                    dims=('electrodes', 'trials', 'lfp_time'),
                    coords={
                        'electrodes': np.arange(self.nelectrodes),
                        'trials': np.arange(self.ntrials),
                        'lfp_time': np.arange(self.lfp_points)/self.sample_rate_lfp,
                    },
                )
                self.data = self.data.merge(self.lfp.to_dataset())
                
        if save_as == 'zarr':
            compressors = [BloscCodec(cname="zstd", clevel=1, shuffle='noshuffle')]
            encoding = {var: {"compressors": compressors} for var in self.data.data_vars}
            output_path = os.path.join(self.path, f"{self.name}.zarr")
            self.data.to_zarr(
                output_path, 
                mode="w",
                consolidated=True,
                write_empty_chunks=False,
                encoding=encoding,
                )

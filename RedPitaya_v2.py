"""
Created on 09/25/2025

@author: mason mandernach
"""
import math

from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os

N_FFT_SHOW = 10

class RedPitaya:

    electrode_map = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True), '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True), '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='lockin_config', hostname='169.254.131.37')
        self.output_dir = output_dir

        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.lockin_Y = []

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival

        self.scope = self.rp_modules.scope
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = 1
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6/self.scope.decimation

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1/self.ref_freq
        ref_amp = params['ref_amp']

        self.ref_sig.setup(waveform='sin',
                           amplitude=ref_amp,
                           frequency=self.ref_freq)

        self.ref_start_t = time.time()

        if params['output_ref'] == 'out1' or params['output_ref'] == 'out2':
            self.ref_sig.output_direct = params['output_ref']
        else:
            self.ref_sig.output_direct = 'off'

        self.lockin.setup(frequency=self.ref_freq,
                       bandwidth=[-self.ref_freq * 2, -self.ref_freq, self.ref_freq, self.ref_freq * 2],  # Hz
                       gain=1.0,
                       phase=((time.time() - self.ref_start_t)/self.ref_period)*360,       #initial phase is in degrees (delta t[ns])-> delta t [s]/(1/f) * 360
                       acbandwidth=0,
                       amplitude=ref_amp,
                       input='in1',
                       output_direct='off',
                       output_signal='output_direct',
                       quadrature_factor=1)

    def setup_pid(self, params):
        self.pid.input = 'iq2'
        self.pid.output_direct = 'out1'
        self.pid.setpoint = params['setpoint']
        self.kp = params['kp']
        self.ki = params['ki']
        self.ival = 0

    def get_ref_phase(self):
        delta_t = (time.time() - self.ref_start_t)
        num_periods = delta_t/(1/self.ref_freq)
        partial = num_periods % 1.0
        return partial

    def get_pid_output(self):
        self.scope.input1 = self.pid.output_direct
        output = np.array(self.scope._data_ch1_current)
        self.scope.input1 = 'iq2'
        return output[-1] / 360

    def get_current_lia_phase(self):
        last_Y = self.lockin_Y[-1]
        last_X = self.lockin_X[-1]
        return math.atan(np.mean(last_Y)/np.mean(last_X))

    def capture_lockin(self):
        """
        captures a self.scope.decimation length capture and appends them to the X and Y arrays
        :return:
        """
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)

        if self.scope.input1 == 'iq2' and self.scope.input2 == 'iq2_2':
            self.lockin_X.append(ch1)
            self.lockin_Y.append(ch2)

        return ch1, ch2

    def see_fft(self, save_file=False):
        iq = self.lockin_X + 1j*self.lockin_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

        idx = np.argmax(psd_lock)  # PSD computed as above
        print("Peak at", freqs_lock[idx], "Hz")

        plt.figure(1, figsize=(12, 4))

        plt.semilogy(freqs_lock, psd_lock, label='Lock-in R')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('Lock-in Output Spectrum (baseband)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        if save_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            path = os.path.join(self.output_dir, f'lockin_FFT_rf_{self.ref_freq}.png')
            plt.savefig(path)
        else:
            plt.show()

    def run(self, params):
        timeout = run_params['timeout']
        loop_f = run_params['loop_f']

        self.setup_lockin(params)

        if params['lock']:
            self.setup_pid(params)
        time.sleep(0.01)


        loop_start = time.time()
        ref_theta = []
        lia_theta = []

        while (time.time() - loop_start) < timeout:
            self.capture_lockin()
            ref_theta.append(self.get_ref_phase())
            lia_theta.append(self.get_current_lia_phase())
            # self.lockin.phase = self.get_pid_output()
            time.sleep(1/loop_f)

        if params['fft']:
            self.see_fft(save_file=params['save_file'])
        else:
            X, Y = np.concatenate(self.lockin_X), np.concatenate(self.lockin_Y)
            t = np.arange(start=0, stop=len(X)/self.sample_rate, step=1/self.sample_rate)
            plt.plot(t, X)
            plt.plot(t, Y)
            plt.title('Lockin Results')
            plt.xlabel('Time (s)')
            plt.ylabel('X and Y outputs')
            plt.legend(['X', 'Y'])
            # plt.plot(ref_theta)
            # plt.plot(lia_theta)
            # plt.legend(['ref_theta', 'lia_theta'])
            # plt.grid(True)

            if params['save_file']:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')
                plt.savefig(path)
            else:
                plt.show()


if __name__ == '__main__':

    rp = RedPitaya()

    run_params = {
        'ref_freq': 100,       # Hz, reference signal frequency for lock-in
        'ref_amp': 0.4,        # V, amplitude of reference signal
        'output_ref': '',  # where to output the ref_signal

        'setpoint': 0.0,    # target setpoint for PID
        'kp': 1.0,          # proportional gain
        'ki': 0.1,          # integral gain

        'lock': True,       # whether or not to phase lock
        'timeout': 5.0,     # seconds, how long to run acquisition loop
        'loop_f': rp.sample_rate/10,       # Hz, how often to capture data in run loop

        'test_freq': 100,           # Hz, used for test or calibration runs
        'test_amp': 0.4,            # V, amplitude of test waveform
        'noise_freq': 10000,        # Hz, optional noise injection frequency
        'noise_amp': 0.1,           # V, optional noise amplitude

        'output_dir': 'test_data',  # where to save FFT and waveform plots
        'save_file': False,         # whether to save plots instead of showing them
        'fft': False,               # whether to perform FFT after run
    }

    rp.run(run_params)



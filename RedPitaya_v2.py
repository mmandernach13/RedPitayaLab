"""
Created on Tue Jan  7 15:17:48 2025

@author: jayse
"""

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
        self.rp = Pyrpl(config='lockin_config', hostname='169.254.148.168')
        self.output_dir = output_dir

        self.rp_modules = self.rp.rp
        self.lia_scope = self.rp_modules.scope
        self.lia_scope.input1 = 'iq2'
        self.lia_scope.input2 = 'iq2_2'
        self.lia_scope.decimation = 16384
        self.lia_scope.average = 'true'
        self.sample_rate = 15300
        self.iq2 = self.rp_modules.iq2

    def capture_single_lockin(self):
        self.lia_scope.single()
        ch1 = np.array(self.lia_scope._data_ch1_current)
        ch2 = np.array(self.lia_scope._data_ch2_current)
        return ch1, ch2

    def run(self, params, save_file=False):

        ref_freq = params['ref_freq']
        ref_amp = params['ref_amp']
        test_freq = params['test_freq']
        test_amp = params['test_amp']

        self.test_sig = self.rp_modules.asg0

        self.test_sig.setup(waveform='sin',
                            frequency=test_freq,
                            amplitude=test_amp,
                            offset=0.00,
                            output_direct='out2',
                            trigger_source='immediately')

        self.iq2.setup(frequency=ref_freq,
                       bandwidth=[-ref_freq*0.99, -ref_freq*0.99,
                                  ref_freq*1.01, ref_freq*1.01],  # Hz
                       gain=0.0,
                       phase=0,
                       acbandwidth=0,
                       amplitude=ref_amp,
                       input='asg0',
                       output_direct='out1',
                       output_signal='quadrature',
                       quadrature_factor=10)

        X, Y = self.capture_single_lockin()

        plt.plot(X[1000:], Y[1000:])
        plt.title('Lockin Results')
        plt.xlabel('Magnitude (X output)')
        plt.ylabel('Phase (Y output)')

        if save_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            path = os.path.join(self.output_dir, f'lockin_results_tf_{test_freq}_rf_{ref_freq}.png')
            plt.savefig(path)
        else:
            plt.show()


if __name__ == '__main__':

    rp = RedPitaya()

    run_params= {
        'test_freq': 500,
        'test_amp': 0.2,
        'ref_freq': 50,
        'ref_amp': 0.2
    }

    rp.run(run_params, save_file=True)



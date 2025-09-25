import time
from pyrpl import Pyrpl
import matplotlib.pyplot as plt
import numpy as np

hostname = "169.254.131.37"
# Connect
lia_rp = Pyrpl(config="mason_config", hostname=hostname)
wave_rp = lia_rp

# Setup modules
lock_in_modules = lia_rp.rp
wave_modules = wave_rp.rp

lockin_iq0 = lock_in_modules.iq0  # In-phase
lockin_iq1 = lock_in_modules.iq1  # Quadrature
lockin_scope = lock_in_modules.scope
wave_gen = wave_modules.asg1

def wave_on(wave_gen):
    wave_gen.output_direct = 'out1'
    wave_gen.setup(
        waveform='sin',
        frequency=500,
        amplitude=1.0,
        offset=0,
        trigger_source='immediately'
    )
    return

def wave_off(wave_gen):
    wave_gen.output_direct = 'off'
    return

# 1. Waveform Generator Setup


# 2. Lock-in Amplifier Setup
lockin_iq0.setup(frequency=1e3,
                 bandwidth=[0,0],
                 gain=0.0,
                 phase=0,  # In-phase
                 acbandwidth=0,
                 amplitude=0.5,
                 input='in1',
                 output_direct='off',
                 output_signal='quadrature',
                 quadrature_factor=0)

lockin_iq1.setup(frequency=1e3,
                 bandwidth=[50, 100],
                 gain=0.0,
                 phase=90,  # Quadrature (90 shifted)
                 acbandwidth=10,
                 amplitude=0.5,
                 input='in1',
                 output_direct='off',
                 output_signal='quadrature',
                 quadrature_factor=10)

# 3. Scope Setup
lockin_scope.input1 = 'out1'  # In-phase (X)
lockin_scope.input2 = 'iq1'  # Quadrature (Y)
lockin_scope.decimation = 16384
lockin_scope.average = 'true'
lockin_scope._start_acquisition_rolling_mode()
if __name__ == "__main__":
    wave_on(wave_gen)

    input()



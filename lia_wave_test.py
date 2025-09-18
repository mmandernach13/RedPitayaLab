import time
from pyrpl import Pyrpl
import matplotlib.pyplot as plt
import numpy as np
import csv

f = 50
# Connect
lia_rp = Pyrpl(config="", hostname="169.254.131.37")

# Setup modules
lock_in_modules = lia_rp.rp
lockin_signal = lock_in_modules.asg0
# lockin_noise = lock_in_modules.asg1
lockin_iq0 = lock_in_modules.iq0
lockin_scope = lock_in_modules.scope

lockin_signal.output_direct = 'out1'
lockin_signal.setup(waveform='halframp', frequency=f, amplitude=0.8, offset=0, trigger_source='immediately')
# lockin_noise.output = 'out1'
# lockin_noise.setup(waveform='noise', frequency=f, amplitude=0.4, offset=0, trigger_source='immediately')

# Lock-in Amplifier Setup
lockin_iq0.setup(frequency=f,
                 bandwidth=[100, 500],
                 gain=0.0,
                 phase=0,
                 acbandwidth=0,
                 amplitude=0.9,
                 input='in1',
                 output_direct='out2',
                 output_signal='quadrature',
                 quadrature_factor=10)

# Scope Setup
lockin_scope.input1 = 'out1'
lockin_scope.input2 = 'out2'
lockin_scope.average = 'true'
lockin_scope._start_acquisition_rolling_mode()

# Wait for rolling buffer to fill
time.sleep(1)

# Grab and process data
x = lockin_scope._data_ch1_current  # In-phase (X)
sin_input = lockin_scope._data_ch2_current

xaxis = np.arange(0, len(sin_input))

plt.plot(xaxis, x, color='red', label='reference signal')
plt.plot(xaxis, sin_input, color='blue', label='')
legend = plt.legend(loc='upper left')
plt.show()
# Save to CSV
# Combine the arrays into a 2D array (column-wise)
# combined_data = np.column_stack((x, sin_input))
# Save to a single CSV with headers for both columns
# np.savetxt("lockin_data.csv", combined_data, delimiter=",", header="X (In-phase Signal), Sin Input", comments="")
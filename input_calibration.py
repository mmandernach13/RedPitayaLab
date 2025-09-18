import time
from matplotlib import pyplot as plt
import numpy as np
from pyrpl import Pyrpl
from tqdm import tqdm

rp_ip = "169.254.131.37"

rp_interface = Pyrpl(config="mason_config", hostname=rp_ip)

rp_con = rp_interface.rp

# take 30 seconds of data
print("Starting ADC calibration")
in1_avg, in2_avg = [],[]
n_tests = 10
n_samples = 3000
with tqdm(total=n_samples*n_tests) as pbar:
    for j in range(n_tests):
        times, data, data1 = [],[],[]
        t0 = time.time()
        for i in range(n_samples):
            times.append(time.time()-t0)
            data.append(rp_con.scope.voltage_in1)
            data1.append(rp_con.scope.voltage_in2)
            time.sleep(0.001)
            pbar.update()
        in1_avg.append(np.mean(data))
        in2_avg.append(np.mean(data1))

#plot results
#Average over 10 for IN1: -0.011468448893229167 V
#Average over 10 for IN2: -0.009819392903645835 V
print("Average over " + str(n_tests) + " for IN1: " + str(np.mean(in1_avg)) + ' V')
print("Average over " + str(n_tests) + " for IN2: " + str(np.mean(in2_avg)) + ' V')
x = np.arange(len(in1_avg))
plt.figure(1,(20,10))
plt.plot(x, in1_avg, color='b', label='IN1')
plt.plot(x, in2_avg, color='r', label='IN2')
plt.title("IN1+IN2 floating ADC")
plt.legend()
plt.show()


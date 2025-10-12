

from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from time import sleep,time


# PATH of control_hugo.py file
import sys
sys.path.append(r'C:\lia_pid_app\rp_lock-in_pid\resources\remote_control')
from control_hugo import red_pitaya_control,red_pitaya_app

AppName      = 'lock_in+pid'
host         = 'rp-f09094.local'
port         = 22  # default port
trigger_type = 6   # 6 is external trigger

#%%


filename = 'test.npz'
rp=red_pitaya_app(AppName=AppName,host=host,port=port,filename=filename,password='root')
osc = red_pitaya_control(cmd='/root/py/osc.py', name='osc', parent=rp)
osc.load()

# reduce log noise on Windows platform
import logging
logging.basicConfig()
logging.getLogger("paramiko").setLevel(logging.WARNING)
rp.verbose = False

print(osc.get_data())
# print(rp)
# print(rp.lock)
# print(rp.osc)
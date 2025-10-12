

from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from time import sleep,time


# PATH of control_hugo.py file
from app_source.resources.remote_control.control_hugo import red_pitaya_control,red_pitaya_app

AppName      = 'lock_in+pid'
host         = 'rp-f09094.local'
port         = 22  # default port
trigger_type = 6   # 6 is external trigger

#%%


filename = 'test.npz'
rp=red_pitaya_app(AppName=AppName,host=host,port=port,filename=filename,password='root')

# reduce log noise on Windows platform
import logging
logging.basicConfig()
logging.getLogger("paramiko").setLevel(logging.WARNING)
rp.verbose = False

# print(rp)
# print(rp.lock)
# print(rp.osc)
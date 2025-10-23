import time
import numpy as np
import matplotlib.pyplot as plt
from pyrpl import Pyrpl
from pyrpl.modules import *

r = Pyrpl(config='', hostname='169.254.131.37')
rp = r.rp

s = rp.scope
p = rp.pid0
i = rp.iq2

print(s.voltage_out2)


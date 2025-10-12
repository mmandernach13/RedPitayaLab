import time
import numpy as np
import matplotlib.pyplot as plt
from pyrpl import Pyrpl

r = Pyrpl(config='liatest.yaml', hostname='169.254.131.37')
rp = r.rp

pid = rp.pid0


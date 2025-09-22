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

class RedPitaya:

    electrode_map = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True), '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True), '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}

    def __init__(self):
        self.rp = Pyrpl(hostname='169.254.131.37')

        self.rp_modules = self.rp.rp
        self.lia_scope = self.rp_modules.scope
        self.lia_scope.input1 = 'iq0'
        self.lia_scope.input2 = 'iq0_2'
        self.lia_scope.decimation = 16384



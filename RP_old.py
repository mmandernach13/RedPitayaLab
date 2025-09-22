# -*- coding: utf-8 -*-
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
    current_range_map = {'10uA': (False, True, True, True),'100uA': (True, False, True, True),'1mA': (True, True, False, True),'10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False),'5X': (False, True), '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65,'1mA': 600,'100uA': 6000,'10uA': 60000}
    
    
    def __init__(self):
        self.lia_rp = Pyrpl(config="", hostname="169.254.131.37") #f09094
        
        self.lia_modules = self.lia_rp.rp
        self.lia_scope = self.lia_modules.scope
        self.lia_scope.input1 = 'iq2'
        self.lia_scope.input2 = 'iq2_2'
        self.lia_scope.decimation = 16384
        self.lia_scope._start_acquisition_rolling_mode()
        self.lia_scope.average = 'true'
        
        # self.acq_modules = self.acq_rp.rp
        # self.acq_scope = self.acq_modules.scope
        # self.acq_scope.input1 = 'in1'
        # self.acq_scope.input2 = 'out1'
        # self.acq_scope.decimation = 16384
        # self.acq_scope._start_acquisition_rolling_mode()
        # self.acq_scope.average = 'true'
        #
        # self.acq2_modules = self.acq2_rp.rp
        # self.acq2_scope = self.acq2_modules.scope
        # self.acq2_scope.input1 = 'in1'
        # self.acq2_scope.decimation = 16384
        # self.acq2_scope._start_acquisition_rolling_mode()
        # self.acq2_scope.average = 'true'
        
    def initialize_gpio(self):
        self.lia_gpio = self.lia_modules.hk
        
        # Set all N pins to output mode
        self.lia_gpio.expansion_N0_output = True
        self.lia_gpio.expansion_P1_output = True
        
#        self.lia_gpio.expansion_N1_output = True
#        self.lia_gpio.expansion_N2_output = True
#        self.lia_gpio.expansion_N3_output = True
#        self.lia_gpio.expansion_N4_output = True
#        self.lia_gpio.expansion_N5_output = True
#        self.lia_gpio.expansion_N6_output = True
#        self.lia_gpio.expansion_N7_output = True
#         
#        # Set P6 and P7 to output mode
#        self.lia_gpio.expansion_P1_output = True
#        self.lia_gpio.expansion_P2_output = True
        
#        
        self.pstat_end()
    
    def set_current_range(self, current_range):
        if current_range in RedPitaya.current_range_map:
            self.lia_gpio.expansion_P2 = RedPitaya.current_range_map[current_range][0]  #TIA_SW1_IN1
            self.lia_gpio.expansion_N1 = RedPitaya.current_range_map[current_range][1]  #TIA_SW1_IN2
            self.lia_gpio.expansion_N2 = RedPitaya.current_range_map[current_range][2]  #TIA_SW1_IN3
            self.lia_gpio.expansion_N3 = RedPitaya.current_range_map[current_range][3]  #TIA_SW1_IN4
    
    def set_working_electrode(self, electrode):
        self.lia_gpio = self.lia_modules.hk
        if electrode in RedPitaya.electrode_map:
            self.lia_gpio.expansion_N4 = RedPitaya.electrode_map[electrode][0]  #SW_ABCD_WRK_0
            self.lia_gpio.expansion_N5 = RedPitaya.electrode_map[electrode][1]  #SW_ABCD_WRK_1
        
    def set_dac_gain(self, dac_gain):
        if dac_gain in RedPitaya.dac_gain_map:
            self.lia_gpio.expansion_N6 = RedPitaya.dac_gain_map[dac_gain][0]  #DAC_GAIN_A0
            self.lia_gpio.expansion_N7 = RedPitaya.dac_gain_map[dac_gain][1]  #DAC_GAIN_A1   
    
    def pstat_configure (self,electrode, current_range, dac_gain):
        self.set_working_electrode(electrode)
        self.set_current_range(current_range)
        self.set_dac_gain(dac_gain)
    
    def pstat_start(self, amp, ramp_offset, f):
        self.lia_gpio.expansion_N0 = True    #SW_Ctr_elect
        self.lia_gpio.expansion_P1 = True    #SW_Display
        self.lia_modules.asg1.setup(waveform='ramp',amplitude=amp,offset=ramp_offset,frequency=f,trigger_source='immediately',output_direct='out1')

    def lcd_test(self):
        self.lia_gpio = self.lia_modules.hk
        self.lia_gpio.expansion_N0 = True  # SW_Ctr_elect
        self.lia_gpio.expansion_P1 = True  # SW_Display
        time.sleep(20)
        self.lia_gpio.expansion_N0 = False # SW_Ctr_elect
        self.lia_gpio.expansion_P1 = False  # SW_Display

    def gpio_switch_test(self):
        print('A')
        self.set_working_electrode('A')
        time.sleep(3)
        print('B')
        self.set_working_electrode('B')
        time.sleep(3)
        print('C')
        self.set_working_electrode('C')
        time.sleep(3)
        print('D')
        self.set_working_electrode('D')
        time.sleep(3)
        self.set_working_electrode('A')

    def pstat_end(self):
        self.lia_gpio.expansion_N0 = False
        self.lia_gpio.expansion_P1 = False
        self.lia_modules.asg1.output_direct = "off" 
        self.lia_modules.iq2.output_direct = "off"
    
    def run (self, params):
        # Extract parameters from the dictionary
        curr_range = params['curr_range']
        electrode = params['electrode']
        volt_range = params['volt_range']
        volt_min = params['volt_min']
        volt_max = params['volt_max']
        volt_per_sec = params['volt_per_sec']
        num_cycles = params['num_cycles']
        ac_amp = params['ac_amp']
        ac_freq = params['ac_freq']
        
        
        #this is to fix the inverted sign from the potentiostat circuit from DAC_BIP to DAC_BIP_NX
#        volt_ = [-volt_min,-volt_max]
#        volt_max_adj = max(volt_)
#        volt_min_adj = min(volt_)
        amplitude = (volt_max - volt_min)/2.0            # Waveform peak amplitude (V)
        offset = (volt_max + volt_min)/2.0               # Waveform offset (V)
        period = (4*amplitude/volt_per_sec)   # Waveform period in (ms)#
        dc_freq = 1/period*4
        runtime = period * num_cycles
        
        self.pstat_configure(electrode, curr_range, volt_range)
        self.pstat_start(amplitude, offset, dc_freq)
        scaling_factor = RedPitaya.current_scaling_map[curr_range]
        
        self.iq2 = self.lia_modules.iq2
        self.iq = self.acq_modules.iq0
        self.iq2.setup(frequency=ac_freq*4, bandwidth=[0,18.97,18.97,18.97], gain=0.0, phase=0, acbandwidth=0, amplitude=ac_amp,input='in1', output_direct='out1', output_signal='quadrature', quadrature_factor= 1000)

        self.iq.setup(input='in2', frequency=0, bandwidth=[0,9.486],gain=0.65, phase=0, acbandwidth=0, amplitude=0, output_direct='out1', output_signal='output_direct', quadrature_factor=0)  

        size_array = int(runtime/(0.00013242*4))
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        start=0
        end2=0
        interval2=0
        last_index = 0
        interval = 0
        end_array = 16350
        temp_R = np.zeros(16350)
        temp_iq2 = np.zeros(16350)
        temp_iq2_2 = np.zeros(16350)
        temp_current = np.zeros(16350)
        temp_dcramp = np.zeros(16350)
        temp_total_dc = np.zeros(16350)
        
        #all data before reduced 
        mag_R = np.zeros(size_array)    #not the same as signal
        mag_iq2 = np.zeros(size_array)
        mag_iq2_2 = np.zeros(size_array)
        mag_current = np.zeros(size_array)
        mag_dcramp = np.zeros(size_array)
        mag_time = np.zeros(size_array)
        mag_total_dc = np.zeros(size_array)

        #final array that get saved in excel
        self.mag_R_avg = []
        self.mag_iq2_avg = []
        self.mag_iq2_2_avg = []
        self.mag_current_avg = []
        self.mag_dcramp_avg = []
        self.mag_time_avg = []
        self.mag_total_dc_avg = []
        
        current_pointer = self.lia_scope._write_pointer_current
        #data acquisition
        while last_index < (size_array):
            print (last_index)

            start = time.time()+interval2
            #in array form
            temp_iq2 = self.lia_scope._data_ch1_current #X, 0.001 due to QF
            temp_iq2_2 = self.lia_scope._data_ch2_current #Y
            temp_R = np.sqrt(np.square(temp_iq2)+np.square(temp_iq2_2)) *10  #divide by QF (1000) times with 10^4 because of SEED signal too low. just it is easier to read the numbers
            temp_current = self.acq_scope._data_ch1_current/scaling_factor #depending current_range, 
            temp_dcramp = self.acq_scope._data_ch2_current
            temp_total_dc = -self.acq2_scope._data_ch1_current
            new_pointer = self.lia_scope._write_pointer_current
            
            #get new num_new_points 
            if new_pointer>current_pointer:
                num_new_points = new_pointer-current_pointer
            else:
                num_new_points = end_array+new_pointer-current_pointer
            if last_index+num_new_points>size_array:
                num_new_points = size_array-last_index
            
            #where you left off
            temp_array_index = end_array-num_new_points
            
            #organizing
            mag_iq2[last_index:last_index+num_new_points] = temp_iq2[temp_array_index:temp_array_index+num_new_points] 
            mag_iq2_2[last_index:last_index+num_new_points] = temp_iq2_2[temp_array_index:temp_array_index+num_new_points]
            mag_R[last_index:last_index+num_new_points] = temp_R[temp_array_index:temp_array_index+num_new_points]
            mag_current[last_index:last_index+num_new_points] = temp_current[temp_array_index:temp_array_index+num_new_points]
            mag_dcramp[last_index:last_index+num_new_points] = temp_dcramp[temp_array_index:temp_array_index+num_new_points]
            mag_total_dc[last_index:last_index+num_new_points] = temp_total_dc[temp_array_index:temp_array_index+num_new_points]
            end = time.time()
            temp_time = np.linspace(interval, interval + end-start, num_new_points)
            mag_time[last_index:last_index+num_new_points] = temp_time
            interval=interval+(end-start)
            
            #Reducing the number of points by 10 to 1 (105723, from size_array reduced by 10)
            self.mag_iq2_avg = mag_iq2[0:last_index+num_new_points:10]   #because of 10^-4
            self.mag_iq2_2_avg = mag_iq2_2[0:last_index+num_new_points:10]
            self.mag_R_avg = mag_R[0:last_index+num_new_points:10]
            self.mag_current_avg = mag_current[0:last_index+num_new_points:10]
            self.mag_dcramp_avg = mag_dcramp[0:last_index+num_new_points:10]
            self.mag_time_avg = mag_time[0:last_index+num_new_points:10]
            self.mag_total_dc_avg = mag_total_dc[0:last_index+num_new_points:10]
            self.mag_signal_avg = np.divide(self.mag_R_avg,self.mag_total_dc_avg)
           
            last_index = last_index+num_new_points
            current_pointer = new_pointer
            #comment this out if you dont wanna plot, this part of the code causes the plotted data to be bad sometimes
            ax2.plot(-self.mag_time_avg[150:], self.mag_R_avg[150:], color='green')#
            ax1.plot(-self.mag_time_avg[150:], self.mag_dcramp_avg[150:], color='blue')#
            plt.show()  #comment this out if you dont wanna plot
            plt.pause(.5)#
            end2 = time.time()
            interval2=end2-end

        self.pstat_end()
        
                
    def savefile (self, filename):
        tempDir=os.getcwd()            
        folder = 'C:\\Users\\lab\\Desktop\\Small_seed_data'
        #self.createFolder(csvDirectory) # This function makes your folder
        os.chdir(folder)
        os.mkdir(filename)
        os.chdir(folder + '\\' + filename)
        
        with open('magnitude.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_R_avg[150:])
            
        with open('signal.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_signal_avg[150:])
            
        with open('in_phase.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_iq2_avg[150:])
            
        with open('out_phase.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_iq2_2_avg[150:])
        
        with open('dcRamp.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_dcramp_avg[150:])

        with open('time.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(-self.mag_time_avg[150:])
          
        with open('current.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_current_avg[150:])
        
        with open('total_dc.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.mag_total_dc_avg[150:])
            
        with open('phase.csv', 'wb') as csvfile:
            phase = np.arctan(self.mag_iq2_2_avg[150:], self.mag_iq2_avg[150:])
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(phase)
        
        
        os.chdir(tempDir)
        
#    def createFolder(self,directory):
#        try:
#            if not os.path.exists(directory):
#                os.makedirs(directory)
#        except OSError:
#            print ('Error: Creating directory. ' +  directory)
    
if __name__ == "__main__":
    RP = RedPitaya()
    RP.initialize_gpio()
#    params = {
#    'curr_range': '10mA',       # fixed
#    'electrode': 'A',           # fixed
#    'volt_range': '1X',         # fixed
#    'volt_min': -0.5,           # Minimum voltage in waveform (V)
#    'volt_max': 0.4,            # Maximum voltage in waveform (V)
#    'volt_per_sec': 1,          # Voltage transition rate (V/s)
#    'num_cycles': 10,           # Number of cycles
#    'ac_amp': 0.2,              # AC amplitude (V)
#    'ac_freq': 500              # AC frequency (Hz)
#}

"""
    interface with lockin+pid app to load configuration, and acquire data
    load the settings on the web app before running this script
"""
import os

import numpy as np
from matplotlib import pyplot as plt
import time as time

from app_source.resources.remote_control.control_hugo import red_pitaya_app
from app_source.resources.remote_control.read_dump import read_dump

AppName      = 'lock_in+pid'
host         = 'rp-f09094.local'
port         = 22  # default port
trigger_type = 6   # 6 is external trigger
decimation = 1  # [1,8,64, 1024, 8192, 65536]
filename = 'test.npz'

def rp_connect():
    rp=red_pitaya_app(AppName=AppName,host=host,port=port,filename=filename,password='root')

    # reduce log noise on Windows platform
    import logging
    logging.basicConfig()
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    rp.verbose = False

    return rp

def calib_dac_adc(rp=None, dec=decimation):
    rp.get_adc_dac_calib()

    print(repr(rp.calib_params))

    rp.osc_trig_fire(trig=trigger_type, dec=dec)
    time.sleep(dec * 8e-9 * 2 ** 14 + 0.2)
    rp.get_curv(log='ruido info')

    ch1_val = np.mean(rp.data[-1][2]['ch1'])
    ch2_val = np.mean(rp.data[-1][2]['ch2'])

    real_val = 1.026

    ch1_act = (ch1_val + rp.calib_params['FE_CH1_DC_offs']) * float(
        rp.calib_params['FE_CH1_FS_G_HI']) / 2 ** 32 * 100 / 8192
    ch2_act = (ch2_val + rp.calib_params['FE_CH2_DC_offs']) * float(
        rp.calib_params['FE_CH2_FS_G_HI']) / 2 ** 32 * 100 / 8192

    rp.calib_params['FE_CH1_FS_G_HI'] = int(rp.calib_params['FE_CH1_FS_G_HI'] / ch1_act * real_val)
    rp.calib_params['FE_CH2_FS_G_HI'] = int(rp.calib_params['FE_CH2_FS_G_HI'] / ch2_act * real_val)

    rp.set_adc_dac_calib()


def capture_data(rp=None, t=1, save_csv=False, save_plt=False, signals=''):
    """
    options for signals, input them as a string separated with spaces
    ['oscA_sw', 'oscB_sw', 'osc_ctrl', 'trig_sw', 'out1_sw', 'out2_sw', 'slow_out1_sw', 'slow_out2_sw', 'slow_out3_sw',
    'slow_out4_sw', 'lock_control', 'lock_feedback', 'lock_trig_val', 'lock_trig_time', 'lock_trig_sw', 'rl_error_threshold',
    'rl_signal_sw', 'rl_signal_threshold', 'rl_config', 'rl_state', 'sf_jumpA', 'sf_jumpB', 'sf_config', 'signal_sw',
    'signal_i', 'sg_amp1', 'sg_amp2', 'sg_amp3', 'sg_amp_sq', 'lpf_F1', 'lpf_F2', 'lpf_F3', 'lpf_sq', 'error_sw',
    'error_offset', 'error', 'error_mean', 'error_std', 'gen_mod_phase', 'gen_mod_phase_sq', 'gen_mod_hp', 'gen_mod_sqp',
    'ramp_A', 'ramp_B', 'ramp_step', 'ramp_low_lim', 'ramp_hig_lim', 'ramp_reset', 'ramp_enable', 'ramp_direction',
    'ramp_B_factor', 'sin_ref', 'cos_ref', 'cos_1f', 'cos_2f', 'cos_3f', 'sq_ref_b', 'sq_quad_b', 'sq_phas_b', 'sq_ref',
    'sq_quad', 'sq_phas', 'in1', 'in2', 'out1', 'out2', 'slow_out1', 'slow_out2', 'slow_out3', 'slow_out4', 'oscA', 'oscB',
    'X_28', 'Y_28', 'F1_28', 'F2_28', 'F3_28', 'sqX_28', 'sqY_28', 'sqF_28', 'cnt_clk', 'cnt_clk2', 'read_ctrl', 'pidA_sw',
    'pidA_PSR', 'pidA_ISR', 'pidA_DSR', 'pidA_SAT', 'pidA_sp', 'pidA_kp', 'pidA_ki', 'pidA_kd', 'pidA_in', 'pidA_out',
    'pidA_ctrl', 'ctrl_A', 'pidB_sw', 'pidB_PSR', 'pidB_ISR', 'pidB_DSR', 'pidB_SAT', 'pidB_sp', 'pidB_kp', 'pidB_ki', 'pidB_kd',
    'pidB_in', 'pidB_out', 'pidB_ctrl', 'ctrl_B', 'aux_A', 'aux_B']
    """
    rp.start_streaming(signals=signals)
    time.sleep(t)
    rp.stop_streaming()
    dump_path = rp.file.name if hasattr(rp, 'file') else os.path.join(os.getcwd(), '20251014_110047_dump.bin')
    d = read_dump(filename=dump_path)
    d.plotr(start=0, end=-1, signals=signals)

    if save_plt:
        plt.savefig(save_plt)
    else:
        plt.show()

if __name__ == '__main__':
    rp_app = rp_connect()
    calib_dac_adc(rp=rp_app, dec=decimation)
    capture_data(rp=rp_app, signals='oscA oscB')

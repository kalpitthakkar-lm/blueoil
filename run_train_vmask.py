from blueoil.cmds import train

if __name__ == '__main__':
    config_file = 'blueoil/configs/core/object_detection/hikariming_160_lmfyolo.py'
    # config_file = 'blueoil/configs/core/object_detection/intelaws_160_lastconv_fp32.py'
    exp_id, ckpt_name = train.train(config_file)

from blueoil.cmds import train

if __name__ == '__main__':
    config_file = 'blueoil/configs/core/object_detection/wf_160_1x1_exp10_alt_v2.py'
    exp_id, ckpt_name = train.train(config_file)

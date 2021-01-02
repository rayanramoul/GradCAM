from train_gradcam import *

"""
This is a script to evaluate performance of already generated gradcam results with choosen tresholds.
"""

if __name__ == '__main__':
    thresholds = [0.01, 0.02, 0.06]
    log = quick_log_setup(logging.INFO)
    for tr in thresholds:
        load_eval_gradcam(threshold=tr)
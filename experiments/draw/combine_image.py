import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


images = [
    os.path.join('learning_rate', f'lr_schedule_comparison_{t}.png')
    for t in ['train_iters', 'test_iters', 'train_runtime', 'test_runtime']
]
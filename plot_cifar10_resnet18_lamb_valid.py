import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme(style="darkgrid")

def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def smoothing(array, width):
    length = len(array)
    output = np.zeros([length], dtype=float)

    ind_begin = 0
    for i in range(length):
        ind_end = i + 1
        if ind_end > width:
            ind_begin = ind_end - width
        output[i] = array[ind_begin:ind_end].mean()
    return output

ROOT = 'csv_data'
rolling_step = 1
smooth_rate = 0.88

csgd512_lr001_path = os.path.join(ROOT, 'CIFAR10_Mar21_12_13_54_jupyter-chenkaixuan_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.01-0.0-0.1-0.0-60-6000-6000-666-True.csv')
csgd512_lr002_path = os.path.join(ROOT, 'CIFAR10_Mar21_12_14_27_jupyter-chenkaixuan_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.02-0.0-0.1-0.0-60-6000-6000-666-True.csv')
dsgd512_lr001_path = os.path.join(ROOT, 'CIFAR10_Mar21_12_16_46_jupyter-chenkaixuan_CIFAR10s56-512-ring-fixed-16-ResNet18_M-1-0.01-0.0-0.1-0.0-60-6000-6000-666-True.csv')
dsgd512_lr002_path = os.path.join(ROOT, 'CIFAR10_Mar21_12_17_12_jupyter-chenkaixuan_CIFAR10s56-512-ring-fixed-16-ResNet18_M-1-0.02-0.0-0.1-0.0-60-6000-6000-666-True.csv')

csgd64_lr001_path = os.path.join(ROOT, 'CIFAR10_Mar22_05_56_23_jupyter-guest00_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.0035-0.0-0.1-0.0-60-6000-6000-666-True.csv')
csgd64_lr002_path = os.path.join(ROOT, 'CIFAR10_Mar22_05_57_23_jupyter-guest00_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.007-0.0-0.1-0.0-60-6000-6000-666-True.csv')
dsgd64_lr001_path = os.path.join(ROOT, 'CIFAR10_Mar22_06_00_58_jupyter-guest00_CIFAR10s56-64-ring-random-16-ResNet18_M-1-0.0035-0.0-0.1-0.0-0-6000-6000-666-True.csv')
dsgd64_lr002_path = os.path.join(ROOT, 'CIFAR10_Mar22_06_02_19_jupyter-guest00_CIFAR10s56-64-ring-random-16-ResNet18_M-1-0.007-0.0-0.1-0.0-0-6000-6000-666-True.csv')

dsgd64_lr001_data = pd.read_csv(dsgd64_lr001_path)
x_data = list(dsgd64_lr001_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd64_lr001_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-LAMB_1024(lr=0.0035)')

dsgd512_lr001_data = pd.read_csv(dsgd512_lr001_path)
x_data = list(dsgd512_lr001_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd512_lr001_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-LAMB_8192(lr=0.01)')


csgd64_lr001_data = pd.read_csv(csgd64_lr001_path)
x_data = list(csgd64_lr001_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd64_lr001_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-LAMB_1024(lr=0.0035)')

csgd512_lr001_data = pd.read_csv(csgd512_lr001_path)
x_data = list(csgd512_lr001_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd512_lr001_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-LAMB_8192(lr=0.01)')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.68,0.96599)
plt.legend(loc='lower right', fontsize=18, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'fig_cifar10_resnet18_lamb_01.pdf', format='pdf', bbox_inches='tight')
plt.close()

rolling_step = 1
smooth_rate = 0.88

dsgd64_lr002_data = pd.read_csv(dsgd64_lr002_path)
x_data = list(dsgd64_lr002_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd64_lr002_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-LAMB_1024(lr=0.007)')

dsgd512_lr002_data = pd.read_csv(dsgd512_lr002_path)
x_data = list(dsgd512_lr002_data.loc[:,'Step'])
y_data = smooth(list(np.array(dsgd512_lr002_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-LAMB_8192(lr=0.02)')

csgd64_lr002_data = pd.read_csv(csgd64_lr002_path)
x_data = list(csgd64_lr002_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd64_lr002_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-LAMB_1024(lr=0.007)')

csgd512_lr002_data = pd.read_csv(csgd512_lr002_path)
x_data = list(csgd512_lr002_data.loc[:,'Step'])
y_data = smooth(list(np.array(csgd512_lr002_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-LAMB_8192(lr=0.02)')

plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.68,0.96599)
plt.legend(loc='lower right', fontsize=18, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'fig_cifar10_resnet18_lamb_02.pdf', format='pdf', bbox_inches='tight')
plt.close()


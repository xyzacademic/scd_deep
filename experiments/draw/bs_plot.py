import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np






def draw_acc_vs_iter(lrc, lrf, key, filename=None):
    plt.figure(dpi=200)
    for lc in lrc:
        for lf in lrf:
            file_name = f'{dataset}_{version}_{act}_i{local_it}_{loss}_' \
                        f'b{batch_size}_lrc{lc}_lrf{lf}_nb{nb}_nw{nw}_' \
                        f'dm{dm}_upc{upc}_upf{upf}_ucf{ucf}_{precision}_{seed}.csv'

            if os.path.exists(os.path.join(record_path, file_name)):
                df = pd.read_csv(os.path.join(record_path, file_name))
                df = df.iloc[np.arange(0, df.shape[0], intercept)]
                if key == 'train':
                    plt.plot(df['epoch'].values, df['train acc'].values, linestyle='-',
                             label=f'conv_lr {lc} fc_lr {lf} train')
                elif key == 'test':
                    plt.plot(df['epoch'].values, df['test acc'].values, linestyle='--',
                             label=f'conv_lr {lc} fc_lr {lf} test')

            else:
                continue
    plt.title(f'Accuracy VS Iterations ({version})')
    plt.legend(loc='best')
    plt.xlabel('iters')
    plt.ylabel('Accuracy')
    plt.plot()
    if filename:
        plt.savefig(os.path.join(save_path, filename))
    plt.show()

    plt.close()

def draw_slope_vs_iter(lrc, lrf, key, filename=None):
    plt.figure(dpi=200)
    for lc in lrc:
        for lf in lrf:
            file_name = f'{dataset}_{version}_{act}_i{local_it}_{loss}_' \
                        f'b{batch_size}_lrc{lc}_lrf{lf}_nb{nb}_nw{nw}_' \
                        f'dm{dm}_upc{upc}_upf{upf}_ucf{ucf}_{precision}_{seed}.csv'

            if os.path.exists(os.path.join(record_path, file_name)):
                df = pd.read_csv(os.path.join(record_path, file_name))
                df = df.iloc[np.arange(0, df.shape[0], intercept)]

                acc = df[f'{key} acc'].values
                epoch = df['epoch'].values
                slope = np.gradient(acc, epoch)
                slope[slope < 0] = np.nan
                slope = pd.Series(slope).interpolate().values
                if key == 'train':
                    plt.plot(epoch, slope, linestyle='-',
                             label=f'conv_lr {lc} fc_lr {lf} {key}')
                elif key == 'test':
                    plt.plot(epoch, slope, linestyle='--',
                             label=f'conv_lr {lc} fc_lr {lf} {key}')

    plt.title('Slope VS Iterations')
    plt.legend(loc='best')
    plt.xlabel('iters')
    plt.ylabel('Slope')
    plt.plot()
    if filename:
        plt.savefig(os.path.join(save_path, filename))
    plt.show()
    plt.close()

def draw_best_vs_iter(lrc, lrf, bc, key, iters=True, filename=None):
    plt.figure(dpi=200)

    nrows = [500, 1000, 2500, 5000, 7500]
    phase_time = [
        (0.112, 0.069, 0.038),  # 500
        (0.178, 0.1, 0.055),  # 1000
        (0.4, 0.225, 0.1),  # 1500
        (0.85, 0.42, 0.18),  # 5000
        (1.17, 0.62, 0.26),  # 7500
    ]
    for j, batch_size in enumerate(nrows):
        phase_fixed = [
            f'cifar10_toy3srr100_sign_i1_mce_b{batch_size}_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_2.csv',
            f'cifar10_toy3ssr100_sign_i1_mce_b{batch_size}_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_2.csv',
            f'cifar10_toy3sss100_sign_i1_mce_b{batch_size}_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_2.csv',

        ]

        acc = []
        epoch = []
        run_time = []
        reserve = 0
        for i, file in enumerate(phase_fixed):
            if os.path.exists(os.path.join(record_path, file)):
                df = pd.read_csv(os.path.join(record_path, file))
                df = df.iloc[np.arange(0, df.shape[0], intercept)]
                acc.append(df[f'{key} acc'].values)
                epoch.append(df['epoch'].values + 15000 * i)
                run_time.append(df['epoch'].values * phase_time[j][i] + reserve)
                try:
                    reserve += 15000 * phase_time[j][i]
                except:
                    pass
        acc = np.concatenate(acc)
        epoch = np.concatenate(epoch)
        run_time = np.concatenate(run_time)
        if key == 'train':
            line_style = '-'
        elif key == 'test':
            line_style = '--'
        if iters:
            plt.plot(epoch, acc, linestyle=line_style, label=f'{key} accuracy batch size of {batch_size}')
        else:
            plt.plot(run_time, acc, linestyle=line_style, label=f'{key} accuracy batch size of {batch_size}')


    nrows = [1000]
    phase_time = [
        (0.112, 0.069, 0.038),  # 500
        (0.178, 0.1, 0.055),  # 1000
        (0.4, 0.225, 0.1),  # 1500
        (0.85, 0.42, 0.18),  # 5000
        (1.17, 0.62, 0.26),  # 7500
    ]
    for j, batch_size in enumerate(nrows):
        phase_fixed = [
            f'cifar10_toy3srr100_sign_i1_mce_b{batch_size}_lrc0.075_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_2.csv',
            f'cifar10_toy3ssr100_sign_i1_mce_b{batch_size}_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_2.csv',
            f'cifar10_toy3sss100_sign_i1_mce_b{batch_size}_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_fp16_2.csv',

        ]

        acc = []
        epoch = []
        run_time = []
        reserve = 0
        for i, file in enumerate(phase_fixed):
            if os.path.exists(os.path.join(record_path, file)):
                df = pd.read_csv(os.path.join(record_path, file))
                df = df.iloc[np.arange(0, df.shape[0], intercept)]
                acc.append(df[f'{key} acc'].values)
                epoch.append(df['epoch'].values + 15000 * i)
                run_time.append(df['epoch'].values * phase_time[j][i] + reserve)
                try:
                    reserve += 15000 * phase_time[j][i]
                except:
                    pass
        acc = np.concatenate(acc)
        epoch = np.concatenate(epoch)
        run_time = np.concatenate(run_time)
        if key == 'train':
            line_style = '-'
        elif key == 'test':
            line_style = '--'
        if iters:
            plt.plot(epoch, acc, linestyle=line_style, label=f'{key} accuracy batch size of {batch_size}')
        else:
            plt.plot(run_time, acc, linestyle=line_style, label=f'{key} accuracy batch size of {batch_size}')


    if iters:
        plt.title('Accuracy VS Iterations')
        plt.xlabel('iters')
    else:
        plt.title('Accuracy VS Runtime')
        plt.xlabel('seconds')

    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.plot()
    if filename:
        plt.savefig(os.path.join(save_path, filename))
    plt.show()
    plt.close()



if __name__ == '__main__':

    dataset = 'cifar10'
    version = 'toy3srr100'
    batch_size = 1500
    local_it = 1
    loss = 'mce'
    nb = 2
    nw = 0
    dm = 0
    upc = 1
    upf = 1
    ucf = 32
    precision = 'fp16'
    act = 'sign'
    seed = 0
    save_path = 'batch_size'
    intercept = 25
    # 1500
    phase_time = [0.247, 0.127, 0.062]


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    record_path = f'../experiments/logs/{dataset}'

    lrc = [0.05]
    lrf = [0.05, 0.1, 0.2]

    blrc = [0.075, 0.3, 0.05, 0.3, 0.05, 0.3]
    blrf = [0.1, 0.2, 0.05, 0.2, 0.05, 0.2]
    vs = ['toy3srr100', 'toy3srr100', 'toy3ssr100', 'toy3ssr100', 'toy3sss100', 'toy3sss100']

    # blrc = [0.3] * 3
    # blrf = [0.2] * 3
    # vs = ['toy3srr100', 'toy3ssr100', 'toy3sss100']
    # draw_acc_vs_iter(lrc, lrf, 'train', 'phase_1_train_fixed_conv_lr')
    # draw_acc_vs_iter(lrc, lrf, 'test', 'phase_1_test_fixed_conv_lr')
    # draw_slope_vs_iter(lrc, lrf, 'train')
    # draw_slope_vs_iter(lrc, lrf, 'test')
    draw_best_vs_iter(blrc, blrf, vs, 'train', True, 'batch_size_comparison_train_iters')
    draw_best_vs_iter(blrc, blrf, vs, 'test', True, 'batch_size_comparison_test_iters')
    draw_best_vs_iter(blrc, blrf, vs, 'train', False, 'batch_size_comparison_train_runtime')
    draw_best_vs_iter(blrc, blrf, vs, 'test', False, 'batch_size_comparison_test_runtime')


import sys



data = []

data.append('#!/bin/sh\n\n')
data.append('#$-q datasci\n')
data.append('#$-q ddlab_test\n')
data.append('#$-cwd\n')

foid = int(sys.argv[1])

data.append('#$-N part%d\n\n\n'%foid)
data.append('cd ..\n')

batch_size = 100
n_features = 2500
gpu = 0 if foid < 6 else 1




fid = 0

for kernel_size in [3, 5]:
    for layers in [2, 3]:
        seed = int('%d%d%d%d' % (kernel_size, layers, foid, fid))
        command = 'python stack_features.py ' \
                  '--kernel-size %d ' \
                  '--layers %d ' \
                  '--folder-id %d ' \
                  '--fid %d ' \
                  '--num-features %d ' \
                  '--batch-size %d ' \
                  '--seed %d ' \
                  '--gpu %d \n' % (kernel_size, layers, foid, fid, n_features,
                                   batch_size, seed, gpu)
        data.append(command)
        fid += 1


with open('job%d.sh'%foid, 'w') as f:
    f.writelines(data)


import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


import re

def smooth_y(array_y,stride):
	array_y = [sum(array_y[i:i+stride+1])/stride for i in range((len(array_y)-stride+1))]
	return array_y


def print_loss_fig(iterations, D_real_loss, D_fake_loss, C1_fake_loss, C2_fake_loss):
    # Data for plotting

    # f = interpolate.interp1d(iterations, D_real_loss, kind='slinear')
    # new_iterations = np.linspace(0, len(iterations), 1*len(iterations))
    # new_D_real_loss = f(new_iterations)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # new_D_real_loss = smooth_y(D_real_loss,20)
    lns1 = ax.plot(iterations[0:len(D_fake_loss):5], D_real_loss[0:len(D_fake_loss):5] , 'r', label='D_real_loss', linewidth=1.2, color ='green',alpha=0.5)
    # plt.scatter(iterations[0:len(D_fake_loss):100], D_real_loss[0:len(D_fake_loss):100], marker='o', color ='green')
    lns2 = ax.plot(iterations[0:len(D_fake_loss):5], D_fake_loss[0:len(D_fake_loss):5], label='D_fake_loss', linewidth=1.2, color ='black',alpha=0.7)


    lns3 =  ax.plot(iterations[:len(C1_fake_loss)], C1_fake_loss, 'r', label='C1_fake_loss', linewidth=1, color='blue' ,alpha=0.5)
    lns4 = ax.plot(iterations[:len(C2_fake_loss)], C2_fake_loss, 'r', label='C2_fake_loss', linewidth=1, color='red',alpha=0.8)
    # added these three lines

    lns = lns1 +lns2 +lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right',markerscale=50, prop={'family': 'Times New Roman', 'size': 12})

    # ax.grid()
    # plt.title('voice embedding network', fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_xlabel("epoch", fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_ylabel("loss", fontdict={'family': 'Times New Roman', 'size': 16})
    # ax2.set_ylabel(r"acc", fontdict={'family': 'Times New Roman', 'size': 12})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax2.set_ylim(0, 35)
    ax.set_ylim(-0.1,8)
    ax.set_xlim(0,42000)
    # plt.show()
    plt.savefig('D_C1+2_loss.png',dpi=300)

if __name__ == '__main__':

    with open("/home/fz/3PythonProject/FE-GAN-voice-to-face-v2.0/models/log/logFile_2020-06-24,22,59.log",'r') as f:
        D_real_loss = []
        D_fake_loss = []
        C1_fake_loss = []
        C2_fake_loss = []
        iterations = []
        lines = f.readlines()[1:]
        for line in lines:
            cuts = line.rstrip().split(',')
            iteration = re.findall("\d+", cuts[0])
            iterations.append(int(iteration[0]))
            D_real = re.findall("\d+.\d+", cuts[2])
            D_real_loss.append(float(D_real[0]))
            D_fake = re.findall("\d+.\d+", cuts[3])
            D_fake_loss.append(float(D_fake[0]))
            C1_fake = re.findall("\d+.\d+", cuts[6])
            C1_fake_loss.append(float(C1_fake[0]))
            C2_fake = re.findall("\d+.\d+", cuts[7])
            C2_fake_loss.append(float(C2_fake[0]))
    print_loss_fig(iterations, D_real_loss, D_fake_loss, C1_fake_loss, C2_fake_loss)

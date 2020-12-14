import os
import glob
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from network import get_network
from utils import rm_sil, get_mffc
from util.config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from util.parse_dataset import csv_to_list
from tqdm import tqdm

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def find_true_image(true_image_folder,gen_image_path):
    image_name = gen_image_path.split('/')[-1]
    Modality, Vocal, Emotion, intensity, Statement, Repetition, Actor = image_name[:-4].split('-')
    Actor_folder = "Actor_{}".format(Actor)
    true_image_paths = os.path.join(true_image_folder, Actor_folder, image_name[:-4], "*.png")
    true_image_paths = glob.glob(true_image_paths)
    true_image_path = true_image_paths[int(len(true_image_paths) / 2)]
    return true_image_path


def voice2feature(voice_file, vad_obj, mfc_obj):

    vad_voice = rm_sil(voice_file, vad_obj)
    voice_feat = get_mffc(vad_voice, mfc_obj)
    # voice_feat = get_fbank(vad_voice, mfc_obj)  #
    # vad_voice1 = rm_sil_librosa(voice_file, vad_obj)
    # voice_feat = get_spectrogram(vad_voice1)
    voice_feat = voice_feat.T[np.newaxis, ...]
    voice_feat = torch.from_numpy(voice_feat.astype('float32'))
    return voice_feat

def feautre2face(e_net, g_net, voice_feat_pth, GPU=True):
    voice_feat = np.load(voice_feat_pth)
    voice_feat = voice_feat[:999, :]   # 手工设置输入音频长度
    voice_feat = voice_feat.T[np.newaxis, ...]
    voice_feat = torch.from_numpy(voice_feat.astype('float32'))
    flag = voice_feat_pth.split('/')[-1][:-4].split('-')
    voice_emotion_label = torch.Tensor([int(flag[2])-1]).type(torch.LongTensor)
    # voice_EM_label_G = torch.zeros((DATASET_PARAMETERS['batch_size'], emotion_class_num)).scatter_(1, voice_emotion_label.unsqueeze(1), 1)

    voice_EM_label_G = torch.zeros((1, 8)).scatter_(1, voice_emotion_label.unsqueeze(1), 1)

    # voice_emotion_label = torch.LongTensor(voice_emotion_label)
    if GPU:
        voice_feat = voice_feat.cuda()
        voice_emotion_label = voice_emotion_label.cuda()
        voice_EM_label_G= voice_EM_label_G.cuda()
    embedding = e_net(voice_feat)
    embedding = F.normalize(embedding)
    embedding = embedding    #  压缩维度从64,128,1,1 --> 64,128
    face = g_net(embedding)     # G条件输入
    return face


if __name__ == '__main__':
    generator_folder_pth = "./models/generator"
    generator_list = []
    for root, dirs, filenames in os.walk(generator_folder_pth):
        filenames.sort()
        for filename in filenames:
            filename = os.path.join(generator_folder_pth,filename)
            generator_list.append(filename)

    for generator_file_pth in generator_list[187:]:
        voice_list, face_list, id_class_num, emotion_class_num = csv_to_list(DATASET_PARAMETERS)
        NETWORKS_PARAMETERS['e']['output_channel'] = id_class_num
        NETWORKS_PARAMETERS['g']['input_channel'][1] = emotion_class_num
        NETWORKS_PARAMETERS['g']['model_path'] = generator_file_pth
        print(NETWORKS_PARAMETERS['g']['model_path'])
        e_net, _ = get_network('e', NETWORKS_PARAMETERS, train=False, test=True)
        g_net, _ = get_network('g', NETWORKS_PARAMETERS, train=False, test=True)

        gen_image_folder = "./datasets/RAVDESS/"+ generator_file_pth.split('/')[-1]

        print(gen_image_folder)
        true_image_folder = "./datasets/RAVDESS/1 image-Actor1-24-single"
        new_true_image_folder = "/home/fz/3PythonProject/0-GAN-model/0-pytorch-FID+IS/RAVDESS_dataset_ori"
        if not os.path.exists(new_true_image_folder):
            os.makedirs(new_true_image_folder)
        if not os.path.exists(gen_image_folder):
            os.makedirs(gen_image_folder)
        for items in tqdm(voice_list):      # 根目录, 子目录, 文件名
            voice_feat_pth = items['filepath']
            filename = voice_feat_pth.split('/')[-1]
            gen_image_name = filename.replace('.npy', '.png')
            true_image_path = find_true_image(true_image_folder, gen_image_name)
            true_image_data = Image.open(true_image_path).convert('RGB').resize([128, 128],Image.ANTIALIAS)
            # face_data = ((face_data - 127.5) / 127.5).astype('float32')
            new_true_image_pth = os.path.join(new_true_image_folder, gen_image_name.split('/')[-1])
            # true_image_data.save(new_true_image_pth)       # 从视频序列中提取single单张图像
            # shutil.copyfile(true_image_path, new_true_image_pth)

            # pass
            gen_image_pth = os.path.join(gen_image_folder, gen_image_name)
            face_image = feautre2face(e_net, g_net, voice_feat_pth,
                                    NETWORKS_PARAMETERS['GPU'])
            vutils.save_image(face_image.detach().clamp(-1,1),
                              gen_image_pth, normalize=True)


import os
import csv
import glob
import shutil


def find_true_image(true_image_folder, voice_fea_name):
    image_name = voice_fea_name.split('.')[-2]
    Modality, Vocal, Emotion, intensity, Statement, Repetition, Actor = image_name.split('-')
    Actor_folder = "Actor_{}".format(Actor)
    true_image_paths = os.path.join(true_image_folder, Actor_folder, image_name, "*.png")
    true_image_paths = glob.glob(true_image_paths)
    true_image_path = true_image_paths[int(len(true_image_paths) / 2)]
    return true_image_path




def get_RAVDESS_csv(voice_data_pth, image_data_pth, csv_pth, data_ext):
    """
    从音频特征或图像文件夹中读取对应文件, 在csv中写入该文件路径,情感,身份,性别标签
    :param image_data_pth: 图像文件抽取中间为代表图像
    :param csv_pth: csv文件输出位置
    :param data_ext: .npy或者.png格式
    :return:
    """
    data_list = []
    # new_image_folder = "./datasets/RAVDESS/1 image-Actor1-24-single"

    list_name ={"wav":"wave", "png":"image", "mfcc":"mfcc", "fbank":"fbank", "spectrogram":"spectrogram"}
    file_path = list_name[data_ext]
    headers = ['actor_ID','gender','vocal_channel','emotion','emotion_intensity','mfcc_path', 'image_path']
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for root, dirs, filenames in os.walk(voice_data_pth):      # 根目录, 子目录, 文件名
        filenames.sort()
        for filename in filenames:
            if filename.endswith("npy"):              # 校验文件后缀名, wav或者npy
                voice_feat_path = os.path.join(root, filename)
                flag = filename[:-4].split('-')
                if flag[0] == '01':
                    true_image_path = find_true_image(image_data_pth, filename)
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_ID':flag[6], 'gender':gend,'vocal_channel':flag[1],'emotion':flag[2],
                                      'emotion_intensity':flag[3], 'mfcc_path': voice_feat_path, 'image_path': true_image_path})
                    print("voice_feat_path:{0}, true_image_path:{1},actor:{2}".format(voice_feat_path,true_image_path, flag[6]))

                    # 从视频序列图像中找出唯一代表图像
                    # new_image_pth = os.path.join(new_image_folder, true_image_path.split('/')[-3], true_image_path.split('/')[-2])
                    # if not os.path.exists(new_image_pth):
                    #     os.makedirs(new_image_pth)
                    # new_true_image_pth = os.path.join(new_image_pth, true_image_path.split('/')[-1])
                    # shutil.copyfile(true_image_path, new_true_image_pth)


    print("number:{}".format(len(data_list)))
    csv_pth = os.path.join(csv_pth, '{}_image_list.csv'.format(list_name[data_ext]))
    with open(csv_pth,'w',newline='') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)


def get_labels(voice_list, face_list):
    """
    合并voice和face中的同类项目
    :param voice_list:
    :param face_list:
    :return:
    """
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names

    # temp = []
    # for item in voice_list:
    #     if item['name'] in names:     # 查询list names中是否包含item name
    #         temp.append(item)
    # voice_list = temp

    #  通过列表推导式 保留同类项
    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = list(names)
    label_dict = dict(zip(names, range(len(names))))
    for item in voice_list+face_list:                # 增加序号label_id
        item['label_id'] = label_dict[item['name']]
    return voice_list, face_list, len(names)
    

def csv_to_list(data_params):
    """
    从list.csv中读取路径, 写入list中,
    :param data_params:
    :return: 数据路径以及标签,speaker数量
    """
    voice_list = []
    face_list = []
    actor_num = []
    emotion_num = []
    with open(data_params['wave_image_file']) as f:
        lines = f.readlines()[1:]
        for line in lines:
            # print(line)
            actor_ID,gender,vocal_channel,emotion,emotion_intensity,wave_path, image_path = line.rstrip("\n").split(',')
            actor_num.append(int(actor_ID))
            emotion_num.append(int(emotion))
            voice_list.append({'filepath': wave_path, 'name_id': actor_ID, 'emotion_id': emotion})
            face_list.append({'filepath': image_path, 'name_id': actor_ID, 'emotion_id': emotion})

    return voice_list, face_list, max(actor_num), max(emotion_num)

if __name__ == '__main__':
    # get_RAVDESS_dataset(DATASET_PARAMETERS)
    # data_dir = 'data/RAVDESS/fbank'

    voice_data_pth = './datasets/RAVDESS/2 fbank-Actor1-24-32k'
    image_data_pth = './datasets/RAVDESS/1 image-Actor1-24-single'
    csv_pth = "./datasets/RAVDESS"
    get_RAVDESS_csv(voice_data_pth, image_data_pth, csv_pth, 'fbank')



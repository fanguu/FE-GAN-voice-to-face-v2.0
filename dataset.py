import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# 表示torch中Dataset的抽象类


def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    voice_identity_label = int(voice_item['name_id'])-1
    voice_emotion_label = int(voice_item['emotion_id'])-1
    return voice_data, voice_identity_label, voice_emotion_label


def load_face(face_item):
    face_data = Image.open(face_item['filepath']).convert('RGB').resize([128, 128],Image.ANTIALIAS)
    face_data = np.transpose(np.array(face_data), (2, 0, 1))
    # face_data = ((face_data / 255)).astype('float32')               # 0,1归一化
    face_data = ((face_data - 127.5) / 127.5).astype('float32')   # -1,1归一化
    face_identity_label = int(face_item['name_id'])-1
    face_emotion_label = int(face_item['emotion_id'])-1
    return face_data, face_identity_label, face_emotion_label

class Voice_Face_Dataset(Dataset):
    def __init__(self,voice_list, face_list, nframe_range):
        self.face_list = face_list
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1] #[300, 800]

    def __getitem__(self, index):
        voice_data, voice_identity_label, voice_emotion_label = load_voice(self.voice_list[index])
        assert self.crop_nframe <= voice_data.shape[1]
        pt = np.random.randint(voice_data.shape[1] - self.crop_nframe + 1)
        voice_data = voice_data[:, pt:pt+self.crop_nframe]

        face_data, face_identity_label, face_emotion_label = load_face(self.face_list[index])
        if np.random.random() > 0.5:
           face_data = np.flip(face_data, axis=2).copy()
        # input_dict = {'voice': voice_data, 'image': face_data, 'emotion_label': voice_identity_label, 'identity_label': voice_emotion_label}
        return voice_data, face_data, voice_identity_label, voice_emotion_label

    def __len__(self):
        return len(self.voice_list)

class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1] #[300, 800]

    def __getitem__(self, index):
        voice_data, voice_identity_label, voice_emotion_label = load_voice(self.voice_list[index])
        assert self.crop_nframe <= voice_data.shape[1]
        pt = np.random.randint(voice_data.shape[1] - self.crop_nframe + 1)
        voice_data = voice_data[:, pt:pt+self.crop_nframe]
        return voice_data, voice_identity_label, voice_emotion_label

    def __len__(self):
        return len(self.voice_list)


class FaceDataset(Dataset):
    def __init__(self, face_list):
        self.face_list = face_list

    def __getitem__(self, index):
        face_data, face_identity_label, face_emotion_label = load_face(self.face_list[index])
        if np.random.random() > 0.5:
           face_data = np.flip(face_data, axis=2).copy()
        return face_data, face_identity_label, face_emotion_label

    def __len__(self):
        return len(self.face_list)

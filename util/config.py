import string
from dataset import VoiceDataset, FaceDataset
from network import VoiceEmbedNet, Generator, Condititon_D, FaceEmbedNet, Classifier
from utils import get_collate_fn
voice_features_len = 64
DATASET_PARAMETERS = {
    # RAVDESS datase csv
    # 'wave_file': 'datasets/RAVDESS/mfcc_list.csv',
    # 'image_file': 'datasets/RAVDESS/image_list.csv',
    'wave_image_file': 'datasets/RAVDESS/fbank_image_list.csv',

    # log model
    'log_dir': 'models/log',

    # train data includes the identities whose names start with the characters of 'FGH...XYZ'
    'split': string.ascii_uppercase[5:],   # 生成大写字母

    # dataloader
    'voice_dataset': VoiceDataset,
    'face_dataset': FaceDataset,
    'batch_size': 64,   #64
    'nframe_range': [300, 800],
    'workers_num': 8,
    'collate_fn': get_collate_fn,

    # test data
    'test_data': './datasets/RAVDESS/example_data/',
}


NETWORKS_PARAMETERS = {
    # VOICE EMBEDDING NETWORK (e)
    'e': {
        'network': VoiceEmbedNet,
        'input_channel': 64,
        'channels': [256, 384, 576, 864],
        'output_channel': 24, # the number of peoples
        'model_path': "./pretrained_models/12-13,22,10-2000-Fbank-cnn-64dim.pth",
        'save_model_path': 'models/voice_embedding/voice_embedding',

    },
    # GENERATOR (g)
    'g': {
        'network': Generator,
        'input_channel': [voice_features_len, 7],    # embedding feature and label class feature dim 默认8
        'channels': [1024, 512, 256, 128, 64, 32], # channels for deconvolutional layers[1024, 512, 512, 256, 128, 64]
        'output_channel': 3, # images with RGB channels

        'model_path': 'models/generator/07-23,05,56-200000-generator.pth',     # 预训练模型存储路径以及名称
        'save_model_path': 'models/generator/generator',
    },
    # DISCRIMINATOR (d)
    'd1-condition': {
        'network': Condititon_D,  # Discrminator ='F'+'C' is a special Classifier with 1 subject
        'input_channel': [3, 8],       # embedding feature and label class
        'channels': [32, 64, 128, 256, 512, 1024, 64],
        'output_channel': 1,
        'model_path': 'models/discriminator.pth',  # 无
        'save_model_path': 'models/discriminator/discriminator',
    },

    # FACE EMBEDDING NETWORK (f)
    'f': {
        'network': FaceEmbedNet,
        'input_channel': 3,
        'channels': [32, 64, 128, 256, 512, 1024, 64],
        'output_channel': -1, # This parameter is depended on the dataset we used
        'model_path': 'models/generator/06-21,14,55-36000-generator.pth', # 无
        'save_model_path': 'models/face_embedding/face_embedding',
    },
    'd0': {
        'network': Classifier,  # Discrminator ='F'+'C' is a special Classifier with 1 subject
        'input_channel': 64,  # embedding feature, default:64
        'channels': [],
        'output_channel': 1,
        'model_path': 'models/discriminator.pth',  # 无
        'save_model_path': 'models/discriminator/discriminator',
    },
    # CLASSIFIER (c)
    'c': {
        'network': Classifier,
        'input_channel': 64,  # default:64
        'channels': [],
        'output_channel': 24,  # This parameter is depended on the dataset we used
        'model_path': 'models/classifier/08,36-175000-classifier1.pth',  # 无
        'save_model_path': 'models/classifier/classifier',
    },
    # OPTIMIZER PARAMETERS 
    'lr': 0.0002,    #0.0002  1e-4
    'beta1': 0.5,
    'beta2': 0.999,

    # MODE, use GPU or not
    'GPU': True,

    # 中继训练
    'finetune': False,
}

if __name__ == '__main__':
    networks_parameters = NETWORKS_PARAMETERS
    net_structure = networks_parameters['e']
    print(net_structure)
    net = net_structure['network'](net_structure['input_channel'],
                                   net_structure['channels'],
                                   net_structure['output_channel'])
    print(net)

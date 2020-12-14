import time
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from util.parse_dataset import csv_to_list
from network import restore_train, get_network
from utils import Meter, cycle, save_model, get_collate_fn, Logger
from dataset import VoiceDataset, FaceDataset

# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num, emotion_class_num = csv_to_list(DATASET_PARAMETERS)
print('voice samples num = %d, face samples num = %d' %(len(voice_list),len(face_list)))
print('Preparing the datasets...')
voice_dataset = VoiceDataset(voice_list,DATASET_PARAMETERS['nframe_range'])
face_dataset = FaceDataset(face_list)

print('Preparing the dataloaders...')
collate_fn = get_collate_fn(DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],  # 使用多进程加载的进程数
                          collate_fn=collate_fn
                        )  # 如何将多个样本数据拼接成一个batch
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycle(face_loader))

print('Initializing networks...')
NETWORKS_PARAMETERS['e']['output_channel'] = id_class_num
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, test=True)   # 部分训练
NETWORKS_PARAMETERS['g']['input_channel'][1] = emotion_class_num
g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)
NETWORKS_PARAMETERS['d1-condition']['input_channel'][1] = emotion_class_num
d1_net, d1_optimizer = get_network('d0', NETWORKS_PARAMETERS, train=True)
d2_net, d2_optimizer = get_network('d0', NETWORKS_PARAMETERS, train=True)
f1_net, f1_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
f2_net, f2_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)

NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num
c1_net, c1_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)
NETWORKS_PARAMETERS['c']['output_channel'] = emotion_class_num
c2_net, c2_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)



# 接力训练,载入已有的模型
if NETWORKS_PARAMETERS['finetune']:
    restore_train(g_net, d1_net, f1_net, f2_net)

# label for real/fake faces
real_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 1)
fake_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 0)
D_loss_positive = torch.tensor(1, dtype=torch.float)
D_loss_negative = D_loss_positive * -1

#  Meters for recording the training status 日志模块 #
writer = SummaryWriter("./models/log")
logger = Logger(DATASET_PARAMETERS['log_dir'], time.strftime("%Y-%m-%d,%H,%M"))
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')
D_real = Meter('D_real', 'avg', ':4.3f')
D_fake = Meter('D_fake', 'avg', ':4.3f')
C1_real = Meter('C1_real', 'avg', ':4.3f')
C2_real= Meter('C2_real', 'avg', ':4.3f')
C1_fake = Meter('C1_fake', 'avg', ':4.3f')
C2_fake= Meter('C2_fake', 'avg', ':4.3f')
GD_fake = Meter('G_D_fake', 'avg', ':4.3f')


print('Training models...')
for it in range(90000+1):
    # data
    start_time = time.time()
    voice, voice_identity_label, voice_emotion_label = next(voice_iterator)
    face, face_identity_label, face_emotion_label = next(face_iterator)
    noise = 0.05*torch.randn(DATASET_PARAMETERS['batch_size'], 128, 1, 1)  # 标准正态分布

    # use GPU or not
    if NETWORKS_PARAMETERS['GPU']:
        voice, voice_identity_label, voice_emotion_label = voice.cuda(), voice_identity_label.cuda(), voice_emotion_label.cuda()
        face, face_identity_label, face_emotion_label = face.cuda(), face_identity_label.cuda(), face_emotion_label.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        noise = noise.cuda()
        D_loss_positive, D_loss_negative = D_loss_positive.cuda(), D_loss_negative.cuda()


    # get embeddings and generated faces
    embeddings = e_net(voice)
    embeddings = F.normalize(embeddings)
    # introduce some permutations
    embeddings = embeddings + noise
    embeddings = F.normalize(embeddings)
    embeddings = embeddings.squeeze()       #  压缩维度从64,128,1,1 --> 64,128

    # 扩展维度从64,1 --> 64, 8, 128, 128 , nn.Embedding(emotion_class_num,emotion_class_num)
    face_EM_label_ = torch.zeros((DATASET_PARAMETERS['batch_size'], emotion_class_num)).scatter_(1, face_emotion_label.type(torch.LongTensor).unsqueeze(1), 1)
    face_EM_label_ = face_EM_label_.unsqueeze(2).unsqueeze(3).expand(DATASET_PARAMETERS['batch_size'], emotion_class_num, face.size(2), face.size(3))
    face_EM_label_ = face_EM_label_.cuda()
    voice_EM_label_ = torch.zeros((DATASET_PARAMETERS['batch_size'], emotion_class_num)).scatter_(1, voice_emotion_label.type(torch.LongTensor).unsqueeze(1), 1)
    voice_EM_label_ = voice_EM_label_.unsqueeze(2).unsqueeze(3).expand(DATASET_PARAMETERS['batch_size'], emotion_class_num, face.size(2), face.size(3))
    voice_EM_label_ = voice_EM_label_.cuda()

    fake_face = g_net(embeddings, voice_emotion_label, voice_identity_label)     # G条件输入

    #### Discriminator#####
    d1_optimizer.zero_grad()
    d2_optimizer.zero_grad()
    f1_optimizer.zero_grad()
    f2_optimizer.zero_grad()
    c1_optimizer.zero_grad()
    c2_optimizer.zero_grad()

    #  ------- 真实real图像loss和得分--------------- #
    # real_score_out = d_net(face, face_EM_label_)     # D条件输入:face_EM_label_, 条件为 face情绪
    real_score_out_1 = d1_net(f1_net(face))           # D1无条件输入
    real_score_out_2 = d2_net(f2_net(face))           # D2无条件输入
    D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out_1), real_label) # BCE loss
    # real_score_out = 1 - real_score_out                   # hinge loss
    # D_real_loss = F.relu(real_score_out).mean()           # hinge loss

    real_id_label_out = c1_net(f1_net(face))             # 计算 c1,c2 loss
    real_emotion_label_out = c2_net(f2_net(face))
    C1_real_loss = F.nll_loss(F.log_softmax(real_id_label_out, dim=1), face_identity_label)
    C2_real_loss = F.nll_loss(F.log_softmax(real_emotion_label_out, dim=1), face_emotion_label)

    #  ------- 生成fake图像loss和得分--------------- #
    # 使两个G, D计算图的梯度传递断开,即轮流优化
    # fake_score_out = d_net(fake_face.detach(), voice_EM_label_)     #D条件输入:voice_EM_label_, 条件为voice情绪
    fake_score_out_1 = d1_net(f1_net(fake_face.detach()))     # D1无条件输入
    fake_score_out_2 = d2_net(f2_net(fake_face.detach()))    # D2无条件输入
    D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out_1), fake_label) # BCE loss
    # fake_score_out = 1 + fake_score_out                          # hinge loss
    # D_fake_loss = F.relu(fake_score_out).mean()                  # hinge loss
    # fake_id_label_out = f1_net(fake_face)
    # fake_emotion_label_out = f2_net(fake_face)
    # C1_fake_loss = F.nll_loss(F.log_softmax(fake_id_label_out, dim=1), voice_identity_label)
    # C2_fake_loss = F.nll_loss(F.log_softmax(fake_emotion_label_out, dim=1), voice_emotion_label)


    (D_fake_loss + D_real_loss +C1_real_loss+ C2_real_loss).backward()
    d1_optimizer.step()
    d2_optimizer.step()
    f1_optimizer.step()
    f2_optimizer.step()
    c1_optimizer.step()
    c2_optimizer.step()

    #   ---------------------------------------------
    D_real.update(D_real_loss.item())
    D_fake.update(D_fake_loss.item())
    C1_real.update(C1_real_loss.item())
    C2_real.update(C2_real_loss.item())
    #   ---------------------------------------------

    #### Generator #####
    g_optimizer.zero_grad()
    # fake_score_out = d_net(fake_face, voice_EM_label_)   #  D条件输入:voice_EM_label_, 条件为情绪
    fake_score_out_1 = d1_net(f1_net(fake_face))    # D无条件输入
    fake_score_out_2 = d2_net(f2_net(fake_face))
    fake_id_label_out = c1_net(f1_net(fake_face))
    fake_emotion_label_out = c2_net(f2_net(fake_face))
    GC1_fake_loss = F.nll_loss(F.log_softmax(fake_id_label_out, dim=1), voice_identity_label)    # 用真实标签替代随机标签？
    GC2_fake_loss = F.nll_loss(F.log_softmax(fake_emotion_label_out, dim=1), voice_emotion_label)
    GD_fake_loss = 1*F.binary_cross_entropy(torch.sigmoid(fake_score_out_1), real_label)    # BCE loss
    # GD_fake_loss = fake_score_out.mul(-1).mean()         # hing loss
    (GD_fake_loss + GC1_fake_loss + GC2_fake_loss).backward()
    g_optimizer.step()
    #   ---------------------------------------------
    GD_fake.update(GD_fake_loss.item())
    C1_fake.update(GC1_fake_loss.item())
    C2_fake.update(GC2_fake_loss.item())
    batch_time.update(time.time() - start_time)
    #   ---------------------------------------------

    # print status
    if it % 10 == 0:
        logger.info([iteration.__str__()  + batch_time.__str__() +
                             D_real.__str__() + D_fake.__str__() + C1_real.__str__() +C2_real.__str__()+C1_fake.__str__()+C2_fake.__str__()+
                             GD_fake.__str__()  ])

        writer.add_scalars('data/scalar_group', {"D_real": D_real_loss,
                                                 "D_fake": D_fake_loss,
                                                 "C1_real_loss":C1_real_loss,
                                                 "C2_real_loss":C2_real_loss,
                                                 "C1_fake_loss": GC1_fake_loss,
                                                 "C2_fake_loss": GC2_fake_loss,
                                                 "GD_fake_loss":GD_fake_loss}, it)

        # info = {'image/real_images': real_images(face, 8), 'image/generated_images': generate_img(fake_face, 8)}
        # writer.add_images('image/generated_images', generate_img(fake_face, 8), it)
        batch_time.reset()
        D_real.reset()
        D_fake.reset()
        C1_real.reset()
        C2_real.reset()
        C1_fake.reset()
        C2_fake.reset()
        GD_fake.reset()

        # snapshot
        if it % 2000 == 0:
            s_time = time.strftime("%m-%d,%H,%M") + '-' + str(it) + '-'
            # save_model(e_net, 'models/voice_embedding/{}voice_embedding.pth'.format(s_time))
            save_model(g_net, 'models/generator/{}generator.pth'.format(s_time))
            # save_model(d1_net, 'models/discriminator/{}discriminator.pth'.format(s_time))
            # save_model(f1_net, 'models/face_embedding/{}face_embedding1.pth'.format(s_time))
            # save_model(f2_net, 'models/face_embedding/{}face_embedding2.pth'.format(s_time))

    iteration.update(1)
# writer.export_scalars_to_json("./models/log/all_scalars.json")
# writer.close()



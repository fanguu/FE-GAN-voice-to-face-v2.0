import torch
from torch.autograd import Variable
from torch import autograd

def generate_img(samples, number_of_images):
    C = 3
    samples = samples.detach()
    samples = samples.cpu().numpy()[:number_of_images]
    # print(samples.shape)
    # print(type(samples))
    generated_images = []
    # for sample in samples:
    #     if C == 3:
    #         generated_images.append(sample)
    #     else:
    #         generated_images.append(sample.reshape(32, 32))
    # print(generated_images.shape)
    return samples


def real_images(images, number_of_images):
    C = 3
    if (C == 3):
        # a = to_np(images.view(-1, C, 128, 128)[:number_of_images])
        # print(a.shape)
        # print(type(a))
        return to_np(images.view(-1, C, 128, 128)[:number_of_images])


def to_np(x):
    return x.data.cpu().numpy()


def calculate_gradient_penalty(batch_size, cuda, real_images, fake_images, voice_EM_label_, d_net):
    lambda_term = 10
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))

    if cuda:
        eta = eta.cuda()
    else:
        eta = eta

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    if cuda:
        interpolated = interpolated.cuda()
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = d_net(interpolated, voice_EM_label_)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated,
                              inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).cuda() if cuda else torch.ones(prob_interpolated.size()),
                              create_graph=True,
                              retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty
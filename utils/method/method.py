import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple


def imgClapm(img, low_bound=-1, up_bound=1):
    if type(low_bound) == list:
        for i in range(3):
            img[:, i, :, :] = torch.clamp(
                img[:, i, :, :], low_bound[i], up_bound[i])
    else:
        img = torch.clamp(img, low_bound, up_bound)
    return img


def DummyAttack(data, target, white_models, loss, epsilon, device):
    return imgClapm(torch.rand_like(data).to(device))


def FGSM(data, target, white_models, device, epsilon, sign=1):
    perturbed_image = data.clone().detach().requires_grad_(True)
    loss = 0
    for model in white_models:
        model.zero_grad()
        output = model(perturbed_image)
        loss += output

    loss = F.cross_entropy(loss, target)

    # compute gradient
    loss.backward()
    data_grad = perturbed_image.grad.data.detach()

    # get sign
    sign_data_grad = data_grad.sign()

    # add perturbance
    t_perturbed_image = perturbed_image + epsilon*sign_data_grad*sign
    t_perturbed_image = imgClapm(t_perturbed_image)

    return t_perturbed_image


def PGD_linf_sum_logit(data, target, white_models, device, epsilon, step=10, sign=1):
    perturbed_image = data.clone().detach().requires_grad_(True)

    for _ in range(step):

        # computing Loss
        loss = 0
        for model in white_models:
            model.zero_grad()
            output = model(perturbed_image)
            loss += output

        loss = F.cross_entropy(loss, target)

        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()

        # get sign
        sign_data_grad = data_grad.sign()

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon/step*sign_data_grad*sign
        t_perturbed_image = imgClapm(t_perturbed_image)

        # stop if larget then the requirement

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_linf_sum_logit_momentum(data, target, white_models, device, epsilon, momentum=0.9, step=10, sign=1, image_size=299):
    perturbed_image = data.clone().detach().requires_grad_(True)
    # inital G
    g = 0
    for _ in range(step):

        # computing Loss
        loss = 0
        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                output = model(perturbed_image)
            loss += output
        loss = loss/len(white_models)
        loss = F.cross_entropy(loss, target)

        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()
        g = momentum*g+data_grad/data_grad.norm(1)
        g = g.detach()
        # get sign
        sign_data_grad = g.sign()

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon/step*sign_data_grad*sign
        t_perturbed_image = imgClapm(t_perturbed_image)

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_l2_sum_logit_momentum(data, target, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7):
    perturbed_image = data.clone().detach().requires_grad_(True)
    # inital G
    g = 0
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1

    
    for _ in range(step):

        # computing Loss
        loss = 0
        for model in white_models:
            model.zero_grad()
            output = model(perturbed_image)
            loss += output
        loss = F.cross_entropy(loss, target)

        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        g = momentum*g+data_grad/data_grad.norm(2)
        g = g.detach()
        # get sign

        sign_data_grad = g.sign()
        data_epsilon = alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
        data_epsilon = data_epsilon/data_epsilon.norm(2)

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon / \
            step*sign_data_grad*data_epsilon*sign
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
        # stop if larget then the requirement
        # if is_better(t_perturbed_image,data):
        # perturbed_image=t_perturbed_image
        # else:
        # break

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_l2_sum_logit_momentum_seg(data, target, mask_file_path, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7):

    perturbed_image = data.clone().detach().requires_grad_(True)

    mask = np.int32(np.load(mask_file_path)['arr_0'] != 0)
    mask = torch.tensor(mask, dtype=torch.float32)
    mask = torch.cat([mask.view(1, mask.shape[0], mask.shape[0])
                      for _ in range(3)])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(mode='RGB'),
        torchvision.transforms.Resize(data.shape[2]), torchvision.transforms.ToTensor()])
    mask = transform(mask).view(1, 3, data.shape[2], data.shape[2])
    mask_sum = mask.sum()
    if (mask_sum <= 0.3*torch.prod(torch.tensor(mask.shape, dtype=torch.float32))) | (mask_sum >= 0.7*torch.prod(torch.tensor(mask.shape, dtype=torch.float32))):
        mask = torch.rand_like(data)  # .to(device)
        mask.zero_()
        mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
             int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1

    mask = mask.to(device)

    # inital G
    g = 0

    for _ in range(step):

        # computing Loss
        loss = 0

        for model in white_models:
            model.zero_grad()
            output = model(perturbed_image)
            loss += output
        loss = F.cross_entropy(loss, target)

        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        g = momentum*g+data_grad/data_grad.norm(2)
        g = g.detach()
        # get sign

        sign_data_grad = g.sign()
        data_epsilon = alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
        data_epsilon = data_epsilon/data_epsilon.norm(2)

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon / \
            step*sign_data_grad*data_epsilon*sign
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
        # stop if larget then the requirement
        # if is_better(t_perturbed_image,data):
        # perturbed_image=t_perturbed_image
        # else:
        # break

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_l2_sum_logit_momentum_seg_mask(data, target, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7, image_size=299):

    perturbed_image = data.clone().detach().requires_grad_(True)
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    g = 0

    for _ in range(step):

        # computing Loss
        loss = 0

        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                output = model(perturbed_image)
            loss += output
        loss = loss/len(white_models)
        loss = F.cross_entropy(loss, target)
        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        g = momentum*g+data_grad/data_grad.norm(2)
        g = g.detach()
        # get sign

        sign_data_grad = g.sign()
        data_epsilon = alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
        data_epsilon = data_epsilon/data_epsilon.norm(2)

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon / \
            step*sign_data_grad*data_epsilon*sign
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
        # stop if larget then the requirement
        # if is_better(t_perturbed_image,data):
        # perturbed_image=t_perturbed_image
        # else:
        # break

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_l1_sum_logit_momentum_seg(data, target, mask_file_path, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7, image_size=299):

    perturbed_image = data.clone().detach().requires_grad_(True)

    mask = np.int32(np.load(mask_file_path)['arr_0'] != 0)
    mask = torch.tensor(mask, dtype=torch.float32)
    mask = torch.cat([mask.view(1, mask.shape[0], mask.shape[0])
                      for _ in range(3)])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(mode='RGB'),
        torchvision.transforms.Resize(data.shape[2]), torchvision.transforms.ToTensor()])
    mask = transform(mask).view(1, 3, data.shape[2], data.shape[2])
    mask_sum = mask.sum()
    if (mask_sum <= 0.3*torch.prod(torch.tensor(mask.shape, dtype=torch.float32))) | (mask_sum >= 0.7*torch.prod(torch.tensor(mask.shape, dtype=torch.float32))):
        mask = torch.rand_like(data)  # .to(device)
        mask.zero_()
        mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
             int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1

    mask = mask.to(device)

    # inital G
    g = 0

    for _ in range(step):

        # computing Loss
        loss = 0

        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                output = model(perturbed_image)
            loss += output
        loss = F.cross_entropy(loss, target)

        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        g = momentum*g+data_grad/data_grad.norm(2)
        g = g.detach()
        # get sign
        sign_data_grad = g.sign()

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon/step*sign_data_grad*sign
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_l2_sum_logit_momentum_seg_v2(data, target, mask_file_path, white_models, device, epsilon, momentum=0.9, step=10, sign=1, mask_ratio=0.7, image_size=299, alpha=0.3):

    perturbed_image = data.clone().detach().requires_grad_(True)

    mask = np.int32(np.load(mask_file_path)['arr_0'] != 0)
    mask = torch.tensor(mask, dtype=torch.float32)
    mask = torch.cat([mask.view(1, mask.shape[0], mask.shape[0])
                      for _ in range(3)])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(mode='RGB'),
        torchvision.transforms.Resize(data.shape[2]), torchvision.transforms.ToTensor()])
    mask = transform(mask).view(1, 3, data.shape[2], data.shape[2])
    mask_sum = mask.sum()
    if (mask_sum < alpha*torch.prod(torch.tensor(mask.shape, dtype=torch.float32))) | (mask_sum >= (1-alpha)*torch.prod(torch.tensor(mask.shape, dtype=torch.float32))):
        mask = torch.rand_like(data)  # .to(device)
        mask.zero_()
        mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
             int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1

    mask = mask.to(device)

    # inital G
    g = 0

    for _ in range(step):

        # computing Loss
        loss = 0
        # wrong_counter=0
        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                output = model(perturbed_image)
            # label=output.argmax()
            loss += output
            # if label!=target:
            # wrong_counter+=1
        # if  wrong_counter==len(white_models):
            # break
        loss = F.cross_entropy(loss, target)

        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        g = momentum*g+data_grad/data_grad.norm(2)
        g = g.detach()
        # get sign

        sign_data_grad = g.sign()
        data_epsilon = alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
        data_epsilon = data_epsilon/data_epsilon.norm(2)

        # add perturbance
        t_perturbed_image = perturbed_image + epsilon / \
            step*sign_data_grad*data_epsilon*sign
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def PGD_linf_sum_logit_l2_Adam(data, target, white_models, device, epsilon, step=10, beta1=0.9, beta2=0.999, epsilon_error=10e-8, alpha=0.001, sign=1):

    perturbed_image = data.clone().detach().requires_grad_(True)

    # inital G
    m = 0
    v = 0
    for t in range(step):
        loss = 0
        for model in white_models:
            model.zero_grad()
            output = torch.sigmoid(model(perturbed_image))
            loss += output

        loss = F.cross_entropy(loss, target)

        model.zero_grad()
        loss.backward()

        # Get Grad
        #beta1=0.9, beta2=0.999, epsilon_error=10e-8, alpha=0.001
        g = perturbed_image.grad.data
        m = beta1*m+(1-beta1)*g
        v = beta2*v+(1-beta2)*(g**2)
        m_hat = m/(1-beta1**(t+1))
        v_hat = v/(1-beta2**(t+1))
        y = alpha*m_hat/(v_hat+epsilon_error)
        sign_data_grad = y.sign()

        data_epsilon = alpha*(-1*(torch.sigmoid(y)-1/2).pow(2)+1)
        data_epsilon = data_epsilon/data_epsilon.norm(2)

        t_perturbed_image = perturbed_image + epsilon / \
            step*sign_data_grad*data_epsilon * sign()
        t_perturbed_image = imgClapm(t_perturbed_image)

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image


def CWOPT(data, target, white_models, device, epsilon, step=20, mask_ratio=0.7, image_size=299, c=15, lr=0.05):
    perturbed_image = data.clone().detach().requires_grad_(True)
    # inital G
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1

    # Init delta
    loss = 0
    for model in white_models:
        model.zero_grad()
        if model.input_size[1] != image_size:
            output = model(nn.functional.interpolate(
                perturbed_image, model.input_size[1]))
        else:
            output = model(perturbed_image)
        loss += output
    loss = loss/len(white_models)
    loss = F.cross_entropy(loss, target)
    loss.backward()
    delta = perturbed_image.grad.data.sign().detach()*epsilon
    delta = imgClapm(perturbed_image+delta)-perturbed_image
    delta = 0.5 * torch.log(1+delta+1e-5) - 0.5 * torch.log(1-delta+1e-5)
    delta = nn.Parameter(delta)

    # use Adam
    optimizer=torch.optim.Adam([delta],lr=lr)
    #optimizer = torch.optim.LBFGS([delta], lr=1)
    for _ in range(step):

        # computing Loss
        loss = 0
        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(imgClapm(nn.functional.interpolate(
                    perturbed_image+torch.tanh(delta), model.input_size[1])))
            else:
                output = model(imgClapm(perturbed_image+torch.tanh(delta)))
            loss += output

        loss = -torch.log(2*F.cross_entropy(loss, target)-2) * \
            c+(delta*delta).mean()
        loss.backward()
        optimizer.step()
        # delta=imgClapm(delta)
    return imgClapm(perturbed_image+torch.tanh(delta))


def PGD_l2_sum_img_momentum_mask(data, target, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7, image_size=299):
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    g = 0
    out_put_image = 0
    for model in white_models:
        perturbed_image = data.clone().detach().requires_grad_(True)
        for _ in range(step):

            # computing Loss
            model.zero_grad()
            if model.input_size[1] != image_size:
                loss = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                loss = model(perturbed_image)
            loss = F.cross_entropy(loss, target)
            # compute gradient
            loss.backward()
            data_grad = perturbed_image.grad.data.detach()*mask
            g = momentum*g+data_grad/data_grad.norm(2)
            g = g.detach()
            # get sign

            sign_data_grad = g.sign()
            data_epsilon = alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
            data_epsilon = data_epsilon/data_epsilon.norm(2)

            # add perturbance
            t_perturbed_image = perturbed_image + epsilon / \
                step*sign_data_grad*PGD_l2_sum_img_momentum_mask_v2data_epsilon*sign
            t_perturbed_image = imgClapm(t_perturbed_image)
            t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
            perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

        out_put_image += perturbed_image
    out_put_image = out_put_image/len(white_models)
    out_put_image = imgClapm(out_put_image)
    return out_put_image


def PGD_l2_sum_img_momentum_mask_v2(data, target, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7, image_size=299):
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    g = 0
    out_put_image = 0
    for model in white_models:
        perturbed_image = data.clone().detach().requires_grad_(True)
        for _ in range(step):

            # computing Loss
            model.zero_grad()
            if model.input_size[1] != image_size:
                loss = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                loss = model(perturbed_image)

            loss = F.cross_entropy(loss, loss.argmin().view(-1))
            # compute gradient
            loss.backward()
            data_grad = perturbed_image.grad.data.detach()*mask
            g = momentum*g+data_grad/data_grad.norm(2)
            g = g.detach()
            # get sign

            #sign_data_grad = g.sign()
            # data_epsilon=alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
            # data_epsilon=data_epsilon/data_epsilon.norm(2)

            # add perturbance
            t_perturbed_image = perturbed_image - epsilon/step*g
            t_perturbed_image = imgClapm(t_perturbed_image)
            t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
            perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

        out_put_image += perturbed_image
    out_put_image = out_put_image/len(white_models)
    out_put_image = imgClapm(out_put_image)
    return out_put_image


def PGD_l2_sum_img_momentum_mask_v3(data, target, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7, image_size=299):
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    g = 0
    out_put_image = 0
    for model in white_models:
        perturbed_image = data.clone().detach().requires_grad_(True)
        for _ in range(step):

            # computing Loss
            model.zero_grad()
            if model.input_size[1] != image_size:
                loss = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                loss = model(perturbed_image)

            loss = F.cross_entropy(loss, loss.argmin().view(-1))
            # compute gradient
            loss.backward()
            data_grad = perturbed_image.grad.data.detach()*mask
            g = momentum*g+data_grad/data_grad.norm(2)
            g = g.detach()
            # get sign

            #sign_data_grad = g.sign()
            # data_epsilon=alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
            # data_epsilon=data_epsilon/data_epsilon.norm(2)

            # add perturbance
            t_perturbed_image = perturbed_image - epsilon/step*g
            t_perturbed_image = imgClapm(t_perturbed_image)
            t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
            perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

        out_put_image += perturbed_image
    out_put_image = out_put_image/len(white_models)
    out_put_image = imgClapm(out_put_image)
    return out_put_image


def PGD_l2_sum_img_momentum_mask_early_stop(data, target, white_models, device, epsilon, momentum=0.9, step=10, alpha=4, sign=1, mask_ratio=0.7, image_size=299):
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    g = 0
    out_put_image = 0

    for idx, model in enumerate(white_models, 0):
        perturbed_image = data.clone().detach().requires_grad_(True)
        for _ in range(step):
            # computing Loss
            model.zero_grad()
            if model.input_size[1] != image_size:
                loss = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                loss = model(perturbed_image)
            loss = F.cross_entropy(loss, target)
            # compute gradient
            loss.backward()
            data_grad = perturbed_image.grad.data.detach()*mask
            g = momentum*g+data_grad/data_grad.norm(2)
            g = g.detach()
            # get sign

            sign_data_grad = g.sign()
            data_epsilon = alpha*(-1*(torch.sigmoid(g)-1/2).pow(2)+1)
            data_epsilon = data_epsilon/data_epsilon.norm(2)

            # add perturbance
            t_perturbed_image = perturbed_image + epsilon / \
                step*sign_data_grad*data_epsilon*sign
            t_perturbed_image = imgClapm(t_perturbed_image)
            t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
            perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)
            counter = 0
            for model in white_models[0:idx]+white_models[idx+1:]:
                label = model(perturbed_image).argmax().view(-1)
                if label == target:
                    counter += 1
                else:
                    break
            if counter == (len(white_models)-1):
                break

        out_put_image += perturbed_image
    out_put_image = out_put_image/len(white_models)
    out_put_image = imgClapm(out_put_image)
    return out_put_image



def PGD_l2_sum_logit_momentum_mask_v2(data, target, white_models, device, epsilon, momentum=0.9, step=10, mask_ratio=0.7, image_size=299):

    perturbed_image = data.clone().detach().requires_grad_(True)
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    g = 0

    for _ in range(step):

        # computing Loss
        loss = 0

        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                output = model(perturbed_image)
            loss += output

        loss = loss/len(white_models)

        _, output_label = torch.topk(loss, 2)

        if output_label[:, 0] == target.cuda():
            target = output_label[:, 1]
        else:
            target = output_label[:, 0]

        loss = F.cross_entropy(loss, target)
        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        g = momentum*g+data_grad/data_grad.norm(2)
        g = g.detach()

        # get sign

        data_epsilon = g.view(1, 3,-1)/g.view(1,3,-1).norm(2, 2).view(1,3,1)
        data_epsilon = data_epsilon.view(1,3,g.shape[2],g.shape[3])

        # add perturbance
        t_perturbed_image = perturbed_image - epsilon/step*data_epsilon
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)

        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image

def PGD_l2_sum_logit_momentum_mask_v3(data, target, white_models, device, epsilon,mode='bilinear',momentum=0.9, step=10, mask_ratio=0.7, image_size=299,stride=5,size=10):

    perturbed_image = data.clone().detach().requires_grad_(True)
    
    
    mask1=np.zeros([image_size,image_size])
    for i in range(image_size//(size+stride)+1):
        for j in range(image_size//(size+stride)+1):
            mask1[i*(size+stride):i*(size+stride)+size,j*(size+stride):j*(size+stride)+size]=1 
    mask2=np.zeros([image_size,image_size])      
    mask2[int((1-mask_ratio)*mask2.shape[1]/2):int((1+mask_ratio)*mask2.shape[1]/2),
            int((1-mask_ratio)*mask2.shape[1]/2):int((1+mask_ratio)*mask2.shape[1]/2)] = 1
    mask=np.int32(mask1+mask2>1)
    mask=torch.tensor(mask,dtype=torch.float32).to(device)
    mask=torch.cat([mask.view(1,image_size,image_size),mask.view(1,image_size,image_size),mask.view(1,image_size,image_size)]).view(1,3,image_size,image_size)
    # inital G
    g = 0
    for _ in range(step):

        # computing Loss
        loss = 0

        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                if mode =='bilinear':
                    output = model(nn.functional.interpolate(
                        perturbed_image, model.input_size[1],mode=mode,align_corners=True))
                else:
                    output = model(nn.functional.interpolate(
                        perturbed_image, model.input_size[1],mode=mode))
            else:
                output = model(perturbed_image)
            loss += output

        loss = loss/len(white_models)

        _, output_label = torch.topk(loss, 2)

        if output_label[:, 0] == target.cuda():
            target = output_label[:, 1]
        else:
            target = output_label[:, 0]

        loss = F.cross_entropy(loss, target)
        # compute gradient
        loss.backward()
        data_grad = perturbed_image.grad.data.detach()*mask
        d_shape=data_grad.shape
        data_grad = data_grad.view(1, 3,-1)/data_grad.view(1,3,-1).norm(2, 2).view(1,3,1)
        data_grad=data_grad.view(d_shape)
        g = momentum*g+data_grad #/data_grad.norm(2)
        g = g.detach()

        # get sign

        data_epsilon = g.view(1, 3,-1)/g.view(1,3,-1).norm(2, 2).view(1,3,1)
        data_epsilon = data_epsilon.view(1,3,g.shape[2],g.shape[3])

        # add perturbance
        t_perturbed_image = perturbed_image - epsilon/step*data_epsilon
        t_perturbed_image = imgClapm(t_perturbed_image)
        t_perturbed_image = t_perturbed_image*mask+data*(1-mask)
        perturbed_image = t_perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image

def PGD_l2_sum_logit_adam_mask(data, target, white_models, device, epsilon, momentum=0.9, step=10, mask_ratio=0.7, image_size=299,interpolate_size=128,beta1=0.9, beta2=0.999, epsilon_error=10e-8, alpha=0.001):

    perturbed_image = data.clone().detach().requires_grad_(True)
    mask = torch.rand_like(data)  # .to(device)
    mask.zero_()
    mask[:, :, int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2),
         int((1-mask_ratio)*mask.shape[2]/2):int((1+mask_ratio)*mask.shape[2]/2)] = 1
    mask = mask.to(device)
    # inital G
    #g = 0
    m = 0
    v = 0
    for t in range(step):

        # computing Loss
        loss = 0

        for model in white_models:
            model.zero_grad()
            if model.input_size[1] != image_size:
                output = model(nn.functional.interpolate(
                    perturbed_image, model.input_size[1]))
            else:
                output = model(perturbed_image)
            loss += output

        loss = loss/len(white_models)

        _, output_label = torch.topk(loss, 2)

        if output_label[:, 0] == target.cuda():
            target = output_label[:, 1]
        else:
            target = output_label[:, 0]

        loss = F.cross_entropy(loss, target)
        # compute gradient
        loss.backward()
        


        #
        data_grad = perturbed_image.grad.data.detach()*mask
        m = beta1*m+(1-beta1)*data_grad
        v = beta2*v+(1-beta2)*(data_grad**2)
        m_hat = m/(1-beta1**(t+1))
        v_hat = v/(1-beta2**(t+1))
        y = alpha*m_hat/(v_hat+epsilon_error)
        y=y.detach()

        data_epsilon = y.view(1, 3,-1)/y.view(1,3,-1).norm(2, 2).view(1,3,1)
        data_epsilon = data_epsilon.view(1,3,y.shape[2],y.shape[3])

        # add perturbance
        perturbed_image = perturbed_image - epsilon/step*data_epsilon
        perturbed_image = imgClapm(perturbed_image)
        perturbed_image = perturbed_image*mask+data*(1-mask)
        perturbed_image = perturbed_image.clone().detach().requires_grad_(True)

    #perturbed_image=perturbed_image-data
    #perturbed_image=nn.functional.interpolate(perturbed_image, interpolate_size)
    #perturbed_image=nn.functional.interpolate(perturbed_image, image_size)
    #perturbed_image=imgClapm(perturbed_image+data)
    return perturbed_image

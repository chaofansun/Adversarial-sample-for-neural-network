from .fbresnet import fbresnet152
from .xception import Xception
from .inceptionv4 import InceptionV4
from .resnet import resnet50
from .inceptionresnetv2 import InceptionResNetV2
from torchvision.models import inception_v3,vgg16_bn
from .fbresnet import fbresnet152
import types
import torch


def getModel(model_name, num_classes, device,path='parameters'):
    if model_name == 'resnet50':
        model = resnet50(num_classes=num_classes)
        
    elif model_name == 'xception':
         model = Xception(num_classes)
         
    elif model_name in  ['inceptionv4','InceptionV4']:
        model=InceptionV4(num_classes)

    elif model_name=='inceptionresnetv2':
        model=InceptionResNetV2(num_classes=num_classes)

    elif model_name=='InceptionV3':
        model=inception_v3(pretrained=False,num_classes=num_classes)

    elif model_name=='fbresnet152':
        model=fbresnet152(num_classes=num_classes)

    elif model_name=='VGG16':
        model=vgg16_bn(pretrained=False,num_classes=num_classes)
    else:
        assert 'No Model'

    model.load_state_dict(torch.load(f'{path}/{model_name}.pth'))
    model = model.to(device).eval()
    if model_name in ['resnet50','fbresnet152','inceptionv1_attack','resnet50_attack','VGG16_attack','VGG16']:
        for k,v in {'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}.items():
            setattr(model, k, v)
    elif model_name in ['inceptionv4','InceptionV4','inceptionresnetv2','InceptionV3','xception']:
        for k,v in {'input_space': 'RGB', 'input_size': [3, 299, 299], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}.items():
            setattr(model, k, v)
    return model

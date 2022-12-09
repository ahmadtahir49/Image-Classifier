from torchvision import datasets, transforms, models
import argparse
import torch as to
import json
from PIL import Image
import numpy as np

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

arch = {'alexnet': alexnet, 'vgg': vgg16}

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = str, help = 'Image Folder with default value "flowers"')
    parser.add_argument('--checkpoint', type = str, default = 'my_checkpoint2.pth', help = 'folder to save trained network')
    parser.add_argument('--top_k', type = int, default = 3, help = 'CNN Model Architecture')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Learning rate of the network')
    parser.add_argument('--gpu', type = str, default = 'cuda', help = 'Whether to use gpu or cpu')

    return parser.parse_args()

def predict(image_path, model, top_k, category_names, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    image = process_image(image_path)
    image = to.from_numpy(image).type(to.FloatTensor)
    image = image.unsqueeze(0)
    output = model(image)
    probs, classes = output.topk(top_k, dim=1)
    
    probs = probs.detach().numpy().tolist()[0]
    classes = classes.detach().numpy().tolist()[0]
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in classes]
    
    return probs, classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img.thumbnail((256,256))
    
    margin = 16
    img = img.crop((margin, margin, margin+224, margin+224))
    
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def load_checkpoint(checkpoint):
    checkpoint = to.load(checkpoint)
    model = models.vgg16(pretrained=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier[1].requires_grad = True
    model.classifier[3].requires_grad = True
    model.classifier[6] = to.nn.Linear(4096, 102)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_mapping']
    
    return model

in_arg = get_input_args()

model = load_checkpoint(in_arg.checkpoint)

with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(in_arg.data_dir, model, in_arg.top_k, cat_to_name, in_arg.gpu)

print(probs, classes)
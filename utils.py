import torch
from process_image import transform
from models import CONV_NN, MorpheusModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES_1 = {'biological': 0, 'cardboard': 1, 'glass': 2, 'metal': 3, 'paper': 4, 'plastic': 5, 'trash': 6} # XL model for bio + trash
CLASSES_2 = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5} # regular model for everything else
    

# first model that determines if it's biological or trash
Neo = CONV_NN(len(CLASSES_1))
Neo.to(DEVICE)
Neo.load_state_dict(torch.load("models/neo.pth", map_location=torch.device('cpu')))

# second model that determines what type of trash it is
Trinity = CONV_NN(len(CLASSES_2))
Trinity.to(DEVICE)
Trinity.load_state_dict(torch.load("models/trinity.pth", map_location=torch.device('cpu')))

# this is what makes this ensemble learning
# it takes both models' outputs and feeds them into a new model
Morpheus = MorpheusModel(len(CLASSES_2)+len(CLASSES_1), len(CLASSES_1))
Morpheus.to(DEVICE)
Morpheus.load_state_dict(torch.load("models/morpheus.pth", map_location=torch.device('cpu')))

def map_classes(predictions, cls):
    output = {}
    keys = list(cls.keys())

    for i in range(len(predictions.squeeze())):
        output[list(keys)[i]] = predictions[0][i].item()
        
    return output

def map_classes_zeros(predictions, cls):
    output = {}
    keys = list(cls.keys())

    for i in range(len(predictions.squeeze())):
        if predictions[0][i].item() > 0:
            output[list(keys)[i]] = predictions[0][i].item()
        
    return output


def get_predictions(img):
    transformed = transform(img)
    
    pred1 = Neo(transformed.unsqueeze(0).to(DEVICE))
    pred2 = Trinity(transformed.unsqueeze(0).to(DEVICE))
    
    softmax = torch.nn.Softmax(dim=1)
    softmax1 = softmax(pred1)
    softmax2 = softmax(pred2)
    
    out1 = list(CLASSES_1.keys())[softmax1.argmax(1).item()]
    out2 = list(CLASSES_2.keys())[softmax2.argmax(1).item()]
            
    c1 = map_classes(softmax1, CLASSES_1)
    c2 = map_classes(softmax2, CLASSES_2)
    
    v1 = map_classes_zeros(softmax1, CLASSES_1)
    v2 = map_classes_zeros(softmax2, CLASSES_2)
    
    return c1, c2, v1, v2, out1, out2
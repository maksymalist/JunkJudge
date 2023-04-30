import torch
import matplotlib.pyplot as plt
from PIL import Image
from process_image import transform
from models import CONV_NN, Predictioneer3000
from gen_dataset import preds_to_data, data_to_preds

seed = 123456789
torch.manual_seed(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES_1 = {'biological': 0, 'cardboard': 1, 'glass': 2, 'metal': 3, 'paper': 4, 'plastic': 5, 'trash': 6} # XL model for bio + trash
CLASSES_2 = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5} # regular model for everything else
    

# first model that determines if it's biological or trash
model_1 = CONV_NN(len(CLASSES_1))
model_1.to(DEVICE)
model_1.load_state_dict(torch.load("models/neo.pth", map_location=torch.device('cpu')))

# second model that determines what type of trash it is
model_2 = CONV_NN(len(CLASSES_2))
model_2.to(DEVICE)
model_2.load_state_dict(torch.load("models/trinity.pth", map_location=torch.device('cpu')))

# this is what makes this ensemble learning
# it takes both models' outputs and feeds them into a new model
predictioner = Predictioneer3000(len(CLASSES_2)+len(CLASSES_1), len(CLASSES_1))
predictioner.to(DEVICE)
predictioner.load_state_dict(torch.load("models/morpheus.pth", map_location=torch.device('cpu')))

img1 = Image.open("./test/IMG_2454.jpg")

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
    
    pred1 = model_1(transformed.unsqueeze(0).to(DEVICE))
    pred2 = model_2(transformed.unsqueeze(0).to(DEVICE))
    
    softmax = torch.nn.Softmax(dim=1)
    softmax1 = softmax(pred1)
    softmax2 = softmax(pred2)
    
    print(list(CLASSES_1.keys())[softmax1.argmax(1).item()])
    print(list(CLASSES_2.keys())[softmax2.argmax(1).item()])
            
    c1 = map_classes(softmax1, CLASSES_1)
    c2 = map_classes(softmax2, CLASSES_2)
    
    v1 = map_classes_zeros(softmax1, CLASSES_1)
    v2 = map_classes_zeros(softmax2, CLASSES_2)
    
    return c1, c2, v1, v2

c1, c2, v1, v2 = get_predictions(img1)

probas = preds_to_data(c1, c2).unsqueeze(0).to(DEVICE)
prediction = predictioner(probas).argmax(1).item()

final_verdict = list(CLASSES_1.keys())[prediction]

print(probas)


# VISUALIZATION

fig, ax = plt.subplots(1, 3, figsize=(10, 10))

ax[0].pie(v1.values(), labels=v1.keys())
ax[0].margins(x=20, y=10)
ax[0].set_title("First Model")

ax[1].pie(v2.values(), labels=v2.keys())
ax[1].margins(x=20, y=10)
ax[1].set_title("Second Model")

ax[2].imshow(img1)
ax[2].axis("off")
ax[2].margins(x=20, y=10)
ax[2].set_title(final_verdict)

plt.show()





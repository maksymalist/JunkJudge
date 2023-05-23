from utils import *
from gen_dataset import preds_to_data
from algo import final_say
from PIL import Image
import os
import matplotlib.pyplot as plt

seed_everything()

img_dir = "./images/"
images = []

def get_images(folder):
    for entry in os.scandir(folder):
        if entry.is_file():
            images.append((entry.path, folder.replace(img_dir, '')))
        elif entry.is_dir():
            get_images(entry.path)
            
            
get_images(img_dir)

accuracy = 0
wrong = []


for idx, (path, label) in enumerate(images):
    
    image = Image.open(path)
    
    c1, c2, v1, v2, out1, out2 = get_predictions(image)
    
    
    probas = preds_to_data(c1, c2).unsqueeze(0).to(DEVICE)
    prediction = Morpheus(probas).argmax(1).item()
    out3 = list(CLASSES_1.keys())[prediction]
    final_verdict = final_say(
        v1=v1,  # dict of the most confident classes from the first model
        v2=v2,  # dict of the most confident classes from the second model
        out1=out1, # the most confident class from the first model
        out2=out2, # the most confident class from the second model
        out3=out3, # the most confident class from the third model
        probas=probas # the probabilities from the first and second model
    )
    
    if label == final_verdict:
        accuracy += 1
    else:
        wrong.append((v1, v2, label, final_verdict, path))
        
    print(f"{ '✅' if label == final_verdict else '❌'} {idx+1}. #{path}", label, "->", final_verdict)
    print("\n")
    
print(f"Accuracy: {accuracy/len(images)}")
print(f"Mistakes {len(wrong)}/{len(images)}")


for (v1, v2, label, final_verdict, path) in wrong:
    
    image = Image.open(path)
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    
    ax[0].pie(v1.values(), labels=v1.keys())
    ax[0].margins(x=20, y=10)
    ax[0].set_title("First Model")

    ax[1].pie(v2.values(), labels=v2.keys())
    ax[1].margins(x=20, y=10)
    ax[1].set_title("Second Model")

    ax[2].imshow(image)
    ax[2].axis("off")
    ax[2].margins(x=20, y=10)
    ax[2].set_title(final_verdict)
    
    plt.show()
    
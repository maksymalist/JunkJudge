from utils import *
from gen_dataset import preds_to_data
from algo import final_say
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from db import query_embedding

seed_everything()

SAVE_PATH = "./data/predictions_4.csv"
img_dir = "./images/"
images = []

def get_images(folder):
    for entry in os.scandir(folder):
        if entry.is_file():
            images.append((entry.path, folder.replace(img_dir, '')))
        elif entry.is_dir():
            get_images(entry.path)
            
            
def jpg_check(path):
    if path[-4:] == ".jpg":
        return True
            
            
get_images(img_dir)

accuracy = 0
wrong = []
prediction_df = pd.DataFrame(columns=["status", "path", "label", "prediction", "pinecone_", "neo_", "trinity_", "morpheus_"])

status = []
paths = []
labels = []
predictions = []
pinecone_ = []
neo_ = []
trinity_ = []
morpheus_ = []

def embedding_preds(out):
    filter={
        "class": {"$eq": out},
    }
    
    data = query_embedding(probas.detach().cpu().numpy().tolist(), top_k=20, filter=filter)
    simularities = []
    
    for match in data["matches"]:
        score = match["score"]
        simularities.append(score)
        
    simularities = torch.tensor(simularities).to(DEVICE)
    return simularities.mean().item()

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
        status.append("✅")
    else:
        status.append("❌")
        wrong.append((v1, v2, label, final_verdict, path))
        
    paths.append(path)
    labels.append(label)
    predictions.append(final_verdict)
    pinecone_.append(embedding_preds(final_verdict))
    neo_.append(out1)
    trinity_.append(out2)
    morpheus_.append(out3)
        
    print(f"{ '✅' if label == final_verdict else '❌'} {idx+1}. #{path}", label, "->", final_verdict)
    print("\n")
    
print(f"Accuracy: {accuracy/len(images)}")
print(f"Mistakes {len(wrong)}/{len(images)}")

prediction_df["status"] = status
prediction_df["path"] = paths
prediction_df["label"] = labels
prediction_df["prediction"] = predictions
prediction_df["pinecone_"] = pinecone_
prediction_df["neo_"] = neo_
prediction_df["trinity_"] = trinity_
prediction_df["morpheus_"] = morpheus_
prediction_df.to_csv(SAVE_PATH, index=False)


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
    
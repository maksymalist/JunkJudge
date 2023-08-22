from utils import *
from gen_dataset import preds_to_data
from algo import final_say
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from db import query_embedding

seed_everything()

SAVE_PATH = "./data/predictions_5.csv"
img_dir = "./images/"
images = []

wrong_big = ['trash130.jpg', 'trash131.jpg', 'trash125.jpg', 'trash13.jpg', 'trash119.jpg', 'trash11.jpg', 'trash132.jpg', 'trash122.jpg', 'trash14.jpg', 'trash121.jpg', 'trash17.jpg', 'trash108.jpg', 'trash71.jpg', 'trash58.jpg', 'trash72.jpg', 'trash63.jpg', 'trash62.jpg', 'trash60.jpg', 'trash8.jpg', 'trash9.jpg', 'trash49.jpg', 'trash50.jpg', 'trash78.jpg', 'trash5.jpg', 'trash92.jpg', 'trash51.jpg', 'trash84.jpg', 'trash7.jpg', 'trash6.jpg', 'trash56.jpg', 'trash42.jpg', 'trash81.jpg', 'trash41.jpg', 'trash55.jpg', 'trash82.jpg', 'trash96.jpg', 'trash83.jpg', 'trash27.jpg', 'trash111.jpg', 'trash110.jpg', 'trash104.jpg', 'trash24.jpg', 'trash106.jpg', 'trash107.jpg', 'trash113.jpg', 'trash25.jpg', 'trash31.jpg', 'trash35.jpg', 'trash21.jpg', 'trash34.jpg', 'trash116.jpg', 'trash22.jpg', 'trash128.jpg', 'trash36.jpg', 'trash115.jpg']

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

def embedding_preds(out, probas):
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
    
    name = f'{label}{idx}.jpg'
    
    print(name)

    if name in wrong_big:
        status.append("❌")
        wrong.append((v1, v2, label, final_verdict, path))
        img = Image.open(path)
        img.save(f"./data/wrong2/{label}{idx}.jpg")
        
    else:
        accuracy += 1
        status.append("✅")
        img = Image.open(path)
        img.save(f"./data/trash/{label}{idx}.jpg")
        

    paths.append(path)
    labels.append(label)
    predictions.append(final_verdict)
    pinecone_.append(embedding_preds(final_verdict, probas))
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


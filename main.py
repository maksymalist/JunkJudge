from utils import *
import torch
from PIL import Image
from gen_dataset import preds_to_data, data_to_preds
from algo import final_say
import matplotlib.pyplot as plt

seed = 123456
torch.manual_seed(seed)

img1 = Image.open("images/trash/IMG_2451.jpg")

c1, c2, v1, v2, out1, out2 = get_predictions(img1)

probas = preds_to_data(c1, c2).unsqueeze(0).to(DEVICE)
prediction = Morpheus(probas).argmax(1).item()
out3 = list(CLASSES_1.keys())[prediction]

final_verdict = final_say(out1, out2, out3)
print(final_verdict)


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





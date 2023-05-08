from collections import Counter
from utils import CLASSES_1, CLASSES_2
from db import query_embedding
import torch
from utils import DEVICE

def final_say(v1, v2, out1, out2, out3, probas):
    
    filter={
        "class": {"$eq": out3},
    }
    
    data = query_embedding(probas.detach().cpu().numpy().tolist(), top_k=20, filter=filter)
    simularities = []
    
    for match in data["matches"]:
        score = match["score"]
        simularities.append(score)
        
    simularities = torch.tensor(simularities).to(DEVICE)
    print(f"confidence {out3}: ", simularities.mean())
        
    
    if "trash" in v2:
        trash_confidence = v2["trash"] / sum(v2.values())
        verdict_confidence = v1[max(v1, key=v1.get)] / sum(v1.values())
        
        if trash_confidence > verdict_confidence:
            return "trash"
    
    return Counter([out1, out2, out3]).most_common(1)[0][0]
    
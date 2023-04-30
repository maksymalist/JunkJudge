import torch

MASTER_INDEX = {
    'biological1': 0, 
    'cardboard1': 1, 
    'glass1': 2, 
    'metal1': 3, 
    'paper1': 4, 
    'plastic1': 5, 
    'trash1': 6,
    'cardboard2': 7, 
    'glass2': 8, 
    'metal2': 9, 
    'paper2': 10, 
    'plastic2': 11, 
    'trash2': 12
}

def preds_to_data(c1, c2):
    
    data = torch.zeros(13)
    
    for key in c1.keys():
        new_key = str(key+"1")
        data[MASTER_INDEX[new_key]] = c1[str(key)]
        
    for key in c2.keys():
        new_key = str(key+"2")
        data[MASTER_INDEX[new_key]] = c2[str(key)]
        
    return data

def data_to_preds(data):
    
    c1 = {}
    c2 = {}
    
    for key in MASTER_INDEX.keys():
        if key[-1] == "1":
            c1[key[:-1]] = data[MASTER_INDEX[key]].item()
        elif key[-1] == "2":
            c2[key[:-1]] = data[MASTER_INDEX[key]].item()
            
    return c1, c2
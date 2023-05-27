from collections import Counter

def final_say(v1, v2, out1, out2, out3, probas):
        
    # makes trash predictions more confident
    if "trash" in v2:
        trash_confidence = v2["trash"] / sum(v2.values())
        verdict_confidence = v1[max(v1, key=v1.get)] / sum(v1.values())
        
        if trash_confidence > verdict_confidence and v1["biological"] < 0.45:
            return "trash"
    
    return Counter([out1, out2, out3]).most_common(1)[0][0]
    
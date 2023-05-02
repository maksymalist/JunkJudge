from collections import Counter
from utils import CLASSES_1, CLASSES_2

def final_say(out1, out2, out3):

    # add your logic here
    
    return Counter([out1, out2, out3]).most_common(1)[0][0]
    
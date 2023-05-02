from collections import Counter

def final_say(out1, out2, out3):

    # add your logic here
    
    return Counter([out1, out2, out3]).most_common(1)[0][0]
    
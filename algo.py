from torch.nn.functional import normalize

CLASSES_1 = {'biological': 0, 'cardboard': 1, 'glass': 2, 'metal': 3, 'paper': 4, 'plastic': 5, 'trash': 6} # XL model for bio + trash
CLASSES_2 = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5} # regular model for everything else

class Electioneer3000:
    def __init__(self, output1, output2):
        self.output1 = output1.squeeze()
        self.output2 = output2.squeeze()
        
    def bio_check(self):
        if self.output1[CLASSES_1["biological"]] > 0.25:
            return True
        return False
        
    def trash_check(self, output, classes):
        if output[classes["trash"]] > 0.25:
            return True
        return False
    
    def lower_class_probs_1(self):
        for i in range(len(self.output1)):
            if i != CLASSES_1["biological"] or i != CLASSES_1["trash"]:
                self.output1[i] *= 0.75
                
    def lower_class_probs_2(self):
        for i in range(len(self.output2)):
            if i != CLASSES_2["trash"]:
                self.output2[i] *= 0.75     
        
    def forward(self):
        
        ## OUTPUT 1 ##
        # this block here influences the model to be more confident in its trash / bio predictions
        if self.bio_check():
            self.output1[CLASSES_1["biological"]] = self.output1[CLASSES_1["biological"]] * 1.25
            self.lower_class_probs_1()
            
        elif self.trash_check(self.output1, CLASSES_1):
            self.output1[CLASSES_1["trash"]] = self.output1[CLASSES_1["trash"]] * 1.25
            self.lower_class_probs_1()


        ## OUTPUT 2 ##
        # this block here influences the model to be more confident in its trash predictions
        if self.trash_check(self.output2, CLASSES_2):
            self.output2[CLASSES_2["trash"]] = self.output2[CLASSES_2["trash"]] * 1.45
            self.lower_class_probs_2()
            
        return normalize(self.output1.unsqueeze(0), dim=1), normalize(self.output2.unsqueeze(0), dim=1)
        
        
            

        
    

    
    
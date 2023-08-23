import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights, ConvNeXt_Tiny_Weights

def CONV_NN(num_classes):
    
    # load the pretrained model because what's the point of re-inventing the wheel??
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT) # in the new version of PyTorch, you can't use pretrained=True
    
    # Freeze all the layers because we only want to train the new layers
    
    for param in model.parameters():
        # disable gradients because we don't need to include pretrained data in the backprop
        param.requires_grad = False
        
        
    # Replace the last layer with a MLP mixer with dropout
    
    model.classifier[-1] = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model
  
def CONV_NN_V2(num_classes):
    
    # load the pretrained model because what's the point of re-inventing the wheel??
    model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT) # in the new version of PyTorch, you can't use pretrained=True
    
    # Freeze all the layers because we only want to train the new layers
    
    for param in model.parameters():
        # disable gradients because we don't need to include pretrained data in the backprop
        param.requires_grad = False
        
        
    # Replace the last layer with a MLP mixer with dropout
    
    model.classifier[-1] = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model


class MorpheusModel(nn.Module):
  def __init__(self, num_inputs, num_outputs):
    super(MorpheusModel, self).__init__()

    self.l1 = nn.Linear(num_inputs, 256)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(256, num_outputs)


  def forward(self, x):
    output = self.l1(x)
    output = self.relu(output)
    output = self.l2(output)
    return output
  
def CONV_NN_V2(num_classes):

    # load the pretrained model because what's the point of re-inventing the wheel??
    model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT) # in the new version of PyTorch, you can't use pretrained=True

    # Freeze all the layers because we only want to train the new layers

    for param in model.parameters():
        # disable gradients because we don't need to include pretrained data in the backprop
        param.requires_grad = False


    # Replace the last layer with a MLP mixer with dropout

    model.classifier[-1] = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize image
    transforms.CenterCrop((224, 224)), # center crop image
    transforms.ToTensor(), # convert to Pytorch tensor
    transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225)) # normalize
])
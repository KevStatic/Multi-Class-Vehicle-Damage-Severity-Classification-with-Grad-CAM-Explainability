import torch
from PIL import Image
from torchvision import transforms
from model_custom import DamageDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DamageDetector(num_classes=2).to(device)
model.load_state_dict(torch.load("custom_damage_detector.pth"))
model.eval()

def predict(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_t)
    print(preds.shape)
    return preds

if __name__ == "__main__":
    predict("Datasets/coco/val/1.jpg")

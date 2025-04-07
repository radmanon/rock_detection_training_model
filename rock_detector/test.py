# test.py

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw

from rock_dataset import RockDataset

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def predict_and_save():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('rock_detector.pth', map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])
    test_dataset = RockDataset('dataset/test', 'dataset/test/_annotations.coco.json', transform)

    os.makedirs("results", exist_ok=True)

    for idx in range(len(test_dataset)):
        img, _ = test_dataset[idx]
        img_tensor = img.to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        img_pil = T.ToPILImage()(img.cpu())
        draw = ImageDraw.Draw(img_pil)

        for box, score in zip(prediction['boxes'], prediction['scores']):
            if score > 0.8:  # confidence threshold
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"{score:.2f}", fill="red")

        img_pil.save(f"results/test_img_{idx+1}.jpg")

    print("✅ Results saved to 'results/' folder.")

if __name__ == "__main__":
    predict_and_save()

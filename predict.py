# predict.py

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def predict_on_image(image_path, model_path='saved_models/rock_detector_epoch11.pth', threshold=0.6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = get_model(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image and convert to tensor
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).to(device).unsqueeze(0)

    # Run prediction
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Draw predictions
    size_threshold = 15
    draw = ImageDraw.Draw(image)
    for box, score in zip(outputs['boxes'], outputs['scores']):
        if score >= threshold:
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # Label based on size
            if area >= size_threshold:
                label = f"Bigger - {score:.2f}"
            else:
                label = f"{score:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), label, fill="red")

    # Save output image
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", os.path.basename(image_path))
    image.save(save_path)
    print(f"✅ Prediction complete. Saved result to: {save_path}")

# Example usage
if __name__ == "__main__":
    # Change this to your test image path (no JSON needed)
    predict_on_image("q7.jpg")  # replace with your own image filename

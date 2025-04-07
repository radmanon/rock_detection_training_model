# train.py

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import time
import os

from rock_dataset import RockDataset

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    return T.Compose([T.ToTensor()])

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    train_data = RockDataset('dataset/train', 'dataset/train/_annotations.coco.json', get_transform())
    valid_data = RockDataset('dataset/valid', 'dataset/valid/_annotations.coco.json', get_transform())

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=10)  # (9 rock classes + 1 background)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 15

    os.makedirs("saved_models", exist_ok=True)  # directory for model checkpoints

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        for idx, (imgs, targets) in enumerate(train_loader):
            print(f"Epoch {epoch+1}, Batch {idx+1}/{len(train_loader)}")

            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Total Loss: {total_loss:.4f}")

        # Save checkpoint for this epoch
        checkpoint_path = f"saved_models/rock_detector_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Saved model for epoch {epoch+1} to {checkpoint_path}")

                # Cooldown to protect GPU
        print("🌬️ Cooling GPU for 30 seconds...\n")
        time.sleep(30)

    # Save final model
    torch.save(model.state_dict(), 'rock_detector.pth')
    print("\n🎉 Final model saved as rock_detector.pth")

if __name__ == "__main__":
    train()
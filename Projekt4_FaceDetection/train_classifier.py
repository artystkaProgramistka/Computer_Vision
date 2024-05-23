import torch
import cv2
from PIL import Image
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CROP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])


def get_dataloader():
    data_dir = './crops/'
    dataset = datasets.ImageFolder(root=data_dir, transform=CROP_TRANSFORM)
    print(f'{dataset.classes=}')
    print(f'{len(dataset)=}')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    return dataset, train_loader, val_loader

def visualize_batch(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig('batch.png')

def visualize_dataloader():
    _, train_loader, val_loader = get_dataloader()

    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)
    visualize_batch(out)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


def main():
    dataset, train_loader, val_loader = get_dataloader()

    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    validate_model(model, val_loader, criterion)

    # inference trained model on some video fragment
    model.eval()
    from main import read_video_frames, detect_faces, draw_boxes
    frames = read_video_frames('ap.mp4', 15000, 16000, 1)
    result = []
    for frame in frames:
        frame = frame[:, :, ::-1].copy()  # BGR to RGB
        boxes = detect_faces(frame)
        labels = []
        for x1, y1, x2, y2 in boxes:
            crop = frame[y1:y2, x1:x2]
            crop = Image.fromarray(crop)
            transformed = CROP_TRANSFORM(crop)
            with torch.no_grad():
                outputs = model(transformed)
                _, predicted = torch.max(outputs, 1)
                labels.append(dataset.classes[predicted])
        draw_boxes(frame, boxes, labels)
        result.append(frame)
    save_images_as_video(result, 'final_video.mp4', 1)


def save_images_as_video(image_list, output_path, frame_rate):
    if not image_list:
        raise ValueError("The image list is empty")

    # Get the height, width, and channels of the first image
    height, width, channels = image_list[0].shape

    if channels != 3:
        raise ValueError("Images must be RGB (3 channels)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec as needed
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for image in image_list:
        if image.shape[0] != height or image.shape[1] != width:
            raise ValueError("All images must have the same dimensions")

        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video_writer.release()



if __name__ == '__main__':
    visualize_dataloader()
    main()

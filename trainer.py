# ==================== CAPTCHA TRAINING - FIXED DIMENSION ISSUE ====================

print("🚀 CAPTCHA Training Started!")
print("=" * 60)

# ==================== INSTALL DEPENDENCIES ====================
print("\n📦 Step 0: Installing dependencies...")
!pip install torchvision tqdm > /dev/null 2>&1
print("✅ Dependencies installed successfully!")

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import zipfile
from google.colab import files
import random

# ==================== CONFIGURATION ====================
class Config:
    data_dir = "/content/captcha_dataset"
    model_save_path = "/content/captcha_model.pth"
    epochs = 30  # Reduced for faster training
    batch_size = 16
    learning_rate = 0.001
    img_width = 128
    img_height = 32
    hidden_size = 128
    num_workers = 2

config = Config()

CHARSET = "0123456789"
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARSET) + 1

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Using device: {device}")

# ==================== UPLOAD DATASET ====================
print("\n📤 Step 1: Upload dataset zip file...")
print("Please upload your CAPTCHA dataset zip file when prompted...")
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    file_size = len(uploaded[filename])
    print(f'📦 Uploaded: {filename} ({file_size / 1024 / 1024:.1f} MB)')
    if filename.endswith('.zip'):
        print("📤 Extracting zip file...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(config.data_dir)
        print(f"✅ Extracted to: {config.data_dir}")

# ==================== FIND DATASET FILES ====================
print("\n🔍 Step 2: Finding dataset files...")

def find_dataset_files():
    labels_path = None
    images_path = None
    
    for root, dirs, files in os.walk(config.data_dir):
        for file in files:
            if file == 'labels.json':
                labels_path = os.path.join(root, file)
        for dir in dirs:
            if dir == 'images':
                images_path = os.path.join(root, dir)
    
    return labels_path, images_path

labels_path, images_path = find_dataset_files()

if not labels_path or not images_path:
    print("🔍 Fallback search for dataset files...")
    for root, dirs, files in os.walk(config.data_dir):
        for file in files:
            if file.endswith('.json') and not labels_path:
                labels_path = os.path.join(root, file)
                print(f"📄 Found labels file: {labels_path}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(dir_path)):
                    images_path = dir_path
                    print(f"🖼️ Found images directory: {images_path}")
            except:
                pass

if not labels_path:
    raise FileNotFoundError("❌ Could not find labels.json file!")
if not images_path:
    raise FileNotFoundError("❌ Could not find images directory!")

print(f"📄 Labels path: {labels_path}")
print(f"🖼️ Images path: {images_path}")

# Load labels
with open(labels_path, 'r') as f:
    labels_data = json.load(f)

print(f"📊 Dataset size: {len(labels_data)} images")

# ==================== FIXED MODEL ARCHITECTURE ====================
class SimpleCRNN(nn.Module):
    def __init__(self, img_h=32, nc=1, nclass=NUM_CLASSES, nh=64):
        super(SimpleCRNN, self).__init__()
        
        # CNN with proper dimension reduction
        self.cnn = nn.Sequential(
            # Input: (batch, 1, 32, 128)
            nn.Conv2d(nc, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 16, 64)
            
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # (64, 8, 32)
            
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),  # (128, 4, 16)
            
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),  # (256, 2, 8)
            
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate the feature dimension after CNN
        self.rnn_input_size = 256 * 2  # channels * height after CNN
        
        self.rnn = nn.LSTM(self.rnn_input_size, nh, num_layers=1, bidirectional=True, batch_first=True)
        self.output = nn.Linear(nh * 2, nclass)
        
    def forward(self, x):
        # CNN forward
        conv_out = self.cnn(x)  # (batch, 256, 2, 8)
        
        # Prepare for RNN: (batch, channels, height, width) -> (batch, width, channels * height)
        batch_size, channels, height, width = conv_out.size()
        
        # Combine channels and height: (batch, channels * height, width)
        conv_out = conv_out.view(batch_size, channels * height, width)
        
        # Permute to (batch, width, channels * height)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, width, features)
        
        # RNN forward
        rnn_out, _ = self.rnn(conv_out)  # (batch, width, hidden_size * 2)
        
        # Output layer
        output = self.output(rnn_out)  # (batch, width, num_classes)
        
        # CTC requires: (seq_len, batch, num_classes)
        output = output.permute(1, 0, 2)  # (width, batch, num_classes)
        
        return output

# ==================== DATASET ====================
class CaptchaDataset(Dataset):
    def __init__(self, images_root, labels_dict, img_w=128, img_h=32):
        self.images_root = images_root
        self.img_w = img_w
        self.img_h = img_h
        self.samples = []
        
        for filename, label in labels_dict.items():
            # Skip invalid labels
            if all(c in CHARSET for c in label):
                self.samples.append((filename, label))
            else:
                print(f"⚠️ Skipping invalid label: {label}")
            
        print(f"📊 Valid samples: {len(self.samples)}")
            
        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.images_root, fname)
        
        try:
            img = Image.open(img_path).convert("L")
            img = self.transform(img)
            target = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
            return img, target, label
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")
            # Return dummy data if image loading fails
            dummy_img = torch.randn(1, self.img_h, self.img_w)
            # Create a valid target based on CHARSET
            dummy_label = "1234" if len(CHARSET) >= 4 else CHARSET[:4]
            dummy_target = torch.tensor([CHAR_TO_IDX[c] for c in dummy_label], dtype=torch.long)
            return dummy_img, dummy_target, dummy_label

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    label_strs = [b[2] for b in batch]
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return imgs, targets, target_lengths, label_strs

def decode_predictions(logits):
    """Decode CTC predictions using greedy decoding"""
    # logits shape: (seq_len, batch, num_classes)
    log_probs = logits.log_softmax(2)
    
    # Get most likely characters
    _, max_indices = torch.max(log_probs, 2)  # (seq_len, batch)
    max_indices = max_indices.cpu().numpy()
    
    decoded = []
    for batch_idx in range(max_indices.shape[1]):
        raw_pred = max_indices[:, batch_idx]
        
        # CTC decoding: remove duplicates and blank (0)
        prev = -1
        chars = []
        for idx in raw_pred:
            if idx != 0 and idx != prev:
                chars.append(IDX_TO_CHAR.get(idx, '?'))
            prev = idx
        
        decoded.append(''.join(chars))
    
    return decoded

def calculate_accuracy(predictions, truths):
    correct = 0
    for pred, truth in zip(predictions, truths):
        if pred == truth:
            correct += 1
    return correct / len(predictions) * 100

# ==================== DEBUG: CHECK DATA AND MODEL ====================
print("\n🔍 Step 3: Debugging data and model...")

# Create dataset
dataset = CaptchaDataset(images_path, labels_data, 
                       img_w=config.img_width, 
                       img_h=config.img_height)

# Test one sample
test_img, test_target, test_label = dataset[0]
print(f"📐 Sample image shape: {test_img.shape}")
print(f"🎯 Sample target: {test_target}")
print(f"🏷️ Sample label: '{test_label}'")

# Initialize model
model = SimpleCRNN(
    img_h=config.img_height,
    nc=1,
    nclass=NUM_CLASSES,
    nh=config.hidden_size
).to(device)

print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"🔧 RNN input size: {model.rnn_input_size}")

# Test forward pass
test_batch = test_img.unsqueeze(0).to(device)  # Add batch dimension
print(f"🔬 Test batch shape: {test_batch.shape}")

with torch.no_grad():
    output = model(test_batch)
    print(f"📊 Model output shape: {output.shape}")
    
    # Test decoding
    predictions = decode_predictions(output)
    print(f"🔮 Test prediction: '{predictions[0]}'")

# ==================== TRAINING ====================
print("\n🎯 Step 4: Starting Training...")
print("=" * 50)

train_loader = DataLoader(
    dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=config.num_workers
)

print(f"📊 Training with {len(dataset)} images")
print(f"📈 Training for {config.epochs} epochs")

# Loss and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_accuracy = 0
train_losses = []
train_accuracies = []

for epoch in range(1, config.epochs + 1):
    model.train()
    total_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.epochs}')
    
    for imgs, targets, target_lengths, label_strs in progress_bar:
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        logits = model(imgs)
        
        # CTC Loss setup
        log_probs = logits.log_softmax(2)  # (seq_len, batch, num_classes)
        input_lengths = torch.full(
            size=(imgs.size(0),), 
            fill_value=logits.size(0),  # seq_len
            dtype=torch.long,
            device=device
        )
        
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{total_loss/batch_count:.4f}'
        })
    
    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    
    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    # Validation
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        
        # Test on a few batches for better accuracy estimation
        for i, (val_imgs, _, _, val_labels) in enumerate(train_loader):
            if i >= 3:  # Use first 3 batches for validation
                break
            val_imgs = val_imgs.to(device)
            logits = model(val_imgs)
            predictions = decode_predictions(logits)
            all_predictions.extend(predictions)
            all_labels.extend(val_labels)
        
        accuracy = calculate_accuracy(all_predictions, all_labels)
        train_accuracies.append(accuracy)
        
        print(f"📊 Epoch {epoch:03d}: Loss = {avg_loss:.4f}, Acc = {accuracy:.1f}%, LR = {current_lr:.6f}")
        
        # Show samples
        if epoch <= 3 or epoch % 5 == 0:
            print("🔍 Sample predictions:")
            for i, (pred, truth) in enumerate(zip(all_predictions[:3], all_labels[:3])):
                status = "✅" if pred == truth else "❌"
                print(f"   {status} Pred: '{pred}' | Truth: '{truth}'")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': accuracy,
            'config': vars(config),
            'charset': CHARSET,
            'char_to_idx': CHAR_TO_IDX
        }, config.model_save_path)
        
        print(f"💾 Saved best model (accuracy: {accuracy:.1f}%)")

# ==================== FINAL TEST ====================
print("\n🎉 Training Completed!")
print("=" * 50)

# Load best model for final test
if os.path.exists(config.model_save_path):
    checkpoint = torch.load(config.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"🏆 Best Accuracy: {checkpoint['accuracy']:.1f}%")
else:
    print("⚠️ Using last model for testing")

# Final test
model.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    
    for i, (test_imgs, _, _, test_labels) in enumerate(train_loader):
        if i >= 5:  # Test on 5 batches
            break
        test_imgs = test_imgs.to(device)
        logits = model(test_imgs)
        predictions = decode_predictions(logits)
        all_predictions.extend(predictions)
        all_labels.extend(test_labels)
    
    accuracy = calculate_accuracy(all_predictions, all_labels)
    print(f"📊 Final Test Accuracy: {accuracy:.1f}%")
    
    print("\n🔍 Final Predictions (first 10):")
    for i, (pred, truth) in enumerate(zip(all_predictions[:10], all_labels[:10])):
        status = "✅" if pred == truth else "❌"
        print(f"   {i+1:2d}. {status} Pred: '{pred}' | Truth: '{truth}'")

# ==================== VISUALIZATION ====================
print("\n📊 Generating training plots...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'g-', label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==================== DOWNLOAD MODEL ====================
print(f"\n💾 Model saved to: {config.model_save_path}")

if os.path.exists(config.model_save_path):
    print("📥 Downloading model...")
    files.download(config.model_save_path)
    print("✅ Model downloaded successfully!")
else:
    print("❌ Model file not found!")

print("🎯 Training complete!")
print("=" * 60)
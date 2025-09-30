#!/usr/bin/env python3
"""
collect_dataset.py - FIXED for 6-digit captcha
Script untuk mengumpulkan dataset captcha secara manual dengan resume capability
"""

import os
import time
import requests
from PIL import Image
from io import BytesIO
import json

# Konfigurasi
LOGIN_URL = "https://mlive.minemedia.tv/"
GENERATE_CAPTCHA_URL = "https://mlive.minemedia.tv/v3/util/generate_captcha?cptc=2&t="
GET_CAPTCHA_URL = "https://mlive.minemedia.tv/v3/util/get_captcha?cptc=2&id="
DATASET_DIR = "captcha_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.json")

# Buat direktori jika belum ada
os.makedirs(IMAGES_DIR, exist_ok=True)

# Session
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': LOGIN_URL
})

def generate_captcha_id():
    """Generate captcha ID"""
    ts = int(time.time() * 1000)
    url = f"{GENERATE_CAPTCHA_URL}{ts}"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        captcha_id = (data.get('data') or {}).get('id') or data.get('id') or data.get('captcha_id')
        return captcha_id
    except Exception as e:
        print(f"Error generating captcha ID: {e}")
        return None

def get_captcha_image(captcha_id):
    """Get captcha image by ID"""
    if not captcha_id:
        return None
    url = f"{GET_CAPTCHA_URL}{captcha_id}"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception as e:
        print(f"Error getting captcha image: {e}")
        return None

def load_existing_labels():
    """Load existing labels from file"""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_labels(labels):
    """Save labels to file"""
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)

def get_next_image_number(labels):
    """Get the next image number based on existing labels"""
    if not labels:
        return 1
    
    # Find the highest number in existing labels
    max_num = 0
    for filename in labels.keys():
        if filename.startswith('captcha_') and filename.endswith('.png'):
            try:
                num = int(filename[8:-4])  # Extract number from "captcha_001.png"
                max_num = max(max_num, num)
            except ValueError:
                continue
    
    return max_num + 1

def find_missing_images(labels):
    """Find images that are in labels but missing from disk"""
    missing = []
    for filename in labels.keys():
        image_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(image_path):
            missing.append(filename)
    return missing

def collect_dataset():
    """Main function to collect dataset"""
    labels = load_existing_labels()
    existing_count = len(labels)
    
    # Check for missing images
    missing_images = find_missing_images(labels)
    if missing_images:
        print(f"⚠️  Warning: {len(missing_images)} images in labels are missing from disk:")
        for img in missing_images[:5]:  # Show first 5 missing images
            print(f"   - {img}")
        if len(missing_images) > 5:
            print(f"   ... and {len(missing_images) - 5} more")
        
        response = input("❓ Do you want to remove missing images from labels? (y/n): ").strip().lower()
        if response == 'y':
            for img in missing_images:
                labels.pop(img, None)
            save_labels(labels)
            print(f"✅ Removed {len(missing_images)} missing images from labels")
            existing_count = len(labels)
    
    next_image_num = get_next_image_number(labels)
    
    print(f"📊 Existing dataset: {existing_count} images")
    print(f"🔜 Next image number: {next_image_num}")
    print("🚀 Starting data collection...")
    print("💡 Instructions:")
    print("   - Enter the 6-digit captcha code")
    print("   - Type 'skip' to skip current image")
    print("   - Type 'quit' to exit and save")
    print("   - Type 'delete' to delete last entry")
    print("-" * 50)
    
    count = existing_count
    session_count = 0
    
    try:
        while True:
            print(f"\n🔄 Collecting image {next_image_num} (Total: {count} images)")
            
            # Get captcha
            captcha_id = generate_captcha_id()
            if not captcha_id:
                print("❌ Failed to get captcha ID, retrying...")
                time.sleep(2)
                continue
                
            image = get_captcha_image(captcha_id)
            if not image:
                print("❌ Failed to get captcha image, retrying...")
                time.sleep(2)
                continue
                
            # Save image
            image_filename = f"captcha_{next_image_num:03d}.png"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            image.save(image_path)
            
            # Show image to user
            print(f"📸 Image saved: {image_filename}")
            try:
                image.show()  # This will open the image with default viewer
            except:
                print(f"⚠️  Could not auto-open image, please check: {image_path}")
            
            # Get label from user
            while True:
                user_input = input("🔢 Enter captcha code (6 digits): ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Saving and exiting...")
                    save_labels(labels)
                    print(f"✅ Session summary: Added {session_count} new images")
                    print(f"📊 Total dataset: {count} images")
                    return
                
                elif user_input.lower() == 'skip':
                    print("⏭️ Skipping this image...")
                    # Remove the skipped image
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    # Don't increment next_image_num so we reuse this number
                    break
                
                elif user_input.lower() == 'delete':
                    if labels:
                        # Find and delete the last entry
                        last_filename = list(labels.keys())[-1]
                        labels.pop(last_filename)
                        # Also delete the image file if it exists
                        last_image_path = os.path.join(IMAGES_DIR, last_filename)
                        if os.path.exists(last_image_path):
                            os.remove(last_image_path)
                        
                        save_labels(labels)
                        count -= 1
                        session_count = max(0, session_count - 1)
                        next_image_num -= 1  # Go back one number
                        print(f"✅ Deleted last entry: {last_filename}")
                        print(f"📊 Total dataset now: {count} images")
                    else:
                        print("❌ No entries to delete!")
                    break
                
                elif len(user_input) == 6 and user_input.isdigit():
                    # Check if this filename already exists in labels
                    if image_filename in labels:
                        response = input(f"⚠️  {image_filename} already exists with label '{labels[image_filename]}'. Overwrite? (y/n): ").strip().lower()
                        if response != 'y':
                            print("Skipping overwrite...")
                            break
                    
                    # Valid input - save label
                    old_label = labels.get(image_filename)
                    labels[image_filename] = user_input
                    count += 1
                    session_count += 1
                    next_image_num += 1
                    
                    save_labels(labels)  # Save after each successful entry
                    
                    if old_label:
                        print(f"✅ Label updated: {old_label} → {user_input}")
                    else:
                        print(f"✅ Label saved: {user_input}")
                    print(f"📊 Progress: {session_count} new images this session")
                    break
                
                else:
                    print("❌ Invalid input! Please enter exactly 6 digits, 'skip', 'delete', or 'quit'")
            
            # Small delay to avoid rate limiting
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        save_labels(labels)
        print(f"✅ Progress saved! Added {session_count} new images this session")
        print(f"📊 Total dataset: {count} images")

def show_statistics():
    """Show dataset statistics"""
    labels = load_existing_labels()
    if not labels:
        print("📊 Dataset is empty")
        return
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total images: {len(labels)}")
    
    # Check for missing images
    missing = find_missing_images(labels)
    if missing:
        print(f"   ⚠️  Missing images: {len(missing)}")
    
    # Show label length distribution
    length_dist = {}
    for label in labels.values():
        length = len(label)
        length_dist[length] = length_dist.get(length, 0) + 1
    
    print("   Label length distribution:")
    for length in sorted(length_dist.keys()):
        print(f"     {length} digits: {length_dist[length]} images")
    
    # Show recent entries
    recent = list(labels.items())[-5:]
    print("   Recent entries:")
    for filename, label in recent:
        print(f"     {filename}: {label}")

if __name__ == "__main__":
    print("🎯 Captcha Dataset Collector")
    print("=" * 40)
    
    show_statistics()
    print("\n")
    
    collect_dataset()
    
    # Show final statistics
    print("\n" + "=" * 40)
    show_statistics()
#!/usr/bin/env python3
import time
import re
import json
from io import BytesIO
from PIL import Image
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2
import numpy as np
import requests

# --- Config ---
LOGIN_URL = "https://mlive.minemedia.tv/"
USERNAME = "99988805682"
PASSWORD = "92076958"
TARGET_URL = "https://mlive.minemedia.tv/device/list/"
MAX_ATTEMPTS = 5

# API Endpoints
GENERATE_CAPTCHA_URL = "https://mlive.minemedia.tv/v3/util/generate_captcha?cptc=2&t="
GET_CAPTCHA_URL = "https://mlive.minemedia.tv/v3/util/get_captcha?cptc=2&id="
LOGIN_API_URL = "https://mlive.minemedia.tv/v3/users/web_login/"

# --- Session Management ---
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Origin': 'https://mlive.minemedia.tv',
    'Referer': 'https://mlive.minemedia.tv/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
})

class CaptchaSolver:
    """Handles CAPTCHA solving operations"""
    
    @staticmethod
    def solve_captcha(image: Image.Image) -> str | None:
        """Solve CAPTCHA using OCR with OpenCV enhancement"""
        # Convert PIL Image to OpenCV format
        img = np.array(image.convert("L"))  # Convert to grayscale numpy array
        
        # Check if image is vertical and rotate if needed
        height, width = img.shape
        print(f"📐 Image dimensions: {width}x{height}")
        
        if height > width * 1.5:  # Image is clearly vertical
            print("🔄 Rotating vertical CAPTCHA 90 degrees")
            img = np.rot90(img)  # Rotate 90 degrees counter-clockwise
        
        # Preprocessing with OpenCV
        img = CaptchaSolver._preprocess_image(img)
        
        # Convert back to PIL Image for Tesseract
        processed_img = Image.fromarray(img)
        
        # OCR with Tesseract
        best_text = CaptchaSolver._ocr_with_configs(processed_img)
        
        if best_text:
            print(f"✓ CAPTCHA solved: {best_text}")
            return best_text
        
        # Fallback: manual rotation if still failing
        return CaptchaSolver._try_rotation_approaches(image)
    
    @staticmethod
    def _preprocess_image(img):
        """Apply image preprocessing for better OCR results"""
        # 1. Resize to improve resolution
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Noise reduction with Gaussian blur
        img = cv2.GaussianBlur(img, (1, 1), 0)
        
        # 3. Adaptive thresholding for binarization
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # 5. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    @staticmethod
    def _ocr_with_configs(processed_img):
        """Try OCR with different configurations"""
        configs = [
            "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--oem 3 --psm 5 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ]
        
        best_text = ""
        for i, config in enumerate(configs):
            text = pytesseract.image_to_string(processed_img, config=config)
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            print(f"🔍 OCR attempt {i+1} with config '{config}': '{text}' -> '{cleaned}'")
            
            if 4 <= len(cleaned) <= 8 and len(cleaned) > len(best_text):
                best_text = cleaned
        
        return best_text
    
    @staticmethod
    def _try_rotation_approaches(image):
        """Try different rotation approaches for difficult CAPTCHAs"""
        print("🔄 Trying manual rotation approaches...")
        img_alt = np.array(image.convert("L"))
        
        # Try different rotations
        for angle in [0, 90, -90, 180]:
            if angle != 0:
                rotated_img = Image.fromarray(img_alt).rotate(angle, expand=True)
                img_rotated = np.array(rotated_img)
            else:
                img_rotated = img_alt.copy()
            
            img_rotated = cv2.resize(img_rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, img_rotated = cv2.threshold(img_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            processed_img_alt = Image.fromarray(img_rotated)
            text_alt = pytesseract.image_to_string(
                processed_img_alt, 
                config="--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            cleaned_alt = re.sub(r'[^A-Z0-9]', '', text_alt.upper())
            
            print(f"🔄 Rotation {angle}°: '{cleaned_alt}'")
            
            if 4 <= len(cleaned_alt) <= 8:
                print(f"✓ CAPTCHA solved (rotation {angle}°): {cleaned_alt}")
                return cleaned_alt
        
        return None


class APILoginHandler:
    """Handles API-based login operations"""
    
    def __init__(self, session):
        self.session = session
    
    def generate_captcha_id(self):
        """Return captcha_id string or None."""
        ts = int(time.time() * 1000)
        url = f"{GENERATE_CAPTCHA_URL}{ts}"
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            # Check several possible locations for id
            captcha_id = (
                data.get('data', {}).get('id') or
                data.get('id') or
                data.get('captcha_id') or
                (data.get('result') or {}).get('id')
            )
            
            if captcha_id:
                print("✅ captcha_id:", captcha_id)
                return captcha_id
                
            print("❌ generate_captcha returned no id:", data)
        except Exception as e:
            print("❌ generate_captcha_id error:", e)
        
        return None
    
    def get_captcha_image_by_id(self, captcha_id):
        """Download captcha image for given id. Returns PIL Image or None."""
        if not captcha_id:
            return None
            
        url = f"{GET_CAPTCHA_URL}{captcha_id}"
        headers = {
            "Referer": "https://mlive.minemedia.tv/",
            "User-Agent": self.session.headers.get("User-Agent")
        }
        
        try:
            r = self.session.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content))
            print("✅ downloaded captcha image, size:", img.size)
            return img
        except Exception as e:
            print("❌ get_captcha_image_by_id error:", e)
        
        return None
    
    def login_with_captcha(self, username, password, captcha_code, captcha_id):
        """Post login as form-urlencoded (what the server expects)."""
        url = f"{LOGIN_API_URL}{username}?cptc=2"
        payload = {
            "loginname": username,
            "password": password,
            "cap_code": captcha_code,
            "id": captcha_id
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://mlive.minemedia.tv",
            "Referer": "https://mlive.minemedia.tv/",
            "User-Agent": self.session.headers.get("User-Agent")
        }
        
        try:
            r = self.session.post(url, data=payload, headers=headers, timeout=10)
            print("🔐 Login response:", r.status_code, r.text)
            
            # ✅ Get token cookie
            if "token" in r.cookies:
                token_val = r.cookies.get("token")
                self.session.cookies.set("token", token_val, domain="mlive.minemedia.tv")
                print("✅ Got session token:", token_val)
            
            # Try to parse JSON if available
            try:
                return r.status_code, r.json()
            except:
                return r.status_code, r.text
        except Exception as e:
            print("❌ login_with_captcha error:", e)
            return None, None
    
    def validate_login(self, username):
        """Validate login by checking device list"""
        url = f"https://mlive.minemedia.tv/v3/users/device_list/{username}"
        try:
            r = self.session.post(url, data={}, timeout=10)
            data = r.json()
            if data.get("error") == 0:
                print("✅ Login validated, device list:", data)
                return True
        except Exception as e:
            print(f"❌ Validation failed: {e}")
        return False
    
    def perform_login_once(self):
        """1 attempt: generate captcha -> download -> OCR -> login. Returns True/False."""
        captcha_id = self.generate_captcha_id()
        if not captcha_id:
            return False

        img = self.get_captcha_image_by_id(captcha_id)
        if not img:
            return False

        code = CaptchaSolver.solve_captcha(img)
        if not code:
            print("❌ OCR gagal, skip")
            return False

        status, resp = self.login_with_captcha(USERNAME, PASSWORD, code, captcha_id)
        
        # Check server response: success usually has code 200 or key `code`==200
        if isinstance(resp, dict) and (resp.get("code") == 200 or resp.get("error") in (None, 0)):
            print("✅ Login berhasil (API).")
            return self.validate_login(USERNAME)
        else:
            print("❌ Login API gagal:", resp)
            return False
    
    def perform_login(self, retries=5):
        """Perform login with multiple retries"""
        for i in range(retries):
            print(f"\nAttempt {i+1}/{retries}")
            if self.perform_login_once():
                return True
            time.sleep(1.2)
        return False


class SeleniumLoginHandler:
    """Handles Selenium-based login operations"""
    
    @staticmethod
    def setup_driver(headless=True):
        """Setup Chrome WebDriver with appropriate options"""
        opts = Options()
        if headless: 
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1200,1000")  # Increased height for vertical CAPTCHAs
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=opts)
    
    @staticmethod
    def setup_visible_driver():
        """Setup visible Chrome WebDriver"""
        opts = Options()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1200,1000")  # Increased height for vertical CAPTCHAs
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_experimental_option("detach", True)
        opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=opts)
    
    @staticmethod
    def get_full_captcha_screenshot(driver, element):
        """Get full CAPTCHA screenshot even if it's vertical"""
        try:
            # Scroll to element
            driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(1)
            
            # Get element location and size
            location = element.location
            size = element.size
            
            # Take full page screenshot
            driver.save_screenshot("full_page.png")
            full_img = Image.open("full_page.png")
            
            # Calculate cropping coordinates
            left = location['x']
            top = location['y']
            right = left + size['width']
            bottom = top + size['height']
            
            # Add margin to capture full vertical CAPTCHA
            margin = 50
            top = max(0, top - margin)
            bottom = min(full_img.height, bottom + margin)
            
            # Crop CAPTCHA area
            captcha_img = full_img.crop((left, top, right, bottom))
            captcha_img.save(f"full_captcha_{int(time.time())}.png")
            
            return captcha_img
            
        except Exception as e:
            print(f"❌ Error taking full screenshot: {e}")
            # Fallback to element screenshot
            return Image.open(BytesIO(element.screenshot_as_png))
    
    @staticmethod
    def perform_selenium_login(driver):
        """Perform login using Selenium with improved CAPTCHA handling"""
        print("🔄 Falling back to Selenium login...")
        
        driver.get(LOGIN_URL)
        time.sleep(3)
        
        # Get CAPTCHA element
        captcha_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='get_captcha']")
        if not captcha_elements:
            print("❌ No CAPTCHA element found")
            return False
            
        captcha_element = captcha_elements[0]
        
        # Get CAPTCHA image with improved method
        captcha_image = SeleniumLoginHandler.get_full_captcha_screenshot(driver, captcha_element)
        captcha_image.save(f"selenium_captcha_full_{int(time.time())}.png")
        
        # Solve CAPTCHA
        captcha_code = CaptchaSolver.solve_captcha(captcha_image)
        if not captcha_code:
            print("❌ Failed to solve CAPTCHA")
            return False
        
        # Fill login form
        try:
            username_field = driver.find_element(By.NAME, "loginname")
            password_field = driver.find_element(By.NAME, "password")
            captcha_field = driver.find_element(By.NAME, "cap_code")
            
            username_field.clear()
            username_field.send_keys(USERNAME)
            password_field.clear()
            password_field.send_keys(PASSWORD)
            captcha_field.clear()
            captcha_field.send_keys(captcha_code)
            
            print("✅ Form filled")
        except Exception as e:
            print(f"❌ Error filling form: {e}")
            return False
        
        # Submit form
        try:
            submit_buttons = driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
            if submit_buttons:
                submit_buttons[0].click()
                print("✅ Form submitted")
                
                # Wait for login result
                time.sleep(5)
                
                # Check if login was successful
                current_url = driver.current_url
                if "device/list" in current_url or "device_list" in current_url:
                    print("✅ Login successful!")
                    return True
                else:
                    print("❌ Login failed - redirected to wrong page")
                    return False
        except Exception as e:
            print(f"❌ Error submitting form: {e}")
        
        return False


def main():
    """Main function to handle login process"""
    # First try API login
    print("🔐 Attempting API login...")
    api_handler = APILoginHandler(session)
    api_success = api_handler.perform_login(retries=MAX_ATTEMPTS)
    
    if api_success:
        print("✅ API login successful!")
        return
    
    # If API fails, try Selenium
    print("❌ API login failed, trying Selenium...")
    
    driver = SeleniumLoginHandler.setup_visible_driver()
    try:
        if SeleniumLoginHandler.perform_selenium_login(driver):
            print("✅ Selenium login successful!")
            print("🌐 You are now logged in and can navigate freely")
            
            # Keep browser open
            input("Press Enter to close the browser...")
        else:
            print("❌ All login methods failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
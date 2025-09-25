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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cv2
import numpy as np
import requests
import browser_cookie3
from urllib.parse import urlparse, parse_qs

# --- Config ---
LOGIN_URL = "https://mlive.minemedia.tv/"
USERNAME = "99988805682"
PASSWORD = "92076958"

# Setup driver untuk monitoring network requests
def setup_driver_for_monitoring():
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1200,1000")
    opts.add_argument("--auto-open-devtools-for-tabs")  # Buka DevTools otomatis
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    
    # Enable performance logging untuk capture network requests
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    
    driver = webdriver.Chrome(options=opts)
    return driver

def analyze_network_requests():
    """Analisis semua network requests selama proses login"""
    driver = setup_driver_for_monitoring()
    
    try:
        print("🌐 Membuka halaman login dan memantau network requests...")
        driver.get(LOGIN_URL)
        
        # Tunggu hingga halaman loaded
        time.sleep(5)
        
        # Isi form login
        username_field = driver.find_element(By.NAME, "loginname")
        password_field = driver.find_element(By.NAME, "password")
        
        username_field.clear()
        username_field.send_keys(USERNAME)
        password_field.clear()
        password_field.send_keys(PASSWORD)
        
        print("✅ Form login diisi, silakan lengkapi CAPTCHA manual dan submit")
        print("🔍 Memantau semua network requests...")
        
        # Biarkan user menyelesaikan CAPTCHA manual dan submit
        input("Setelah login berhasil, tekan Enter untuk menganalisis requests...")
        
        # Dapatkan semua network logs
        logs = driver.get_log("performance")
        
        # Analisis logs untuk mencari token-related requests
        token_requests = []
        for log in logs:
            try:
                message = json.loads(log["message"])["message"]
                if message["method"] == "Network.requestWillBeSent":
                    url = message["params"]["request"]["url"]
                    
                    # Cari requests yang mungkin berhubungan dengan token
                    if any(keyword in url for keyword in ["token", "auth", "login", "session", "validate"]):
                        token_requests.append({
                            "url": url,
                            "method": message["params"]["request"]["method"],
                            "headers": message["params"]["request"]["headers"]
                        })
                    
                    # Juga cari requests dengan cookies
                    if "headers" in message["params"]["request"]:
                        headers = message["params"]["request"]["headers"]
                        if "cookie" in headers and "token" in headers["cookie"].lower():
                            token_requests.append({
                                "type": "cookie_request",
                                "url": url,
                                "headers": headers
                            })
            except:
                continue
        
        print(f"🔍 Ditemukan {len(token_requests)} requests yang berpotensi terkait token:")
        for i, req in enumerate(token_requests):
            print(f"\nRequest #{i+1}:")
            print(f"URL: {req['url']}")
            if "method" in req:
                print(f"Method: {req['method']}")
            
        return token_requests
        
    except Exception as e:
        print(f"❌ Error selama monitoring: {e}")
        return []
    finally:
        driver.quit()

def intercept_token_creation():
    """Coba intercept pembuatan token dengan proxy"""
    print("🔧 Mencoba intercept token creation...")
    
    # Gunakan Selenium dengan extension untuk intercept requests
    opts = Options()
    opts.add_extension('proxy.crx')  # Anda perlu download extension proxy terlebih dahulu
    
    driver = webdriver.Chrome(options=opts)
    
    try:
        driver.get(LOGIN_URL)
        
        # Biarkan user login manual
        print("Silakan login manual di browser yang terbuka...")
        input("Setelah login berhasil, tekan Enter untuk melanjutkan...")
        
        # Dapatkan cookies setelah login
        cookies = driver.get_cookies()
        token_cookie = None
        
        for cookie in cookies:
            if 'token' in cookie['name'].lower():
                token_cookie = cookie
                print(f"✅ Token ditemukan dalam cookie: {cookie['name']} = {cookie['value']}")
                break
        
        if token_cookie:
            # Coba analisis struktur token
            analyze_token_structure(token_cookie['value'])
            
        return token_cookie
        
    except Exception as e:
        print(f"❌ Error selama intercept: {e}")
        return None
    finally:
        driver.quit()

def analyze_token_structure(token):
    """Analisis struktur token untuk memahami bagaimana dibuat"""
    print(f"🔍 Menganalisis struktur token: {token}")
    
    # Coba decode sebagai JWT (jika menggunakan format JWT)
    if len(token.split('.')) == 3:
        print("📝 Token terdeteksi sebagai JWT (3 bagian dipisahkan titik)")
        
        try:
            # Decode header
            import base64
            header_encoded = token.split('.')[0]
            header_encoded += '=' * (-len(header_encoded) % 4)  # Padding
            header_decoded = base64.urlsafe_b64decode(header_encoded)
            print(f"JWT Header: {header_decoded}")
            
            # Decode payload
            payload_encoded = token.split('.')[1]
            payload_encoded += '=' * (-len(payload_encoded) % 4)  # Padding
            payload_decoded = base64.urlsafe_b64decode(payload_encoded)
            print(f"JWT Payload: {payload_decoded}")
        except:
            print("Token bukan JWT standar atau tidak bisa decode")
    
    # Analisis panjang dan pattern
    print(f"📏 Panjang token: {len(token)} karakter")
    
    # Cek karakter khusus
    import string
    allowed_chars = set(string.ascii_letters + string.digits + '-_=+.')
    token_chars = set(token)
    unusual_chars = token_chars - allowed_chars
    
    if unusual_chars:
        print(f"⚠️  Karakter tidak biasa dalam token: {unusual_chars}")
    else:
        print("✅ Hanya karakter alfanumerik dan simbol standar")

def check_local_storage():
    """Cek local storage dan session storage untuk token"""
    print("🔍 Memeriksa local storage dan session storage...")
    
    driver = webdriver.Chrome()
    
    try:
        driver.get(LOGIN_URL)
        
        # Biarkan user login manual
        print("Silakan login manual di browser yang terbuka...")
        input("Setelah login berhasil, tekan Enter untuk memeriksa storage...")
        
        # Check local storage
        local_storage = driver.execute_script("return window.localStorage;")
        print("Local Storage:", local_storage)
        
        # Check session storage
        session_storage = driver.execute_script("return window.sessionStorage;")
        print("Session Storage:", session_storage)
        
        # Cari token di storage
        for key, value in local_storage.items():
            if 'token' in key.lower():
                print(f"✅ Token ditemukan di local storage: {key} = {value}")
                analyze_token_structure(value)
                
        for key, value in session_storage.items():
            if 'token' in key.lower():
                print(f"✅ Token ditemukan di session storage: {key} = {value}")
                analyze_token_structure(value)
                
    except Exception as e:
        print(f"❌ Error memeriksa storage: {e}")
    finally:
        driver.quit()

def main_reverse_engineering():
    """Fungsi utama untuk reverse engineering token"""
    print("🕵️  Memulai reverse engineering token...")
    print("Pilih metode:")
    print("1. Analisis network requests")
    print("2. Intercept token creation")
    print("3. Cek local/session storage")
    
    choice = input("Masukkan pilihan (1-3): ")
    
    if choice == "1":
        analyze_network_requests()
    elif choice == "2":
        intercept_token_creation()
    elif choice == "3":
        check_local_storage()
    else:
        print("❌ Pilihan tidak valid")

if __name__ == "__main__":
    main_reverse_engineering()
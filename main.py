#!/usr/bin/env python3
import time
import os
import re
import glob
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# --- Konfigurasi ---
LOGIN_URL = "https://mlive.minemedia.tv/login"
USERNAME = "99988805682"
PASSWORD = "92076958"
MAX_ATTEMPTS = 20
CAPTCHA_FOLDER = "captchas"

KEEP_OPEN_ON_SUCCESS = True
LEAVE_ORPHAN = False

# --- OCR preprocessing ---
def preprocess_image(image: Image.Image) -> Image.Image:
    """Melakukan preprocessing pada gambar untuk meningkatkan akurasi OCR."""
    img = image.convert("L")
    img = ImageOps.autocontrast(img)
    base_w, base_h = img.size
    img = img.resize((base_w * 2, base_h * 2), resample=Image.NEAREST)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.point(lambda p: 255 if p > 160 else 0)
    return img

def solve_captcha(image: Image.Image) -> str | None:
    """Memecahkan CAPTCHA menggunakan Tesseract OCR."""
    proc = preprocess_image(image)
    custom_config = r"-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8"
    try:
        text = pytesseract.image_to_string(proc, config=custom_config)
        text = re.sub(r'[^A-Za-z0-9]', '', text.strip())
        return text if text else None
    except Exception as e:
        print(f"OCR error: {e}")
        return None

# --- Setup WebDriver ---
def setup_driver():
    """Menginisialisasi WebDriver Chrome."""
    chrome_options = Options()
    # kalau mau headless: chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1280,720")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# --- Helper login ---
def is_login_successful(driver, timeout=8) -> bool:
    """Cek apakah login berhasil (ganti selector sesuai kondisi real)."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".navbar, .user-profile"))
        )
        return True
    except TimeoutException:
        return False

# --- Fungsi Utama ---
def main():
    driver = setup_driver()
    try:
        os.makedirs(CAPTCHA_FOLDER, exist_ok=True)

        existing_files = glob.glob(os.path.join(CAPTCHA_FOLDER, "*.jpg"))
        count = len(existing_files)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n=== Percobaan {attempt}/{MAX_ATTEMPTS} ===")
            driver.get(LOGIN_URL)

            try:
                # 1. Tunggu elemen CAPTCHA muncul
                print("Menunggu elemen CAPTCHA...")
                captcha_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'img[src*="get_captcha"]'))
                )

                # 2. Tunggu atribut 'src' selesai dimuat
                print("Menunggu URL CAPTCHA selesai dimuat...")
                WebDriverWait(driver, 10).until(
                    lambda d: captcha_element.get_attribute("src") and "get_captcha" in captcha_element.get_attribute("src")
                )

                # Screenshot CAPTCHA
                png = captcha_element.screenshot_as_png
                img = Image.open(BytesIO(png))
                captcha_code = solve_captcha(img)

                if not captcha_code:
                    print("OCR gagal membaca CAPTCHA, mencoba lagi...")
                    continue

                print(f"Kode CAPTCHA yang dibaca: {captcha_code}")

                # Simpan gambar CAPTCHA
                count += 1
                filename = f"{count:04d}_{captcha_code}.jpg"
                filepath = os.path.join(CAPTCHA_FOLDER, filename)
                img.save(filepath, "JPEG")
                print(f"Gambar disimpan di: {filepath}")

                # 3. Isi formulir dan login
                driver.find_element(By.NAME, "loginname").send_keys(USERNAME)
                driver.find_element(By.NAME, "password").send_keys(PASSWORD)
                driver.find_element(By.NAME, "cap_code").send_keys(captcha_code)

                # Klik tombol login
                driver.find_element(By.CSS_SELECTOR, "button.button.is-info").click()

                # Cek apakah login berhasil
                if is_login_successful(driver, timeout=8):
                    print("🎉 Login BERHASIL!")
                    break
                else:
                    print("Login GAGAL. Ulangi dari awal dengan klik captcha baru.")
                    try:
                        # kosongkan field
                        driver.find_element(By.NAME, "loginname").clear()
                        driver.find_element(By.NAME, "password").clear()
                        driver.find_element(By.NAME, "cap_code").clear()

                        # klik gambar captcha biar regenerate
                        captcha_element = driver.find_element(By.CSS_SELECTOR, 'img[src*="get_captcha"]')
                        captcha_element.click()
                        time.sleep(1)
                    except Exception as e:
                        print(f"Gagal reset form/captcha: {e}")
                        driver.refresh()
                        time.sleep(1.5)
                        continue

            except Exception as e:
                print(f"Error saat percobaan login: {e}")
                driver.refresh()
                time.sleep(1.5)
                continue

    except Exception as e:
        print(f"Error fatal: {e}")

    finally:
        if not is_login_successful(driver):
            driver.quit()
            print("Sesi otomasi selesai.")
        else:
            print("Browser dibiarkan terbuka (login berhasil).")

if __name__ == "__main__":
    main()

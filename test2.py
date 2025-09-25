#!/usr/bin/env python3
"""
mlive_bootstrap.py

Goals:
 - Bootstrap a session for mlive.minemedia.tv with minimal manual captcha solves.
 - After initial bootstrap (one captcha solve), subsequent runs can refresh `access_token` using the same `bid` without captcha.
 - Optionally open a real browser (Selenium) preloaded with cookies + localStorage so you can use the web UI seamlessly.

Requirements:
 pip install requests pillow pytesseract selenium opencv-python

Also install chromedriver matching Chrome if you use --open.

Usage:
  # bootstrap (tries OCR first, falls back to manual input)
  python mlive_bootstrap.py --bootstrap

  # refresh only (no captcha)
  python mlive_bootstrap.py --refresh

  # open browser using saved session
  python mlive_bootstrap.py --open

  # do all: refresh if possible, otherwise bootstrap
  python mlive_bootstrap.py

Notes:
 - Session is saved to session.json in current folder.
 - All values (USERNAME/PASSWORD) are configurable below.
"""

import os
import sys
import time
import json
import random
import string
import requests
from io import BytesIO
from PIL import Image
import re

# Try import OCR libs; if not available we'll fallback to manual
try:
    import pytesseract
    import cv2
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Selenium for opening browser with session (optional)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except Exception:
    webdriver = None

# ---------- CONFIG ----------
BASE = "https://mlive.minemedia.tv"
VITE_API_KEY = "65665c6cc310aa782a4fffaf01863d4e"

USERNAME = "99988805682"
PASSWORD = "92076958"

WEB_TOKEN = BASE + "/v3/access/web_token"
GEN_CAPTCHA = BASE + "/v3/util/generate_captcha?cptc=2&t="
GET_CAPTCHA = BASE + "/v3/util/get_captcha?cptc=2&id="
WEB_LOGIN = BASE + "/v3/users/web_login/{}"   # append username? will add ?cptc=2 later
DEVICE_LIST = BASE + "/v3/users/device_list/{}"

SESSION_FILE = "session.json"
CAPTCHA_FILE = "captcha.png"

REQUEST_TIMEOUT = 12.0

# ---------- helpers ----------
def zw(e):
    """Same as frontend zw(16) - random alphanumeric string"""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(e))

def save_session(data):
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved session -> {SESSION_FILE}")

def load_session():
    if not os.path.exists(SESSION_FILE):
        return None
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def pretty_json(x):
    try:
        return json.dumps(x, indent=2, ensure_ascii=False)
    except:
        return str(x)

# ---------- Captcha OCR (best-effort) ----------
def preprocess_for_ocr(pil_img):
    # convert to grayscale numpy
    arr = cv2.cvtColor(cv2.UMat(cv2.cvtColor(cv2.UMat(cv2.imread(CAPTCHA_FILE)) if False else cv2.UMat(cv2.cvtColor(cv2.UMat(np.array(pil_img)), cv2.COLOR_RGB2BGR).get())), cv2.COLOR_BGR2GRAY).get(), cv2.COLOR_GRAY2BGR)  # dummy fallback to avoid undefined np if not imported
    # simpler: just convert using PIL -> cv2 as fallback if above fails
    try:
        import numpy as np
        img = np.array(pil_img.convert("L"))
        h, w = img.shape
        scale = 2 if max(h,w) < 300 else 1
        if scale != 1:
            img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2,2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        kernel_sharp = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel_sharp)
        return img
    except Exception:
        # fallback: return pil image converted to grayscale
        return pil_img.convert("L")

def try_ocr(pil_img):
    """Return cleaned uppercase alnum string or None"""
    if not OCR_AVAILABLE:
        return None
    try:
        # minimal preprocessing
        img = pil_img.convert("L")
        import numpy as np
        arr = np.array(img)
        arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proc = Image.fromarray(arr)
        configs = [
            "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ]
        best = ""
        for cfg in configs:
            txt = pytesseract.image_to_string(proc, config=cfg)
            cleaned = re.sub(r'[^A-Z0-9]', '', txt.upper())
            # debug print suppressed
            if len(cleaned) >= 4 and len(cleaned) > len(best):
                best = cleaned
        if best:
            return best
    except Exception:
        pass
    return None

# ---------- Core flows ----------
def get_web_token(session_requests, bid):
    """POST /v3/access/web_token with key+bid -> returns access_token or None"""
    try:
        data = {"key": VITE_API_KEY, "bid": bid}
        r = session_requests.post(WEB_TOKEN, data=data, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        token = (j.get("data") or {}).get("access_token") or j.get("access_token")
        return token, j
    except Exception as e:
        print("❌ get_web_token failed:", e)
        return None, None

def generate_captcha(session_requests):
    try:
        ts = str(int(time.time()*1000))
        r = session_requests.get(GEN_CAPTCHA + ts, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        cid = (j.get("data") or {}).get("id") or j.get("id") or j.get("captcha_id") or (j.get("result") or {}).get("id")
        return cid, j
    except Exception as e:
        print("❌ generate_captcha failed:", e)
        return None, None

def download_captcha(session_requests, captcha_id, out_path=CAPTCHA_FILE):
    try:
        r = session_requests.get(GET_CAPTCHA + captcha_id, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        pil = Image.open(out_path).convert("RGB")
        print(f"✅ Captcha image saved -> {out_path} (size: {pil.size})")
        return pil
    except Exception as e:
        print("❌ download captcha failed:", e)
        return None

def do_web_login(session_requests, username, password, captcha_id, captcha_code, access_token):
    """
    Attempt login via POST /v3/users/web_login/<username>?cptc=2
    We'll send fields that frontend/curl examples used: key, access_token, password, id, code,
    and include loginname/cap_code (some server code expects those names).
    """
    url = WEB_LOGIN.format(username) + "?cptc=2"
    payload = {
        "key": VITE_API_KEY,
        "access_token": access_token or "",
        "password": password,
        "id": captcha_id,
        "code": captcha_code,
        # some variants
        "loginname": username,
        "cap_code": captcha_code
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Origin": BASE,
        "Referer": BASE + "/",
        "User-Agent": session_requests.headers.get("User-Agent")
    }
    try:
        r = session_requests.post(url, data=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        try:
            j = r.json()
        except:
            j = None
        print("🔐 web_login response (status):", r.status_code)
        return r, j
    except Exception as e:
        print("❌ web_login request failed:", e)
        return None, None

def validate_device_list(session_requests, username):
    try:
        url = DEVICE_LIST.format(username)
        data = {"key": VITE_API_KEY, "access_token": session_requests.cookies.get("token") or ""}
        r = session_requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        return j
    except Exception as e:
        print("❌ device_list check failed:", e)
        return None

# ---------- Utilities to open browser with session ----------
def launch_browser_with_session(session_data, headless=False):
    if webdriver is None:
        print("❌ Selenium not installed. Install selenium and chromedriver to use --open.")
        return False

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1200,1000")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=opts)
    try:
        # open blank page on domain to set cookies
        driver.get(BASE)
        time.sleep(1)
        # set cookies
        cookies = session_data.get("cookies") or {}
        for name, value in cookies.items():
            try:
                driver.add_cookie({"name": name, "value": value, "domain": ".minemedia.tv", "path": "/"})
            except Exception:
                try:
                    driver.add_cookie({"name": name, "value": value})
                except Exception:
                    pass
        # set localStorage items if present
        ls = session_data.get("localStorage") or {}
        for k, v in ls.items():
            try:
                driver.execute_script("localStorage.setItem(arguments[0], arguments[1]);", k, v)
            except Exception:
                pass
        # reload to let site pick up session
        driver.get(BASE + "/device/list/")
        time.sleep(3)
        print("✅ Browser launched. Check if you're logged in.")
        return True
    except Exception as e:
        print("❌ Error launching browser:", e)
        driver.quit()
        return False

# ---------- High-level flows ----------
def bootstrap_session():
    """Full bootstrap: generate bid, get web_token, download captcha, solve, login, save session.json"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "application/json, text/plain, */*",
    })

    # 1) bid - reuse existing if session file present
    session_data = load_session()
    bid = None
    if session_data and session_data.get("bid"):
        bid = session_data.get("bid")
        print("Reusing saved bid:", bid)
    else:
        bid = zw(16)
        print("Generated new bid:", bid)

    # 2) get web_token (server issues access_token)
    access_token, resp = get_web_token(s, bid)
    if not access_token:
        # the server might have set cookie token instead; check cookies
        access_token = s.cookies.get("token")
    print("access_token (initial):", access_token)
    # still continue; some servers require login after getting token

    # 3) captcha
    cid, gresp = generate_captcha(s)
    if not cid:
        print("❌ cannot obtain captcha id - abort.")
        return False
    pil = download_captcha(s, cid)
    if pil is None:
        print("❌ cannot download captcha image - abort.")
        return False

    # 4) attempt OCR
    code = None
    if OCR_AVAILABLE:
        try:
            print("Trying OCR on captcha...")
            code = try_ocr(pil)
            if code:
                print("OCR thinks captcha is:", code)
            else:
                print("OCR failed to reliably read captcha.")
        except Exception as e:
            print("OCR error:", e)

    if not code:
        # fallback to manual input
        print(f"Please open '{CAPTCHA_FILE}' and type the captcha result here (or press Enter to abort):")
        print(f"File path: {os.path.abspath(CAPTCHA_FILE)}")
        code = input("Captcha: ").strip()
        if not code:
            print("Aborted by user.")
            return False

    # 5) login
    r, j = do_web_login(s, USERNAME, PASSWORD, cid, code, access_token)
    if r is None:
        print("Login request failed.")
        return False

    # print response snippet
    try:
        print("Login response JSON:", pretty_json(j))
    except Exception:
        print("Login response text:", r.text[:400])

    # 6) check response for success - several heuristics
    success = False
    if isinstance(j, dict):
        if j.get("error") == 0 or j.get("code") == 200 or ("user" in j and j["user"]):
            success = True

    # server may also set cookie 'token'; capture
    cookie_token = s.cookies.get("token")
    if cookie_token:
        print("Received cookie token:", cookie_token)

    # If server responded success but device_list still required, try device_list
    dev = validate_device_list(s, USERNAME)
    if dev and isinstance(dev, dict) and dev.get("error") == 0:
        print("Device list OK.")
        success = True

    if not success:
        print("⚠️ Login not validated. You can inspect the outputs above.")
        # still save partial session if cookie exists? we'll not save if not successful
        return False

    # 7) assemble session data to save
    saved = {
        "bid": bid,
        "access_token": access_token or cookie_token or "",
        "cookies": {c.name: c.value for c in s.cookies},
        "localStorage": {
            # web client stores access_token and possibly login_expires/user - we only persist what we know
            "access_token": access_token or cookie_token or "",
        },
        # if login response included user/login_expires put them here
    }

    # try to extract more fields from login json
    if isinstance(j, dict):
        user = j.get("user") or j.get("data") or {}
        if isinstance(user, dict):
            saved["user"] = user
            # login expiry might be in response fields
            if user.get("regtime"):
                saved["user_regtime"] = user.get("regtime")
        # some responses have login_expires or expires
        if j.get("login_expires"):
            saved["login_expires"] = j.get("login_expires")
        elif j.get("expires"):
            saved["login_expires"] = j.get("expires")

    save_session(saved)
    print("✅ Bootstrap successful. Session saved. You can now run --open to launch browser with session.")
    return True

def refresh_access_token():
    """Use saved bid to request a fresh access_token (no captcha) and update session.json"""
    session_data = load_session()
    if not session_data or not session_data.get("bid"):
        print("❌ No saved session/bid found. Run --bootstrap first.")
        return False
    bid = session_data["bid"]
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    token, resp = get_web_token(s, bid)
    if not token:
        print("❌ Could not refresh token (server returned nothing).")
        return False
    # update session data
    session_data["access_token"] = token
    session_data.setdefault("localStorage", {})["access_token"] = token
    # also update cookies from response session (if server set one)
    for k,v in s.cookies.get_dict().items():
        session_data.setdefault("cookies", {})[k] = v
    save_session(session_data)
    print("✅ access_token refreshed.")
    return True

# ---------- CLI ----------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", action="store_true", help="Run full bootstrap (captcha solve required).")
    p.add_argument("--refresh", action="store_true", help="Refresh access_token using saved bid (no captcha).")
    p.add_argument("--open", action="store_true", help="Open browser with saved session (requires selenium).")
    p.add_argument("--headless", action="store_true", help="Open browser headless (only with --open).")
    args = p.parse_args()

    # default behavior: try refresh, if fails then bootstrap
    if not any([args.bootstrap, args.refresh, args.open]):
        print("No flags, trying refresh first; if that fails will run bootstrap.")
        ok = refresh_access_token()
        if not ok:
            print("Refresh failed - running bootstrap.")
            ok = bootstrap_session()
            if not ok:
                print("Bootstrap failed. Exiting.")
                sys.exit(1)
    else:
        if args.refresh:
            ok = refresh_access_token()
            if not ok:
                sys.exit(1)
        if args.bootstrap:
            ok = bootstrap_session()
            if not ok:
                sys.exit(1)
        if args.open:
            session_data = load_session()
            if not session_data:
                print("No session.json found. Run --bootstrap first.")
                sys.exit(1)
            launched = launch_browser_with_session(session_data, headless=args.headless)
            if not launched:
                sys.exit(1)

if __name__ == "__main__":
    main()

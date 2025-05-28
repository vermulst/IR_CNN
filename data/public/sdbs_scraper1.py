"""Download all IR spectra and structure files available from SDBS."""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import re
import os
import shutil
import requests  # Added for direct downloads

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
other_path = os.path.join(BASE_DIR, '../sdbs_dataset/other/')
gif_path = os.path.join(BASE_DIR, '../sdbs_dataset/gif/')
ids_path = os.path.join(BASE_DIR, '../sdbs_dataset/sdbs_ids.txt')
down_path = os.path.expanduser('~/Downloads')  # Platform-independent downloads path

# Define URLs
disclaimer_url = 'https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_disclaimer.cgi'
main_url = 'https://sdbs.db.aist.go.jp/sdbs/cgi-bin/landingpage?sdbsno='

# Regex patterns (optimized)
patterns = {
    'formula': re.compile(r'Molecular Formula:\s+(.*?)\s*$'),
    'mw': re.compile(r'Molecular Weight:\s+(.*?)\s*$'),
    'inchikey': re.compile(r'InChIKey:\s+(.*?)\s*$'),
    'cas': re.compile(r'RN:\s+(.*?)\s*$'),
    'name': re.compile(r'Compound Name:\s+(.*?)(?=\s*Description:|$)'),
    'inchi': re.compile(r'InChI=.*$')
}

def check_dir():
    """Initialize directories and load SDBS IDs"""
    os.makedirs(other_path, exist_ok=True)
    os.makedirs(gif_path, exist_ok=True)
    
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Create {ids_path} with SDBS IDs (one per line)")
    
    with open(ids_path) as f:
        return [line.strip() for line in f if line.strip()]

def safe_click(driver, xpath, timeout=10):
    """Click element with JS fallback"""
    element = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    try:
        element.click()
    except:
        driver.execute_script("arguments[0].click();", element)

def handle_disclaimer(driver):
    """Handle disclaimer page with robust element interaction"""
    driver.get(disclaimer_url)
    safe_click(driver, '//input[@value="I agree the disclaimer and use SDBS."]')
    driver.switch_to.window(driver.window_handles[0])

def download_spectrum(url, save_path):
    """Direct download without pyautogui"""
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    ids = check_dir()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        handle_disclaimer(driver)
        errors = []

        for idx, sdbs_id in enumerate(ids, 1):
            try:
                print(f"\nProcessing ID: {sdbs_id} ({idx}/{len(ids)})")
                
                # Refresh session every 30 items
                if idx % 30 == 0:
                    handle_disclaimer(driver)

                # Navigate to compound page
                driver.get(f"{main_url}{sdbs_id}")
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//table'))
                )

                # Process textual data
                other_file = os.path.join(other_path, f'{sdbs_id}_other.txt')
                if not os.path.exists(other_file):
                    with open(other_file, 'w') as f:
                        for row in driver.find_elements(By.XPATH, '//tr'):
                            text = row.text.replace('\n', ': ')
                            for key, pattern in patterns.items():
                                match = pattern.search(text)
                                if match:
                                    f.write(f"{key.capitalize()}: {match.group(1)}\n")

                # Process IR spectra
                for link in driver.find_elements(By.XPATH, '//a[contains(text(), "IR")]'):
                    spectrum_type = link.text.split(':')[1].strip()
                    gif_name = f"{sdbs_id}_{spectrum_type.replace(' ', '_')}.gif"
                    gif_path_full = os.path.join(gif_path, gif_name)

                    if not os.path.exists(gif_path_full):
                        link.click()
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, '//img[@alt="spectrum"]'))
                        )
                        img_url = driver.find_element(By.XPATH, '//img[@alt="spectrum"]').get_attribute('src')
                        download_spectrum(img_url, gif_path_full)
                        driver.back()

            except (NoSuchElementException, TimeoutException) as e:
                errors.append(sdbs_id)
                print(f"Error processing {sdbs_id}: {str(e)}")
                continue

        print(f"\nCompleted with {len(errors)} errors. Problem IDs: {errors}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()

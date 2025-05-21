"""Download all IR spectra and structure files available from SDBS."""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import re
import os
import shutil
import pyautogui


# Define paths.
other_path = 'sdbs_dataset/other/'
gif_path = 'sdbs_dataset/gif/'
ids_path = 'sdbs_dataset/sdbs_ids.txt'
down_path = '/Users/migue/Downloads/' # Change path to your Downloads folder.

# Define URLs for get requests.
disclaimer = 'https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_disclaimer.cgi?REQURL=/sdbs/cgi-bin/direct_frame_top.cgi&amp;REFURL='
main = 'https://sdbs.db.aist.go.jp/sdbs/cgi-bin/landingpage?sdbsno='
land = 'https://sdbs.db.aist.go.jp/SearchInformation.aspx'

# Define regex strings used to match.
formula = re.compile('Molecular Formula: (.*?)$')
mw = re.compile('Molecular Weight: (.*?)$')
#inchi = re.compile('InChI: (InChI=.*?)$')
#inchikey = re.compile('InChIKey: (.*?)$')
#cas = re.compile('RN: (.*?)$')
name = re.compile('Description: Compound Name: (.*?)$')


def check_dir():
    ids = []
    """Check if file directories exist and create them if they do not."""
    if not os.path.exists('../sdbs_dataset/'):
        os.makedirs('../sdbs_dataset')
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    if not os.path.exists(other_path):
        os.makedirs(other_path)
    if not os.path.exists(down_path):
        print('Please set the down_path on line 21 to your Downloads folder.')
    # Check if IDs file exists.
    if not os.path.exists(ids_path):
        print('sdbs_ids.txt does not exist')
    elif os.path.exists(ids_path):
        print('sdbs_ids.txt alrealdy exists')
        ids = [line.rstrip('\n') for line in open(ids_path)]
    return ids

# Check directories.
ids = check_dir()


# Install latest driver for Chrome.
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(disclaimer)

wait = WebDriverWait(driver, 10)
agree_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[contains(@value, "I agree")]')))
driver.execute_script("arguments[0].click();", agree_button)
#agree_button.click()

#driver.find_element("xpath", '/html/body/form/input').click()
driver.switch_to.window(driver.window_handles[0])


# Define variables.
errors = []
count = 0
# Loop through each ID in the list and download.
for j in ids[:]:
    print(errors)
    # Attempt to download information for selected ID.
    try:
        count += 1
        print('\nSession download count: %s' % count)
        driver.get(disclaimer)
        wait = WebDriverWait(driver, 10)
        agree_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[contains(@value, "I agree")]')))
        driver.execute_script("arguments[0].click();", agree_button)

            #driver.find_element("xpath", '/html/body/form/input').click()

        # After scraping 30 unique compounds, start a new browser.
        if count % 30 == 0:
            print('New disclaimer')
            driver.get(disclaimer)
            wait = WebDriverWait(driver, 10)
            agree_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[contains(@value, "I agree")]')))
            driver.execute_script("arguments[0].click();", agree_button)

            #driver.find_element("xpath", '/html/body/form/input').click()
            

        # Select first tab in the controlled window.
        #driver.switch_to.window(driver.window_handles[0])
        # Open profile of selected compound.
        #driver.get(land)
        wait = WebDriverWait(driver, 5)
        sdbs_input = driver.find_element(By.XPATH, '//input[contains(@name, "ctl00$BodyContentPlaceHolder$INP_sdbsno")]')
        sdbs_input.clear()
        sdbs_input.send_keys(str(j))
        
        # Click the Search button (adjust selector if needed)'
        wait = WebDriverWait(driver, 5)
        search_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[contains(@name, "SearchButton")]')))
        driver.execute_script("arguments[0].click();", search_button)
        #driver.get(main + str(j))
        
        # Create new file or skip if already exists.
        filepath = os.path.join(other_path, '%s_other.txt' % j)
        if not os.path.isfile(filepath):
            print('Downloading: %s_other.txt' % j)
            otherpath = os.path.join(other_path, '%s_other.txt' % j)
            file = open(otherpath, 'w+')
            # Grab texts with relevant information and write in file.
            formul = driver.find_element(By.XPATH, '//*[@id="pnlScroll"]/table/tbody/tr[3]/td[2]')
            file.write('Formula: '+str(formul.text)+'\n')
            wm = driver.find_element(By.XPATH, '//*[@id="pnlScroll"]/table/tbody/tr[3]/td[3]')
            file.write('Molecular Weight: '+str(wm.text)+'\n')
            name = driver.find_element(By.XPATH, '//*[@id="pnlScroll"]/table/tbody/tr[3]/td[10]')
            file.write('Name: '+str(name.text)+'\n')
            file.close()
        else:
            print('%s_other.txt already exists' % j)
        sleep(2)
        wait = WebDriverWait(driver, 10)
        yes_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="pnlScroll"]/table/tbody/tr[3]/td[7]/a')))
        driver.execute_script("arguments[0].click();", yes_button)
        sleep(2)
        wait = WebDriverWait(driver, 10)
        print("d")
        #irs = wait.until(EC.presence_of_all_elements_located( (By.XPATH, '//*[@id="CtlMasterSideMenu_SpectralLink"]')))
        new_list=[]
        counti=[]
        sleep(3)
        for i in range(1,10):
            try:
                irs = driver.find_element(By.XPATH, '//*[@id="CtlMasterSideMenu_SpectralLink"]/a['+str(i)+']')
                sleep(2)
                string = irs.text
                print(string)
                if 'IR' in string:
                    new_list.append(irs)
                    counti.append(i)
            except:
                continue
        sleep(3)
        count2=0
        for i in new_list:
            #if count2==len(counti):
            #    break
            ind = counti[count2]
            l = driver.find_element(By.XPATH, '//*[@id="CtlMasterSideMenu_SpectralLink"]/a['+str(ind)+']')
            sleep(5)
            l.click()
            sleep(5)
            l = driver.find_element(By.XPATH, '//*[@id="CtlMasterSideMenu_SpectralLink"]/a['+str(ind)+']') 
            sleep(10)
            method1 = l.text
            method =method1[5:]
            sleep(2)
            picpath = os.path.join(gif_path, str(j) + '_' + method + '.gif')
            # Check if spectrum exists otherwise download.
            count2+=1
            filename = str(j) + '_' + method + '.gif'
            if not os.path.isfile(picpath):
                print('Downloading: ' + filename)
                sleep(3)
                pyautogui.rightClick(x=814, y=504)
                sleep(1)
                pyautogui.press('down')
                sleep(1)
                pyautogui.press('down')
                sleep(1)
                pyautogui.press('enter')
                sleep(2)
                pyautogui.write(filename)
                pyautogui.hotkey('enter')
                sleep(5)
                # Move downloaded spectrum from Downloads folder to gif folder.
                #shutil.move(down_path + filename, picpath)
                print('File moved to gif folder')
                sleep(15)
            else:
                print(filename + '.gif already exists')
                #continue
                sleep(5)
                continue
                
            

    except TimeoutError:
        errors.append(j)
        print('Error IDs this session: ', errors)
        continue



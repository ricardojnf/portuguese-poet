from selenium import webdriver
import time

fil = open('lusiadas.txt', 'w')

driver = webdriver.Firefox()
driver.get('https://oslusiadas.org/i')
driver.implicitly_wait(10)

elements = driver.find_elements_by_xpath('/html/body/div[2]/div[2]/div[3]/div[1]/div')
for e in elements:
	fil.write(e.text)

for i in range(2,100):
	print(i)
	driver.get('https://oslusiadas.org/i/' + str(i) + '.html#_')
	driver.implicitly_wait(10)
	time.sleep(10)
	elements = driver.find_elements_by_xpath('/html/body/div[2]/div[2]/div[3]/div[1]/div')
	for e in elements:
		fil.write(e.text)
	time.sleep(5)


fil.close()

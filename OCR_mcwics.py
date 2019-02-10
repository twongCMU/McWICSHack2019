import pytesseract
import cv2
import numpy as np

''' Locate tesseract exe, if its not in PATH. Even if in PATH it doesn't seem to find it without this line...'''
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

''' load image '''
img = cv2.imread('C:/Users/Soraya/Downloads/temp_02.jpg')
img = cv2.bitwise_not(img)

''' pixel which are not black (0) put white (255) '''
img[np.where((img!=(0)))] = [255]

''' Make function. use Tesseract to extracts strings from images '''


def extract_string(image):
    text = pytesseract.image_to_string((image))
    return print(text)


extract_string(img)


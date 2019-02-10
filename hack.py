#!/usr/bin/python3
import memegenerator
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pytesseract
import cv2
import numpy as np

''' Locate tesseract exe, if its not in PATH. Even if in PATH it doesn't seem to find it without this line...'''
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = '/opt/local/bin/tesseract'

#before img = cv2.bitwise_not(img)
user_input = input("Filename path:")
img = cv2.imread(user_input)

f = open('pride.txt', 'r')

# remove the periods after an honorific so they don't get
# interpreted as a sentence delimiter and split apart
f_d = f.read()
f_d = re.sub('Mr\.', 'Mr', f_d)
f_d = re.sub('Miss.', 'Miss', f_d)
f_d = re.sub('Ms\.', 'Ms',f_d)
f_d = re.sub('Mrs\.', 'Mrs',f_d)
f_d = re.sub('St', 'St',f_d)

# split apart the text by sentence
f_d_array = re.split("[\.\!\?]", f_d)
new_arr = []

#make a new array with all the original lines but
#long amounts of whitespace are compressed into one spce
# "to be            or not to be" -> "to be or not to be"
for line in f_d_array:
    processed_line = re.sub(r'\s+',' ', line)
    new_arr.append(processed_line)
print("found " + str(len(new_arr)) + " sentences")
#new_arr.append("I are serious cat. Please you step into my office now")

''' load image '''
#img = cv2.imread('temp_02.jpg')
img = cv2.bitwise_not(img)
#cv2.imshow('test', img)
#cv2.waitKey(0)
''' pixel which are not black (0) put white (255) '''
#img[np.where((img!=(0)))] = [255]
#img[np.where((img>(40)))] = [255]

h = img.shape[0]
w = img.shape[1]
for y in range(h):
    for x in range(w):
        if img[y,x, 0] > 30 or img[y,x,1] > 30 or img[y,x,2] > 30:
            img[y,x] = (255,255,255)

#cv2.imshow('test', img)
#cv2.waitKey(0)
''' Make function. use Tesseract to extracts strings from images '''


def extract_string(image):
    text = pytesseract.image_to_string((image))
    return text


ext_str = extract_string(img)
ext_str = re.sub(r'\s+',' ', ext_str)
print(ext_str)
# now we have the string extracted from the image
# add it to the array of sentences from the book
new_arr.append(ext_str)

# use tfidf to make the pairwise similarities for all sentences
vectorizer=TfidfVectorizer()
response=vectorizer.fit_transform(new_arr)
# we only are interested in what is similar to the last sentence, the one from our
# extracted meme. [-1] is the array index to that line
cos_sim = linear_kernel(response[-1], response).flatten()
# we only want the 10 highest matches
related_docs_indices = cos_sim.argsort()[:-10:-1]

print(related_docs_indices)
count = 0
# generate the actual memes using the highest scoring sentences
# the original sentence will naturally be the first one so we could skip it but we don't here
for a in related_docs_indices:
    meme_text = new_arr[a]
    meme_text_top = meme_text
    meme_text_bottom = ""
    # ignore really long lines since they won't fit in the image well
    if len(meme_text) > 80:
        continue
    # if the line is kind of long, split it in half and put half on top and half on the bottom
    if len(meme_text) > 30:
        offset = meme_text.find(' ', int(len(meme_text)/2) - 5)
        if offset != -1:
            meme_text_top = meme_text[0: offset]
            meme_text_bottom = meme_text[offset + 1:]
    memegenerator.make_meme(meme_text_top, meme_text_bottom, "aliens.jpg", "out" + str(count) + ".jpg")
    count+=1
    # arbitrarily stop after generating 5 images so we don't make too many
    if count > 5:
        break

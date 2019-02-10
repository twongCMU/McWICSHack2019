#!/usr/bin/python3
import memegenerator
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

f = open('/home/tw/pride.txt', 'r')
f_d = f.read()
f_d = re.sub('Mr.', 'Mr', f_d)
f_d = re.sub('Miss.', 'Miss', f_d)
f_d = re.sub('Ms.', 'Ms',f_d)
f_d = re.sub('Mrs.', 'Mrs',f_d)
f_d = re.sub('St', 'St',f_d)
f_d_array = re.split("[\.\!\?]", f_d)
new_arr = []
for line in f_d_array:
    processed_line = re.sub(r'\s+',' ', line)
    new_arr.append(processed_line)
print("found " + str(len(new_arr)) + " sentences")
#new_arr.append("I are serious cat. Please you step into my office now")
new_arr.append("Earth rotates aroud the sun, moon rotates aroud the earth")


vectorizer=TfidfVectorizer()
response=vectorizer.fit_transform(new_arr)

cos_sim = linear_kernel(response[-1], response).flatten()
related_docs_indices = cos_sim.argsort()[:-10:-1]

print(related_docs_indices)
count = 0
for a in related_docs_indices:
    meme_text = new_arr[a]
    meme_text_top = meme_text
    meme_text_bottom = ""
    if len(meme_text) > 60:
        continue
    if len(meme_text) > 30:
        offset = meme_text.find(' ', int(len(meme_text)/2) - 5)
        if offset != -1:
            meme_text_top = meme_text[0: offset]
            meme_text_bottom = meme_text[offset + 1:]
    memegenerator.make_meme(meme_text_top, meme_text_bottom, "aliens.jpg", "out" + str(count) + ".jpg")
    count+=1
    if count > 5:
        break

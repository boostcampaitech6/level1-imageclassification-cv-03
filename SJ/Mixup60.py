import csv
import cv2
import glob
import numpy as np
import os
import random


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Create_data(index, female_normal, female_inc, female_mask5, male_normal, male_inc, male_mask5):
    cnt = 0
    W_data = []
    for age in range(1,3):
        cnt = 0
        random.shuffle(female_normal[age])
        random.shuffle(female_inc[age])
        random.shuffle(female_mask5[age])
        random.shuffle(male_normal[age])
        random.shuffle(male_inc[age])
        random.shuffle(male_mask5[age])
        for i in range(len(female_inc[age])):
            if cnt > 400:
                break
            for j in range(len(female_inc[age])):
                if cnt > 400:
                    break
                if i == j:
                    continue
                female_normal_add = cv2.add(female_normal[age][i], female_normal[age][j])
                female_inc_add = cv2.add(female_normal[age][i], female_inc[age][j])
                female_mask5_add = cv2.add(female_normal[age][i], female_mask5[age][j])
                cnt += 1

                if age == 0:
                    tmp = '20'
                elif age == 1:
                    tmp = '45'
                else:
                    tmp = '60'
                
                if index >= 10000:
                    new_path = f'images/0{index}_female_Asian_{tmp}'
                else:
                    new_path = f'images/00{index}_female_Asian_{tmp}'

                makedirs(new_path)

                cv2.imwrite(f'{new_path}/normal.jpg',female_normal_add)
                cv2.imwrite(f'{new_path}/incorrect_mask.jpg',female_inc_add)
                cv2.imwrite(f'{new_path}/mask5.jpg',female_mask5_add)

                if index >= 10000:
                    W_data.append([f'{index}','female','Asian',tmp,f'0{index}_female_Asian_{tmp}'])
                else:
                    W_data.append([f'{index}','female','Asian',tmp,f'00{index}_female_Asian_{tmp}'])
                
                print(index,'female')
                index += 1

        for i in range(len(male_inc[age])):
            if cnt > 1000:
                break
            for j in range(len(male_inc[age])):
                if cnt > 1000:
                    break
                if i == j:
                    continue
                male_normal_add = cv2.add(male_normal[age][i], male_normal[age][j])
                male_inc_add = cv2.add(male_normal[age][i], male_inc[age][j])
                male_mask5_add = cv2.add(male_normal[age][i], male_mask5[age][j])
                cnt += 1

                if age == 0:
                    tmp = '20'
                elif age == 1:
                    tmp = '45'
                else:
                    tmp = '60'
                
                if index >= 10000:
                    new_path = f'images/0{index}_male_Asian_{tmp}'
                else:
                    new_path = f'images/00{index}_male_Asian_{tmp}'

                makedirs(new_path)

                cv2.imwrite(f'{new_path}/normal.jpg',male_normal_add)
                cv2.imwrite(f'{new_path}/incorrect_mask.jpg',male_inc_add)
                cv2.imwrite(f'{new_path}/mask5.jpg',male_mask5_add)

                if index >= 10000:
                    W_data.append([f'{index}','male','Asian',tmp,f'0{index}_male_Asian_{tmp}'])
                else:
                    W_data.append([f'{index}','male','Asian',tmp,f'00{index}_male_Asian_{tmp}'])
                
                print(index,'male')
                index += 1

    return W_data



W_data = []
before_gender = 'male'
f = open('train.csv','r')
rdr = csv.reader(f)
i = 0
index = 10498
limit = 2350

female_normal = [[],[],[]]
female_inc = [[],[],[]]
female_mask5 = [[],[],[]]


male_normal = [[],[],[]]
male_inc = [[],[],[]]
male_mask5 = [[],[],[]]


for line in rdr:
    if i > limit:
        break
    if line[3] =='age':
        continue
    age = int(line[3])
    if age <= 20:
        path = line[4]
        cur_gender = line[1]

        image_normal = glob.glob(f'./images/{path}/normal.jpg')
        image_inc = glob.glob(f'./images/{path}/incorrect_mask.jpg')
        image_mask5 = glob.glob(f'./images/{path}/mask5.jpg')
        try:
            img_normal = cv2.imread(image_normal[0],cv2.IMREAD_COLOR)
            img_inc = cv2.imread(image_inc[0],cv2.IMREAD_COLOR)
            img_mask5 = cv2.imread(image_mask5[0],cv2.IMREAD_COLOR)
        except:
            continue

        if cur_gender == 'female':
            female_normal[0].append(img_normal)
            female_inc[0].append(img_inc)
            female_mask5[0].append(img_mask5)

        else:
            male_normal[0].append(img_normal)
            male_inc[0].append(img_inc)
            male_mask5[0].append(img_mask5)

    if 30 <= age <= 45:
        path = line[4]
        cur_gender = line[1]

        image_normal = glob.glob(f'./images/{path}/normal.jpg')
        image_inc = glob.glob(f'./images/{path}/incorrect_mask.jpg')
        image_mask5 = glob.glob(f'./images/{path}/mask5.jpg')
        try:
            img_normal = cv2.imread(image_normal[0],cv2.IMREAD_COLOR)
            img_inc = cv2.imread(image_inc[0],cv2.IMREAD_COLOR)
            img_mask5 = cv2.imread(image_mask5[0],cv2.IMREAD_COLOR)
        except:
            continue

        if cur_gender == 'female':
            female_normal[1].append(img_normal)
            female_inc[1].append(img_inc)
            female_mask5[1].append(img_mask5)

        else:
            male_normal[1].append(img_normal)
            male_inc[1].append(img_inc)
            male_mask5[1].append(img_mask5)

    if age >= 60:
        path = line[4]
        cur_gender = line[1]

        image_normal = glob.glob(f'./images/{path}/normal.jpg')
        image_inc = glob.glob(f'./images/{path}/incorrect_mask.jpg')
        image_mask5 = glob.glob(f'./images/{path}/mask5.jpg')
        try:
            img_normal = cv2.imread(image_normal[0],cv2.IMREAD_COLOR)
            img_inc = cv2.imread(image_inc[0],cv2.IMREAD_COLOR)
            img_mask5 = cv2.imread(image_mask5[0],cv2.IMREAD_COLOR)
        except:
            continue

        if cur_gender == 'female':
            female_normal[2].append(img_normal)
            female_inc[2].append(img_inc)
            female_mask5[2].append(img_mask5)

        else:
            male_normal[2].append(img_normal)
            male_inc[2].append(img_inc)
            male_mask5[2].append(img_mask5)

    
    i += 1
        

f.close

W_data = Create_data(index, female_normal, female_inc, female_mask5, male_normal, male_inc, male_mask5)

g = open('train.csv','a')
wr = csv.writer(g)

for data in W_data:
    wr.writerow(data)
g.close

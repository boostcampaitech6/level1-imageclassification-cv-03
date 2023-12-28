import csv
import cv2
import glob
import numpy as np
import os


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Create_data(index):
    cnt = 0
    W_data = []
    for i in range(1,len(female_left_inc)):
        if cnt > 400:
            break
        for j in range(len(female_left_inc)-i):
            female_normal = cv2.hconcat([female_left_normal[j], female_right_normal[j+i]])
            female_inc = cv2.hconcat([female_left_inc[j], female_right_inc[j+i]])
            female_mask1 = cv2.hconcat([female_left_mask1[j], female_right_mask1[j+i]])
            female_mask2 = cv2.hconcat([female_left_mask2[j], female_right_mask2[j+i]])
            female_mask3 = cv2.hconcat([female_left_mask3[j], female_right_mask3[j+i]])
            female_mask4 = cv2.hconcat([female_left_mask4[j], female_right_mask4[j+i]])
            female_mask5 = cv2.hconcat([female_left_mask5[j], female_right_mask5[j+i]])
            cnt += 1

            new_path = f'images/00{index}_female_Asian_60'

            makedirs(new_path)

            cv2.imwrite(f'{new_path}/normal.jpg',female_normal)
            cv2.imwrite(f'{new_path}/incorrect_mask.jpg',female_inc)
            cv2.imwrite(f'{new_path}/mask1.jpg',female_mask1)
            cv2.imwrite(f'{new_path}/mask2.jpg',female_mask2)
            cv2.imwrite(f'{new_path}/mask3.jpg',female_mask3)
            cv2.imwrite(f'{new_path}/mask4.jpg',female_mask4)
            cv2.imwrite(f'{new_path}/mask5.jpg',female_mask5)

            W_data.append([f'{index}','female','Asian','60',f'00{index}_female_Asian_60'])
            print(index,'female')
            index += 1

    for i in range(1,len(male_left_inc)):
        if cnt > 1000:
            break
        for j in range(len(male_left_inc)-i):
            male_normal = cv2.hconcat([male_left_normal[j], male_right_normal[j+i]])
            male_inc = cv2.hconcat([male_left_inc[j], male_right_inc[j+i]])
            male_mask1 = cv2.hconcat([male_left_mask1[j], male_right_mask1[j+i]])
            male_mask2 = cv2.hconcat([male_left_mask2[j], male_right_mask2[j+i]])
            male_mask3 = cv2.hconcat([male_left_mask3[j], male_right_mask3[j+i]])
            male_mask4 = cv2.hconcat([male_left_mask4[j], male_right_mask4[j+i]])
            male_mask5 = cv2.hconcat([male_left_mask5[j], male_right_mask5[j+i]])
            cnt += 1

            new_path = f'images/00{index}_male_Asian_60'

            makedirs(new_path)

            cv2.imwrite(f'{new_path}/normal.jpg',male_normal)
            cv2.imwrite(f'{new_path}/incorrect_mask.jpg',male_inc)
            cv2.imwrite(f'{new_path}/mask1.jpg',male_mask1)
            cv2.imwrite(f'{new_path}/mask2.jpg',male_mask2)
            cv2.imwrite(f'{new_path}/mask3.jpg',male_mask3)
            cv2.imwrite(f'{new_path}/mask4.jpg',male_mask4)
            cv2.imwrite(f'{new_path}/mask5.jpg',male_mask5)
            W_data.append([f'{index}','male','Asian','60',f'00{index}_male_Asian_60'])
            index += 1
            print(index,'male')

    return W_data



W_data = []
before_gender = 'male'
f = open('train.csv','r')
rdr = csv.reader(f)
i = 0
index = 6960

female_left_normal = []
female_left_inc = []
female_left_mask1 = []
female_left_mask2 = []
female_left_mask3 = []
female_left_mask4 = []
female_left_mask5 = []

female_right_normal = []
female_right_inc = []
female_right_mask1 = []
female_right_mask2 = []
female_right_mask3 = []
female_right_mask4 = []
female_right_mask5 = []

male_left_normal = []
male_left_inc = []
male_left_mask1 = []
male_left_mask2 = []
male_left_mask3 = []
male_left_mask4 = []
male_left_mask5 = []

male_right_normal = []
male_right_inc = []
male_right_mask1 = []
male_right_mask2 = []
male_right_mask3 = []
male_right_mask4 = []
male_right_mask5 = []


for line in rdr:
    if line[3] =='age':
        continue
    age = int(line[3])
    if age >= 60:
        path = line[4]
        cur_gender = line[1]

        image_normal = glob.glob(f'./images/{path}/normal.jpg')
        image_inc = glob.glob(f'./images/{path}/incorrect_mask.jpg')
        image_mask1 = glob.glob(f'./images/{path}/mask1.jpg')
        image_mask2 = glob.glob(f'./images/{path}/mask2.jpg')
        image_mask3 = glob.glob(f'./images/{path}/mask3.jpg')
        image_mask4 = glob.glob(f'./images/{path}/mask4.jpg')
        image_mask5 = glob.glob(f'./images/{path}/mask5.jpg')
        
        img_normal = cv2.imread(image_normal[0],cv2.IMREAD_COLOR)
        img_inc = cv2.imread(image_inc[0],cv2.IMREAD_COLOR)
        img_mask1 = cv2.imread(image_mask1[0],cv2.IMREAD_COLOR)
        img_mask2 = cv2.imread(image_mask2[0],cv2.IMREAD_COLOR)
        img_mask3 = cv2.imread(image_mask3[0],cv2.IMREAD_COLOR)
        img_mask4 = cv2.imread(image_mask4[0],cv2.IMREAD_COLOR)
        img_mask5 = cv2.imread(image_mask5[0],cv2.IMREAD_COLOR)

        if cur_gender == 'female':
            female_left_normal.append(img_normal[:, :192])
            female_left_inc.append(img_inc[:, :192])
            female_left_mask1.append(img_mask1[:, :192])
            female_left_mask2.append(img_mask2[:, :192])
            female_left_mask3.append(img_mask3[:, :192])
            female_left_mask4.append(img_mask4[:, :192])
            female_left_mask5.append(img_mask5[:, :192])

            female_right_normal.append(img_normal[:, 192:])
            female_right_inc.append(img_inc[:, 192:])
            female_right_mask1.append(img_mask1[:, 192:])
            female_right_mask2.append(img_mask2[:, 192:])
            female_right_mask3.append(img_mask3[:, 192:])
            female_right_mask4.append(img_mask4[:, 192:])
            female_right_mask5.append(img_mask5[:, 192:])

        else:
            male_left_normal.append(img_normal[:, :192])
            male_left_inc.append(img_inc[:, :192])
            male_left_mask1.append(img_mask1[:, :192])
            male_left_mask2.append(img_mask2[:, :192])
            male_left_mask3.append(img_mask3[:, :192])
            male_left_mask4.append(img_mask4[:, :192])
            male_left_mask5.append(img_mask5[:, :192])

            male_right_normal.append(img_normal[:, 192:])
            male_right_inc.append(img_inc[:, 192:])
            male_right_mask1.append(img_mask1[:, 192:])
            male_right_mask2.append(img_mask2[:, 192:])
            male_right_mask3.append(img_mask3[:, 192:])
            male_right_mask4.append(img_mask4[:, 192:])
            male_right_mask5.append(img_mask5[:, 192:])

        i += 1
        

f.close

W_data = Create_data(index)

g = open('train.csv','a')
wr = csv.writer(g)

for data in W_data:
    wr.writerow(data)
g.close

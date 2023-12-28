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
    for i in range(len(male_left_mask5)):
        for j in range(len(male_right_mask5)):
            male_mask5 = cv2.vconcat([male_left_mask5[i], male_right_mask5[j]])

            new_path = f'images/00{index}_male_Asian_50'

            makedirs(new_path)

            cv2.imwrite(f'{new_path}/mask5.jpg',male_mask5)
            W_data.append([f'{index}','male','Asian','50',f'00{index}_male_Asian_50'])
            index += 1
            print(index,'male')

    return W_data



W_data = []
f = open('train.csv','r')
rdr = csv.reader(f)
i = 0
index = 9257

male_left_mask5 = []

male_right_mask5 = []



Mask_list = ['005499_male_Asian_55', '005497_male_Asian_56', '005547_male_Asian_56', '005558_male_Asian_52', '005523_male_Asian_52', '005545_male_Asian_53', '005559_male_Asian_52', '005553_male_Asian_48', '005555_male_Asian_48', '005556_male_Asian_50', '005537_male_Asian_50', '005488_male_Asian_50', '005491_male_Asian_46', '005557_male_Asian_46', '005551_male_Asian_43']
Face_list = ['003575_male_Asian_43', '003632_male_Asian_43', '001401_male_Asian_48', '003550_male_Asian_48', '001412_male_Asian_50','001420_male_Asian_49', '003748_male_Asian_50', '001704_male_Asian_51', '001703_male_Asian_49', '000650_male_Asian_54', '000655_male_Asian_53', '000808_male_Asian_54', '003534_male_Asian_54', '003751_male_Asian_52', '001669_male_Asian_52', '003812_male_Asian_52']
for m in Mask_list:

    image_mask5 = glob.glob(f'./images/{m}/mask5.jpg')

    img_mask5 = cv2.imread(image_mask5[0],cv2.IMREAD_COLOR)


    male_right_mask5.append(img_mask5[256:, :])


for m in Face_list:

    image_mask5 = glob.glob(f'./images/{m}/mask5.jpg')

    img_mask5 = cv2.imread(image_mask5[0],cv2.IMREAD_COLOR)



    male_left_mask5.append(img_mask5[:256, :])

        

f.close

W_data = Create_data(index)

g = open('train.csv','a')
wr = csv.writer(g)

for data in W_data:
    wr.writerow(data)
g.close

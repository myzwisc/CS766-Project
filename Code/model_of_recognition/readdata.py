import numpy as np
import cv2
import os
import glob
import pdb
import xml.etree.ElementTree as ET


wsi_mask_paths = glob.glob(r"./images/test/*.JPG")
wsi_mask_paths.sort()


data = []
label = []

labeldict = {'zyw':0,'myz':1}

for imgpath in wsi_mask_paths:
    xmlpath = imgpath[:-3]+'xml'
    print imgpath
    img = cv2.imread(imgpath)
    print xmlpath
    xml_list = []
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text)
                 )
        xml_list.append(value)

    img = cv2.transpose(img)
    img = cv2.flip(img, 0)

    for value in xml_list:
        labelclass = value[3]
        xmin = value[4]
        ymin = value[5]
        xmax = value[6]
        ymax = value[7]
        width = xmax-xmin
        height = ymax-ymin

        rate = width * 1.0 / height

        if rate > 1.5:
            ymin = max(int(ymax - width/1.3),0)
        elif rate < 1.0:
            xmin = max(int(xmax - height * 1.3),0)


        subimg = img[ymin:ymax,xmin:xmax,:]

        subimg = cv2.transpose(subimg)
        subimg = cv2.resize(subimg,(47, 57), interpolation=cv2.INTER_CUBIC)

        subimg = cv2.cvtColor( subimg, cv2.COLOR_RGB2GRAY )
        data.append(subimg)
        label.append(labeldict[labelclass])
data = np.asarray(data)
label = np.asarray(label)

np.save('testdata.npy',data)
np.save('testlabel.npy',label)

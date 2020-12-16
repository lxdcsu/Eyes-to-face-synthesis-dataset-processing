import cv2
import numpy
import os
import glob as gb
from PIL import Image


img_path = []
for root, dirs, files in os.walk("img", topdown=False):
    for name in files:
        if name.endswith('.jpg'):
            print(os.path.join(root, name))
            img_path.append(os.path.join(root, name))
#    for name in dirs:
#        print(os.path.join(root, name))
#Import opencV's own classification model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('model/haarcascade_frontalface_default.xml')#the path to import the classifier
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
eye_cascade.load('model/haarcascade_eye.xml')

#Face detection and save
save_path1 = 'new faces/'#Save path of face picture
i=0
#img_path = gb.glob("new faces input/*.jpg")#Traversing the image path

for path in img_path:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        roiImg = img[y:y+h,x:x+w]#intercept face detection images
        outImg = cv2.resize(roiImg,(256,256))#normalized to 256*256
        cv2.imwrite(save_path1+str(i)+'.jpg',outImg)
        i+=1
    print("Save the path",path)
print("Face detection completed！")
#the eye detects and saves

save_path2 = 'new eyes/'
save_path3 = 'new results/'
isExists=os.path.exists(save_path3)
if not isExists:
    os.makedirs(save_path3)
    
j=0
img_path2 = gb.glob("new faces/*.jpg")

def min(a,b):#define a function that returns a minimum value
    if a<=b:
        return a
    else:
        return b

def max(a,b):#define a function that returns a maximum value
    if a>=b:
        return a
    else:
        return b

for path2 in img_path2:
    img2 = cv2.imread(path2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    roi_gray = gray2[0:128, 0:256]#set the area of interest to detect the eyes
    roi_color = img2[0:128, 0:256]
    eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5)
    data=[]
    for (ex,ey,ew,eh) in eyes:
        data.append(ex)
        data.append(ey)
        data.append(ew)
        data.append(eh)#read the coordinates of the eyes
   
            
        #roiImg2 = roi_color[ey:ey+eh,ex:ex+ew]
        #outImg2 = cv2.resize(roiImg2,(55,55))
        #cv2.imwrite(save_path2+str(j)+'.jpg',roiImg2)
        #j+=1
    if len(data) == 8: #two eyes need to be detected 
        if data[0]<data[4]:#read first with the left eye
            NewImage = Image.new('RGB', (256, 256),(0,0,0))#create a 256 full black graph
            NewImage.save('256.jpg')
            NewImage=cv2.imread("256.jpg")
            
            NewImage1 = Image.new('RGB', (512, 256),(0,0,0))#create a 256 full black graph
            NewImage1.save('512.jpg')
            NewImage1=cv2.imread("512.jpg")
            
            roiImg2 = roi_color[min(data[1],data[5]):max(data[1]+data[3],data[5]+data[7]),data[0]:data[4]+data[6]]#intercept the eye area
            NewImage[min(data[1],data[5]):max(data[1]+data[3],data[5]+data[7]),data[0]:data[4]+data[6]]=roiImg2#copy the eye to the same location at 256
#            a1=128-data[3]/2
#            a2=128+data[3]/2
#            a3=128-(data[4]+data[6]-data[0])/2
#            a4=128+(data[4]+data[6]-data[0])/2
            #NewImage[128-data[3]/2:128+data[3]/2,128-(data[4]+data[6]-data[0])/2:128+(data[4]+data[6]-data[0])/2]=roiImg2#将眼睛图复制到256黑图的正中
            

            NewImage1[0:256,0:256]=NewImage#put the eye area to the left
            NewImage1[0:256,256:512]=img2#face on the right
            
            cv2.imwrite(save_path2+str(j)+'.jpg',NewImage)# save eye area
            cv2.imwrite(save_path3+str(j)+'.jpg',NewImage1)#save the result
            
            j+=1
            
        else:#read the right eye coordinates first
            NewImage = Image.new('RGB', (256, 256),(0,0,0))
            NewImage.save('256.jpg')
            NewImage=cv2.imread("256.jpg")
            
            NewImage1 = Image.new('RGB', (512, 256),(0,0,0))
            NewImage1.save('512.jpg')
            NewImage1=cv2.imread("512.jpg")
            
            roiImg2 = roi_color[min(data[1],data[5]):max(data[1]+data[3],data[5]+data[7]),data[4]:data[0]+data[2]]
            NewImage[min(data[1],data[5]):max(data[1]+data[3],data[5]+data[7]),data[4]:data[0]+data[2]]=roiImg2
            #NewImage[128-data[7]/2:128+data[7]/2,128-(data[0]+data[2]-data[4])/2:128+(data[0]+data[2]-data[4])/2]=roiImg2#j

            NewImage1[0:256,0:256]=NewImage
            NewImage1[0:256,256:512]=img2

            cv2.imwrite(save_path2+str(j)+'.jpg',NewImage)
            cv2.imwrite(save_path3+str(j)+'.jpg',NewImage1)
            j+=1       
    else:
        j+=1
#toImage = Image.new('RGBA', (110, 55))#创建新的图片
#fromImage=open("0.jpg")
#pt=(0,0)
#toImage.paste(fromImage,pt)
#toImage.save('00.jpg')  
    
#cv2.imshow('img',img)
#cv2.imwrite("face_detected_1.jpg", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
import numpy as np 
import cv2
import os

#knn algorithm code
def distance(v1,v2):
    #eucledian distance
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist=[]

    for i in range(train.shape[0]):
        #getting the vector and lable
        ix=train[i,:-1]
        iy=train[i,-1]
        #computing the distance from the  test pt/
        d=distance(test,ix)
        dist.append([d,iy])
    #sort based on distance and get  top k
    dk=sorted(dist, key=lambda x:x[0])[:k]
    #retrieve only the lables
    lables =np.array(dk)[:,-1]

    #Get freqkuency of each lable
    output=np.unique(lables, return_counts=True)
    #find max frequency and corresponding label 
    index=np.argmax(output[1])
    return output[0][index]
########################

cap =cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
dataset_path="./face_dataset/"

face_data=[]
lables=[]
class_id =0

names ={}


#dataset preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        data_item =np.load(dataset_path +fx)
        face_data.append(data_item)

        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        lables.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_lables=np.concatenate(lables,axis=0).reshape((-1,1))
print(face_lables .shape) 
print(face_dataset.shape)

trainset=np.concatenate((face_dataset,face_lables), axis=1)
print(trainset.shape)

font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    #convert to grayscale
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect multifaces in the image
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for face in faces:
        x ,y ,w ,h=face
        
        #get the face ROI
        offset=5
        face_section=frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section=cv2.resize(face_section, (100,100))

        out=knn(trainset, face_section.flatten())

        #draw rectgangle in the original image 
        cv2.putText(frame,names[int(out)],cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    
    cv2.imshow("Faces",frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


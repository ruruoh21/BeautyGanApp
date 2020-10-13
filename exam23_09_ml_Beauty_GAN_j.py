#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# In[2]:


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')   #생김새(모양) 예측하는거 #랜드마크 5개를 찾아준다.


# In[3]:


img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.show()


# In[4]:


#얼굴인식(사각형 표시)
img_result = img.copy()
dets = detector(img)   #detector에 주어진 이미지에서 얼굴을 찾아줌   
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16, 10))   #1 : 한줄로 이미지를 표시해라
    for det in dets:    #dets에는 이미지에 존재하는 얼굴 개수만큼 값(fig, ax)이 저장되어있음
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')  #사각형 그리는 함수
        ax.add_patch(rect)    #ax=이미지크기  #figure:도화지, 배경
    ax.imshow(img_result)
    plt.show()  #인지한 얼굴을 빨간 사각형으로 표시


# In[5]:


#눈코 위치 찾기
fig, ax = plt.subplots(1, figsize=(16, 10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)


# In[6]:


#얼굴만 따로 뽑는다
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[7]:


#코드 합친거

#얼굴을 정렬해주는 함수를 정의(함수선언)
def align_faces(img):    
    dets = detector(img, 1)   #dets:얼굴 영역정보들이 들어있음.(폭, 높이, 위치 등)
    objs = dlib.full_object_detections()   #객체를 찾아 
    for detection in dets:
        s = sp(img, detection)   #점에 대한 정보
        objs.append(s) 
        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)  #faces: 얼굴 이미지들이 들어있음.
        return faces

#함수호출
test_img = dlib.load_rgb_image('./imgs/12.jpg')  #이미지를 불러옴
test_faces = align_faces(test_img)   #얼굴 찾아주는 애(face_detertor)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20, 16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)


# In[8]:


#화장시키기


# In[11]:


sess =  tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[12]:


#스케일링
def preprocess(img):
    return (img / 255. - 0.5) * 2


#다시 원본 이미지로 출력
def deprocess(img):
    return (img + 1) / 2


# In[47]:


# source 이미지
img1 = dlib.load_rgb_image('./imgs/12.jpg')  #no_makeup/xfsy_0405
img1_faces = align_faces(img1)

# reference 이미지
img2 = dlib.load_rgb_image('./imgs/makeup/2020.jpg')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# img1 = dlib.load_rgb_image('./imgs/no_makeup/vSYYZ306.png')
# img1_faces = alingn_faces(img1)
# 
# 
# img2 = dlib.load_rgb_image('./imgs/makeup/002.jpg')
# img2_faces = alingn_faces(img2)
# 
# fig, axes = plt.subplots(1,2,figsize=(16,10))
# axes[0].inshow(img1_faces[0])
# axes[1].inshow(img2_faces[0])
# plt.show()

# In[48]:


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})  #화장한 이미지를 만들어서 output으로 줌
output_img = deprocess(output[0])   #postprocess로 스케일링

#출력
fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()


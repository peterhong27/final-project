import cv2
import numpy as np

img_ori = cv2.imread('1.jpg')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
img_thresh = cv2.adaptiveThreshold(
    img_blurred, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
)
#ret, img_binary = cv.threshold(img_gray, 127, 255, 0) #검은창에 도형두개잇는거
contours, hierachy = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

min_x=[]
max = 0
mix = 0
may = 0
miy = 0
min_y=[]
max_w=[]
max_h=[]
a = []
cnt =0
xy=[]
#xy = [[0 for rows in range(2)] for cols in range(len(contours))]   #len(contours) x 2 행렬
for t in range(len(contours)):
    for u in range(len(contours[t])):
        #xy[cnt][0] = contours[t][u][0][0]
        #xy[cnt][1] = contours[t][u][0][1]
        xy.append([contours[t][u][0][0], contours[t][u][0][1]])
        cnt += 1


np.savetxt('contour.txt', xy, delimiter=" ", fmt='%x')
#print(len(xy))
#print(len(contours))
#print(xy[0][0])
#print(xy[0][1])
#print(xy[3][0])

for k in range(len(contours)):
    max = contours[k][0][0][0]
    mix = contours[k][0][0][0]
    may = contours[k][0][0][1]
    miy = contours[k][0][0][1]
    for i in range(len(contours[k])) :

        if max < contours[k][i][0][0] :
            max = contours[k][i][0][0]
        if mix > contours[k][i][0][0] :
            mix = contours[k][i][0][0]
        if may < contours[k][i][0][1] :
            may = contours[k][i][0][1]
        if miy > contours[k][i][0][1] :
            miy = contours[k][i][0][1]
    min_x.append(mix)
    max_w.append(max-mix)
    min_y.append(miy)
    max_h.append(may-miy)
    a.append(len(contours[k]))

#a.append(len(contours[k]))
b = [len(contours)]
print(len(xy))
np.savetxt('num of contour.txt', b, delimiter=" ", fmt='%x')
np.savetxt('array num of each contours.txt', a , delimiter=" ", fmt='%x')

contours_dicts =[]
for k in range(len(min_x)) :
    #print('min x', k, min_x[k])
    #print('min y', k, min_y[k])
    #print('max w', k, max_w[k])
    #print('max h', k, max_h[k])
    contours_dicts.append({
        'contour': contours[k],
        'x': min_x[k],
        'y': min_y[k],
        'w': max_w[k],
        'h': max_h[k],
        'cx': min_x[k] + (max_w[k] / 2),  # 중점 x
        'cy': min_y[k] + (max_h[k] / 2)  # 중점 y
    })
    #cv2.rectangle(temp_result, (min_x[k], min_y[k]), (min_x[k] + max_w[k], min_y[k] + max_h[k]),
                  #(255, 255, 255), 1)
    cv2.rectangle(img_ori, (min_x[k], min_y[k]),(min_x[k]+max_w[k],min_y[k]+max_h[k]),(255,255,255),2)
#print(contours_dicts)
np.savetxt('contours_dict.txt',contours_dicts,delimiter=" ", fmt='%s')
cv2.imshow("result",img_ori)
cv2.waitKey(0)
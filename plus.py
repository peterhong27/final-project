import cv2
import numpy as np
import matplotlib.pyplot as plt#이미지 결과 표시 import
import os
import time
#background를 어둡게
y = time.time()
plt.style.use('dark_background')
#video capture
cap = cv2.VideoCapture("KakaoTalk_Video_20191001_1644_44_980.mp4")
#image폴더 생성
if not os.path.exists('images'):
    os.makedirs('images')
#for frame identity
index = 0
t1 = 0
t2 = 0
t3 = 0
t4 = 0
a1 = 0
a2 = 0
a3 = 0
a4 = 0
f = open("total.txt", 'w')
e = open("c1.txt", 'w')
e1 = open("c2.txt", 'w')
e2 = open("c3.txt", 'w')
e3 = open("c4.txt", 'w')
while(True):
    c10=time.time()
    # Extract images
    ret, frame = cap.read()
    # end of frames
    if not ret:
        break
    # Saves images
    frame1 = frame[100:400, 180:540]  # 150:360
    frame2 = cv2.resize(frame1, dsize=(0, 0), fx=2, fy=404/300, interpolation=cv2.INTER_LINEAR)
    name = './images/frame' + str(index) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame2)
    #이미지 불러오기
    img_ori = cv2.imread('./images/frame' + str(index) + '.jpg')
    #image shape에 넓이 높이 channel 저장
    height, width, channel = img_ori.shape

    # gray scale로 바꾸기, hsv로 변경후 v채널만 사용 가능
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    #노이즈를 줄이기 위해 gaussianblur 사용
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # image에 threshold를 지정해주어 이미지를 구별하기 쉽게 만들어준다.

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,#grayscale 이미지
        maxValue=255.0,#임계값을 흰색으로
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,#threshold value를 X, Y를 중심으로 block Size * block Size 안에 있는 픽셀 값의 평균에서 C를 뺸 값을 문턱값으로 함
        thresholdType=cv2.THRESH_BINARY_INV,#threshold type을 반전된 결과가 나오도록 지정
        blockSize=19,#threshold를 적용할 영역 사이즈
        C=9#평균값
    )
    #contour(윤곽선)찾기
    c7= time.time()
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    t7=time.time()-c7
    #img_thresh->이미지 소스
    #mode->contour 간 계층구조 상관관계를 고려하지 않고 contour추출
    #method->contour의 수평, 수직, 대각선 방향의 점은 모두 버리고 끝점만 남겨둠->예를 들어 직사각형의 경우 4개의 모서리점 제외하고 다 버림
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    #np.zeros->0으로 초기화된 shape차원의 ndarray배열 객체 반환
    #shape는 행렬의 차원 ndarray는 배열객체를 반환
    #np.unit8->양수만 표현 가능, 2^8갯수만큼 표현 가능, 0~255
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    #drawcontours를 사용하여 그림
    #contours->contour를 그림
    #contourIdx=-1->전체 contour를 그림
    #color->흰색으로 그림
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []
    #리스트 생성
    #contour정보 저장 위해 for문 사용
    t= time.time()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)#contour의 사각형 범위 찾기 하드웨어로 구현
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
       #rectangle함수를 사용하여 이미지에 위에서 찾은 사각형을 그린다
        # insert to dict
        #contour_dict에 contour의 정보 저장
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)#contour를 감싼 사각형의 중심 좌표
        })
    c1 = time.time() - t
    a1 = a1 + c1
    if c1 != 0:
        t1 += 1
    #사각형의 정보에서 번호판이 될 가능성 있는 정보들을 걸러냄
    MIN_AREA = 80#최소 넓이
    MIN_WIDTH, MIN_HEIGHT = 2, 8#최소 높이와 길이
    MIN_RATIO, MAX_RATIO = 0.25, 1.0#가로 대비 세로 비율의 최소 최대값
    #가능한 사각형을 possible contours에 저장
    possible_contours = []

    cnt = 0
    t = time.time()
    for d in contours_dict:
        #contours_dict에 있는 사각형의 넓이와 가로 대비 세로의 비율 계산
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            #조건 비교
            d['idx'] = cnt#각 윤곽선의 index값을 매겨놓고 저장(나중에 조건에 맞는 윤곽선들의 index만 따로 분리)
            cnt += 1
            possible_contours.append(d)#possible contour에 저장

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    c2 = time.time() - t
    a2 = a2 + c2
    if c2 != 0:
        t2 += 1
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)#possible contour의 이미지를 다시 사각형으로 그리기
    #possible contour에 있는 배열의 모양을 보고 가능성 높은 이미지를 다시 한번 추려내기
    MAX_DIAG_MULTIPLYER = 5#contour 중심 사이의 길이 최대값이 contour의 대각선 길이의 5배
    MAX_ANGLE_DIFF = 12.0#contour의 중심끼리 선분을 이었을때 각도 최대값
    MAX_AREA_DIFF = 0.5#contour의 면적 차이 최대값
    MAX_WIDTH_DIFF = 0.8#contour의 너비 차이 최대값
    MAX_HEIGHT_DIFF = 0.2#contour의 높이 차이 최대값
    MIN_N_MATCHED = 5#contour의 후보군그룹이 3개 미만이면 제외
    MAX_N_MATCHED = 7# contour의 후보군그룹이 7개 이상이면 제외


    def find_chars(contour_list):#find_chars 함수 지정(나중에 recursive(재귀적)방식으로 번호판 후보군 지정을 위해)
        matched_result_idx = []#최종적으로 남는 후보군의 index값 저장

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:#contour d1과 d2를 비교하여 두개의 index가 같으면 비교 할 필요 없이 같다고 가정하고 continue로 넘어감
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])#센터의 중심을 이어주는 선을 그린뒤 그 선의 삼각형 밎변과 높이

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)#d1의 대각선 길이

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))#사각형 사이의 거리를 구하기 위한 함수
                #두 contour의 각도차이 구하기
                if dx == 0:
                    angle_diff = 90#dx가 0이면 각도를 90도라 지정
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))#각도를 구하기 위해 아크탄젠트 사용하고 라디안 값을 degree로 나타내기 위해 np.degrees사용
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])#면적의 비율
                width_diff = abs(d1['w'] - d2['w']) / d1['w']#너비의 비율
                height_diff = abs(d1['h'] - d2['h']) / d1['h']#높이의 비율
                #파라미터 기준에 맞는 index d2만 matched_contours_idx에 추가
                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF \
                        and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])
            #d1도 index 추가
            matched_contours_idx.append(d1['idx'])
            #번호판의 후보군의 갯수가 MIN_N_MATCHED보다 작으면 번호판이 아니라고 간주
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue
            #최종 후보군에 넣기
            matched_result_idx.append(matched_contours_idx)
            #남은 후보들끼리 다시 비교함
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            #다시 뽑힌 index는 possible contour에 넣고 재귀함수로 돌림
            recursive_contour_list = find_chars(unmatched_contour)
            #다시 살아남은 index를 최종 결과에 다시 넣어줌
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx
    t = time.time()
    #선별된 index를 result_idx에 저장
    result_idx = find_chars(possible_contours)
    c3 = time.time()-t
    a3 = a3 + c3
    if c3 != 0:
        t3 += 1
    matched_result = []

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    #마지막 결과를 사각형으로 다시 그리기
    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255),
                          thickness=2)
    #번호판의 후보군중 기울어진 후보군을 다시 직선으로 돌려놓기
    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []
    t = time.time()
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])#x방향 순차적으로 정렬

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2#번호판의 중심좌표 구하기

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        #번호판의 너비 구하기
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        #번호판의 높이 구하기

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )#번호판의 기울어진 각도를 알기위해 삼각형의 밑변과 높이 구하기

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        #아크사인을 사용하여 각도를 구함
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        #물체를 평면상의 한 점을 중심으로 세타만큼 회전하는 변환 행렬을 구함
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        #변환 행렬로 이미지의 위치를 변경
        #회전된 이미지에서 번호판 부분만 잘라내기
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        #번호판의 좌표와 높이 너비 정보 저장
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
    c4 = time.time()-t
    a4 = a4 + c4
    if c4 != 0:
        t4 += 1

    longest_idx = -1
    #번호판이 2개 이상이거나 없을경우 예외처리
    try:
        info = plate_infos[longest_idx]

        img_out = img_ori.copy()
    #번호판 전체를 빨간색 사각형으로 그리기

        cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
                  color=(255, 0, 0), thickness=2)

    #번호판의 좌표와 높이 너비 정보 출력
        print(info['w'], info['h'])
        print(info['x'], info['y'])
    #이미지 저장
        cv2.imwrite('./images/frame result' + str(index) + '.jpg', img_out)
    except IndexError:
        print('번호판이 없습니다.')
    except NameError:
        print('NameError')
    # next frame
    index += 1
    a10=time.time()-c10
    f.write(str(a10))
    f.write("\n")
    e.write(str(c1))
    e.write("\n")
    e1.write(str(c2))
    e1.write("\n")
    e2.write(str(c3))
    e2.write("\n")
    e3.write(str(c4))
    e3.write("\n")
y1 = time.time() - y
f.close()
e.close()
e1.close()
e2.close()
e3.close()
print("1", t1)
print("2", t2)
print("3", t3)
print("4", t4)
print("a1", a1)
print("a2", a2)
print("a3", a3)
print("a4", a4)
print("a1/t", a1/t1)
print("a2/t", a2/t2)
print("a3/t", a3/t3)
print("a4/t", a4/t4)
print("y", y1)
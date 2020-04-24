import cv2
import numpy as np
import paddlehub as hub

# 人脸关键点检测器
module = hub.Module(name="face_landmark_localization")

# 猫脸检测器
cat_path = "data/model/haarcascade_frontalcatface_extended.xml"
facecascade = cv2.CascadeClassifier(cat_path)
ret = facecascade.load(cat_path)
old_cat_face_loc = np.array([-1, -1])


# 人脸关键点检测 + 猫脸检测 + 泊松融合
def human_mouth_paste_to_cat(human_frame, cat_frame):
    global old_cat_face_loc
    result = module.keypoint_detection(images=[human_frame])
    landmarks = result[0]['data'][0]
    landmarks = np.array(landmarks, dtype=np.int)

    mouth_landmark = landmarks[48:, :]
    # 扩个边
    border = 8
    mouth = human_frame[int(np.min(mouth_landmark[:, 1])) - border: int(np.max(mouth_landmark[:, 1]) + border),
            int(np.min(mouth_landmark[:, 0])) - border: int(np.max(mouth_landmark[:, 0])) + border, :]
    mouth_landmark[:, 0] -= (np.min(mouth_landmark[:, 0]) - border)
    mouth_landmark[:, 1] -= (np.min(mouth_landmark[:, 1]) - border)

    # 制作用于泊松融合的mask
    mask = np.zeros((mouth.shape[0], mouth.shape[1], 3)).astype(np.float32)
    for i in range(mouth_landmark.shape[0]):  # 先画线
        cv2.line(mask, (mouth_landmark[i, 0], mouth_landmark[i, 1]), (
        mouth_landmark[(i + 1) % mouth_landmark.shape[0], 0], mouth_landmark[(i + 1) % mouth_landmark.shape[0], 1]),
                 (255, 255, 255), 10)
    mask_tmp = mask.copy()
    for i in range(6, mask.shape[0] - 6):  # 将线内部的范围都算作mask=255
        for j in range(6, mask.shape[1] - 6):
            if (np.max(mask_tmp[:i, :j, :]) == 0 or np.max(mask_tmp[i:, :j, :]) == 0 or np.max(
                    mask_tmp[:i, j:, :]) == 0 or np.max(mask_tmp[i:, j:, :]) == 0):
                mask[i, j, :] = 0
            else:
                mask[i, j, :] = 255

    # 猫脸检测
    width, height, channels = cat_frame.shape
    cat_gray = cv2.cvtColor(cat_frame, cv2.COLOR_BGR2GRAY)
    cat_face_loc = facecascade.detectMultiScale(cat_gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    cat_face_loc = np.array(cat_face_loc)
    if len(cat_face_loc) == 0:
        return cat_frame
    ind = np.argmin(cat_face_loc[:, 1])
    cat_face_loc = cat_face_loc[ind]
    if old_cat_face_loc[0] != -1:  # 因为猫脸检测抖动太厉害，所以此处用历史坐标缓冲一下
        cat_face_loc = 0.9 * old_cat_face_loc + 0.1 * cat_face_loc
    old_cat_face_loc = cat_face_loc
    center = (int(cat_face_loc[0] + cat_face_loc[2] / 2),
              int(cat_face_loc[1] + cat_face_loc[3] * 0.8))  # 0.8为手动设定的猫嘴位置，因为没找到猫脸landmark
    # 泊松融合
    normal_clone = cv2.seamlessClone(mouth, cat_frame, mask.astype(mouth.dtype), center, cv2.NORMAL_CLONE)
    return normal_clone

# 图片融合结果
# human = cv2.imread('huaman2.png')
# cat = cv2.imread('cat.png')
# cat_with_human_mouth = human_mouth_paste_to_cat(human, cat)
# cv2.imwrite("opencv-mixed-clone-example.jpg", cat_with_human_mouth)

human_video_cap = cv2.VideoCapture("data/videos/human.mp4")
cat_video_cap = cv2.VideoCapture("data/videos/cat1.mp4")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter('data/cat_with_humanmouth.MP4', fourcc, 25, (544, 960))

index = 0
while True:
    index += 1
    human_ret, human_frame = human_video_cap.read()
    cat_ret, cat_frame = cat_video_cap.read()
    if human_ret == True and cat_ret == True:
        human_frame = cv2.resize(human_frame, dsize=None, fx=2, fy=2)
        cat_with_human_mouth = human_mouth_paste_to_cat(human_frame, cat_frame)
        video_writer.write(cat_with_human_mouth.astype(np.uint8))
        # cv2.imwrite("data/frame/%d.jpg" % index, cat_with_human_mouth)
    else:
        break

video_writer.release()

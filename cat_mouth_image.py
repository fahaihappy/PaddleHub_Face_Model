import cv2
import numpy as np
import paddlehub as hub

# 人脸关键点检测器
module = hub.Module(name="face_landmark_localization")
# 猫脸检测器
cat_path = "data/model/haarcascade_frontalcatface_extended.xml"
facecascade = cv2.CascadeClassifier(cat_path)

cat = cv2.imread('data/images/cat3.jpg')
cat_gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
cat_face_loc = facecascade.detectMultiScale(cat_gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
cat_face_loc = np.array(cat_face_loc[0])
# 猫嘴中心位置
center = (int(cat_face_loc[0] + cat_face_loc[2] / 2), int(cat_face_loc[1] + cat_face_loc[3] * 0.8))

def human_mouth_paste_to_cat(human_frame, cat_frame):
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

    normal_clone = cv2.seamlessClone(mouth, cat_frame, mask.astype(mouth.dtype), center, cv2.NORMAL_CLONE)

    return normal_clone

# 图片视频
# human = cv2.imread('huaman2.png')
# cat_with_human_mouth = human_mouth_paste_to_cat(human, cat)
# cv2.imwrite("opencv-mixed-clone-example.jpg", cat_with_human_mouth)

# 合成视频
human_video_cap = cv2.VideoCapture("data/videos/human2.mp4")

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter('data/cat_with_humanmouth_image.MP4', fourcc, 25, (1080, 2340))

index = 0
while True:
    index += 1
    human_ret, human_frame = human_video_cap.read()
    if human_ret:
        human_frame = cv2.resize(human_frame, dsize=None, fx=2, fy=2)
        cat_with_human_mouth = human_mouth_paste_to_cat(human_frame, cat)
        video_writer.write(cat_with_human_mouth.astype(np.uint8))
        # cv2.imwrite("frame/%d.jpg" % index, cat_with_human_mouth)
    else:
        break

video_writer.release()


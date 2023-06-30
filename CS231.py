import cv2
import mediapipe as mp
import numpy as np
import math

# Các hằng số để điều chỉnh ánh sáng
BRIGHTNESS_MAX = 3  # Giá trị tối đa độ sáng
BRIGHTNESS_MIN = 0  # Giá trị tối thiểu độ sáng
DISTANCE_MAX = 30  # Khoảng cách tối đa giữa hai ngón tay
DISTANCE_MIN = 0  # Khoảng cách tối thiểu giữa hai ngón tay

# Hàm tính khoảng cách giữa hai điểm
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)*100

# Hàm chuyển đổi giá trị ánh sáng dựa trên khoảng cách
def convert_brightness(distance):
    brightness_factor = BRIGHTNESS_MIN + ((distance - DISTANCE_MIN) / (DISTANCE_MAX - DISTANCE_MIN)) * (BRIGHTNESS_MAX - BRIGHTNESS_MIN)
    return brightness_factor

# Khởi tạo bộ phát hiện bàn tay và theo dõi ngón tay từ MediaPipe
mp_hands = mp.solutions.hands.Hands()

# Khởi tạo video capture
cap = cv2.VideoCapture(0)

# Khởi tạo biến
previous_distance = 0

while True:
    # Đọc frame video
    ret, frame = cap.read()

    # Chuyển đổi frame sang định dạng RGB để sử dụng với MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện bàn tay và theo dõi ngón tay
    results = mp_hands.process(frame_rgb)

    # Lấy ra tọa độ của ngón cái và ngón trỏ
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_pos = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_pos = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_x, thumb_y = int(thumb_pos.x * frame.shape[1]), int(thumb_pos.y * frame.shape[0])
            index_x, index_y = int(index_pos.x * frame.shape[1]), int(index_pos.y * frame.shape[0])
            break  # Lấy tọa độ của bàn tay đầu tiên

        # Tính toán khoảng cách giữa ngón cái và ngón trỏ
        distance = calculate_distance(thumb_pos, index_pos)

        # Chuyển đổi giá trị ánh sáng dựa trên khoảng cách
        brightness = convert_brightness(distance)

        # Điều chỉnh ánh sáng video
        frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)

        # Cập nhật khoảng cách trước
        previous_distance = distance

        # Vẽ đường kết nối từ ngón cái đến ngón trỏ
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
        print("Distande: ", distance)
        print("Brightness: ", brightness)

    # Hiển thị video
    cv2.imshow('Video', frame)
    

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhận dạng bàn tay và giải phóng video capture
mp_hands.close()
cap.release()
cv2.destroyAllWindows()


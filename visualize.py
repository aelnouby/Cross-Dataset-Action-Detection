import cv2
import numpy as np

cap = cv2.VideoCapture('/home/alaaelnouby/Desktop/Branch and Bound/stip-2.0-linux/MSR Action Dataset/1.avi')

with open('cube.txt') as f:
    content = f.readlines()

floats=[]

for i in range(int(len(content))):
    f = [float(x) for x in content[i].split()]
    floats.append(f)

arr = np.array(floats)
arr = arr.astype(np.float16)

x = arr[:, 0]
width = arr[:, 1]
y = arr[:, 2]
height = arr[:, 3]
t = arr[:, 4]
depth = arr[:, 5]
action = arr[:, 7]

print(arr)
time = 0

cv2.namedWindow('frame', flags=cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(len(arr)):
        if action[i] == 1:
            color = (0, 0, 255)
            text = "HandClapping"
        elif action[i] == 2:
            color = (0, 255, 0)
            text = "HandWaving"
        elif action[i] == 3:
            color = (255, 0, 0)
            text = "Boxing"

        # print(time)
        # print(t[i])
        # print(depth[i])
        # print("=================")
        if time >= t[i] and time <= t[i]+depth[i]:
            cv2.rectangle(frame, (x[i], y[i]), (x[i]+width[i], y[i]+height[i]), color)
            cv2.putText(frame, text, (x[i], int(y[i]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    time += 1

cap.release()
cv2.destroyAllWindows()


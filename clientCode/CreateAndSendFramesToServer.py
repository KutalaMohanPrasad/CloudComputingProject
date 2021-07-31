import cv2
import requests
import os

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Handvideo.mp4')
currentFrame = 0

if not os.path.exists('data'):
    os.makedirs('data')

while (True):
	ret, frame = cap.read()
	#url = 'http://13.82.182.203/uploader' # mohan azure VM IP address
	url = 'http://23.101.132.49/uploader' # manpreet azure VM IP address
	name = './data/frame' + str(currentFrame) + '.jpg'
	print('Creating...' + name)
	cv2.imshow("hand gestures",frame)
	cv2.imwrite(name, frame) #cv2.imwrite('Frame'+str(i)+'.jpg', frame)
	files = {'file': open(name, 'rb').read()}
	requests.post(url, files=files)
	currentFrame +=1
	if ret == False:
		break
	if cv2.waitKey(1) == ord('q'): #press q to quit
		break
		
cap.release()
cv2.destroyAllWindows()
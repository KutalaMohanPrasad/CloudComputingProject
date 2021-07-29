from flask import Flask,url_for,render_template,request,abort
import cv2
import logging
import mediapipe

app = Flask(__name__)

@app.route('/')
def WelcomeMsg():
  return 'Welcome to Hand Recongnition application! Please re-check URL once..'

@app.route('/video')
def video():
	cap=cv2.VideoCapture(0) # 0 for direct camera access
	#cap=cv2.VideoCapture('Handvideo.mp4') # give video path to read and recognise

	medhands=mediapipe.solutions.hands
	hands=medhands.Hands(max_num_hands=1,min_detection_confidence=0.7)
	draw=mediapipe.solutions.drawing_utils
	
	logging.basicConfig(filename='logs.log', level=logging.INFO)
	while True:
		success, img=cap.read()
		img = cv2.flip(img,1)
		imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
		res = hands.process(imgrgb)
		
		lmlist=[]
		tipids=[4,8,12,16,20] #list of all landmarks of the tips of fingers
		
		cv2.rectangle(img,(20,350),(90,440),(0,255,204),cv2.FILLED)
		cv2.rectangle(img,(20,350),(90,440),(0,0,0),5)
		
		if res.multi_hand_landmarks:
			for handlms in res.multi_hand_landmarks:
				for id,lm in enumerate(handlms.landmark):
					
					h,w,c= img.shape
					cx,cy=int(lm.x * w) , int(lm.y * h)
					lmlist.append([id,cx,cy])
					if len(lmlist) != 0 and len(lmlist)==21:
						fingerlist=[]
						
						#thumb and dealing with flipping of hands
						if lmlist[12][1] > lmlist[20][1]:
							if lmlist[tipids[0]][1] > lmlist[tipids[0]-1][1]:
								fingerlist.append(1)
							else:
								fingerlist.append(0)
						else:
							if lmlist[tipids[0]][1] < lmlist[tipids[0]-1][1]:
								fingerlist.append(1)
							else:
								fingerlist.append(0)
						
						#others
						for id in range (1,5):
							if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
								fingerlist.append(1)
							else:
								fingerlist.append(0)
						
						
						if len(fingerlist)!=0:
							fingercount=fingerlist.count(1)
						
						
						cv2.putText(img,str(fingercount),(25,430),cv2.FONT_HERSHEY_PLAIN,6,(0,0,0),5)
						if fingercount == 1:
							str1 = "This means that we could detect 1 finger"
							cv2.putText(img,str1, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
							logging.info(str1)
						elif fingercount == 2:
							str2 = "This means that we could detect 2 fingers"
							cv2.putText(img, str2, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
							logging.info(str2)
						elif fingercount == 3:
							str3 = "This means that we could detect 3 fingers"
							cv2.putText(img,str3, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
							logging.info(str3)
						elif fingercount == 4:
							str4="This means that we could detect 4 fingers"
							cv2.putText(img,str4, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
							logging.info(str4)
						elif fingercount == 5:
							str5="This means that we could detect 5 fingers"
							cv2.putText(img,str5, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
							logging.info(str5)
										
					#change color of points and lines
					draw.draw_landmarks(img,handlms,medhands.HAND_CONNECTIONS,draw.DrawingSpec(color=(0,255,204),thickness=2,circle_radius=2),draw.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=3))
		
		if success:
			cv2.imshow("hand gestures",img)
			if cv2.waitKey(1) == ord('q'): #press q to quit
				logging.info('You pressed "q" button')
				break
		else :
			logging.info('Video ended here...')
			errorhandler(500)
			break
		
		
		
	cv2.destroyAllWindows()
			
@app.errorhandler(500)
def internal_error(error):

    return 'You Clicked the Esc Button... Please refresh the page or  run again!!'

@app.errorhandler(404)
def not_found(error):
    return 'You Clicked the Esc Button... Please refresh the page or run again!!'
	
@app.errorhandler(Exception)
def exception_handler(error):
    return '!!\n'+internal_error(500)  #+"\n"+ repr(error)
	
if __name__ == '__main__':
  app.run()
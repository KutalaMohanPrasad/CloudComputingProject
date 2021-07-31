from flask import Flask, render_template, request,url_for,abort
from werkzeug.utils import secure_filename
import os 
import cv2
import numpy as np
import logging
import mediapipe

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:/inetpub/wwwroot/CCProject/images/'

@app.route('/')
def upload_f():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	#logging.basicConfig(filename='logs.log', level=logging.INFO)
	logging.basicConfig(filename='logs.log',format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
	try:
		if not os.path.exists('images'):
			os.makedirs('images')
	except OSError:
		print ('Error: Creating directory of images')
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],'frame.jpg'))
		#f.save("frame.jpg")
		medhands=mediapipe.solutions.hands
		hands=medhands.Hands(max_num_hands=1,min_detection_confidence=0.7)
		draw=mediapipe.solutions.drawing_utils
		count=len([item for item in os.listdir(app.config['UPLOAD_FOLDER'])])-1
		print(count)
		str1=''
		while True:
			#success, img=cap.read()
			#img = cv2.flip(img,1)
			
			img1=os.path.join(app.config['UPLOAD_FOLDER'],'frame.jpg')
			print(img1)
			img = cv2.imread(img1)
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
								str1 += "This means that we could detect 1 finger \n"
								cv2.putText(img,str1, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
								logging.info(str1)
							elif fingercount == 2:
								str1 += "This means that we could detect 2 fingers"
								cv2.putText(img, str1, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
								logging.info(str1)
							elif fingercount == 3:
								str1 += "This means that we could detect 3 fingers"
								cv2.putText(img,str1, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
								logging.info(str1)
							elif fingercount == 4:
								str1 +="This means that we could detect 4 fingers"
								cv2.putText(img,str1, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
								logging.info(str1)
							elif fingercount == 5:
								str1 +="This means that we could detect 5 fingers"
								cv2.putText(img,str1, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
								logging.info(str1)
											
						#change color of points and lines
						draw.draw_landmarks(img,handlms,medhands.HAND_CONNECTIONS,draw.DrawingSpec(color=(0,255,204),thickness=2,circle_radius=2),draw.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=3))
						
			cv2.imshow("hand gestures",img)
			count =count-1
			if(count<0):
				logging.info('Given frames completed')
				break
				
			if cv2.waitKey(1) == ord('q'): #press q to quit
				logging.info('You pressed "q" button')
				break
			
			
		cv2.destroyAllWindows()
		return 'file uploaded successfully.. File Name:'+secure_filename(f.filename)+'Response From Server:'+str1

if __name__ == '__main__':
	app.run(debug = False)
# CloudComputingProject
Hand Recognition Project

It will Recognize the number of fingers from the segmented hand region using local machine webcam

Prerequisites
Python 3.5
OpenCV2
MediaPipe from google

Please install above prerequisites by Pip install command in Console.

Create the Microsoft Azure Windows virtual machine and update the IP address inside CreateAndSendFramesToServer.py (line number 15) and start the server it should access the handRecognition.py usng web.config (*change the details inside configuration file*)

After launching the server in AZURE use the below command to run flask in local machine console

python CreateAndSendFramesToServer.py 

it will open the camera and pass the each webcam frame to server and server will log the output in the given server location 

sample output looks like below:

Frames colelcted in local machine 
![frame5](https://user-images.githubusercontent.com/77629263/127752023-89123b8d-6268-4800-9905-54756bae82c4.jpg)

below the output given the server and stored in the VM

![image](https://user-images.githubusercontent.com/77629263/127752046-9f546a6c-1473-4483-a0ec-86d7ca18623a.png)

if you want to quit the webcam and process just press 'q' button in local machine keyboard.

how to run a flask app in IIS.

Clone this server code repo to wwwroot

Install Python in the root directory

Enable CGI for IIS

Install wfastcgi from pip

Ensure proper file permissions

Configure Web.config

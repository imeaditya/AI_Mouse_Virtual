# Stamatics Project: Computer Vision

Team Members:                  
Aditya Agarwal - 190058                 
Hitesh Anand - 200449                 
Ashutosh Sharma - 200216                  

# AI Virtual Mouse

## Documentation

## OBJECTIVE:- 

Through this project, we have created a virtual mouse. The principle  of this virtual mouse is based upon hand detection. We detect the fingers and use the information accordingly to move the cursor on the screen and click as and when required. 

If our index finger is up and the middle finger is folded, then it indicates that we want only to move the cursor on the screen without clicking anywhere.

If both our index and middle fingers are out and the distance between the tips of both the fingers is less than a particular value, then it leads to a mouse click. 

## DEPENDENCIES:-

OpenCV               
Numpy                 
Mediapipe                
Autopy                
Time                  

## AI_Mouse_Implementation.py

Set the value of variable    
  -> wCam, hCam are the dimensions of the camera frame.
  -> frameR helps in frame reduction. Otherwise it becomes 
     difficult to access the extreme parts of the screen. Hence we 
 create a rectangle  to confine the movement of our hand within 
 the rectangle. Corners and extreme parts of the rectangle  correspond to the respective parts of the screen.                                                                                             
  -> Smoothening corrects the problem of flickering
     of the mouse.
  -> While using cap = cv2.VideoCapture(0), ( 0 or 1 
     according to our device camera ), captures the video.

## Find hand landmark
-> We create an object detector using the
   HandTrackingModule.
   using findHands() and findPosition(), we locate the hands 
   , and get a list of landmarks and a bounding box respectively.


## Get the tip of index and middle fingers:-

-> we get the coordinates of the index and middle fingers given 
   by (x1,y1) and (x2, y2).

## Check which fingers are up:-

-> fingers is an array of length 5 where index i corresponds to 
   the state of each individual finger. For ex.- if finger[1] = 1, it 
   means that our index finger is up and it is down if finger[1]=0.
-> Also, we put a rectangle around our hand having the start 
   Coordinates and end coordinates as (frameR, frameR) and
   (wCam-frameR, hCam-frameR).

## Only index finger up:- moving mode

-> if finger[1]=1 and finger[2]=0, it means that we are in only moving mode.

## Convert the coordinates:-

-> We convert the coordinates into the coordinate system of our
   screen(using the values of wScr and hScr --dimensions of 
   our screen (autopy.screen.size() gives their values).

Smoothen the values of coordinates:-

-> Non-smoothened coordinates cause flickering of cursor 
   leading to an uncomfortable experience for the user. Hence
   we smoothen the coordinates using the smoothening 
   variable defined earlier.
-> Try out different values to check the optimal value. Higher
   values of smoothening lead to slower movement of cursor.

   clocX = plocX + (x3-plocX)
   clocY = plocY + (y3-plocY)

   Where (clocX, clocY) and (plocX, plocY) are the current and the 
   previous coordinates of the cursor. 
		
   Consider the case when plocX = plocY = clocX = clocY = 0
   initially. 
   In absence of any smoothening, there will be a direct 
   transition in the coordinates of the cursor from (0,0) to (x3, y3) 
   which will cause significant flickering of the cursor. 
				
   But after we apply smoothening, there will be a smoother 
   Transition from (0,0) to (x3/smoothening, y3/smoothening) 
   Instead of a direct transition between discrete coordinates.

Hence, we try multiple values of smoothening to choose an optimal value. 


## Move mouse:-

-> We can move the cursor using the autopy library and put a 
    Circle around the tip of our index finger.

autopy.mouse.move(wScr-clocX, clocY)

where (wScr - clocX, clocY) are the coordinates of the location
at which we desire to move our cursor. 
		
Since our webcam will generally display a horizontally inverted 
Image of our hands, the cursor will move in the horizontally 
Opposite direction to the direction of our hand’s movement.

To solve this problem, we replace clocX by wScr - clocX.

Assume that clocX = clocY = 20 and wScr = 100 
and previously the cursor was at (21,20) in the frame. 

In order to move at (clocX, clocY) in the frame the cursor is required to move towards left on screen. 
But due to horizontal inversion, 
We would have to move our hands towards right causing 
inconvenience.

Therefore, we replace the desired x-coordinate with wScr-clocX
where wScr is the horizontal width of our screen and clearly,

Horizontal distance of a point from left side of screen + 
Horizontal distance from right side of screen = wScr.

In the above example, instead of using
autopy.mouse.move(20, 20),
We use autopy.mouse.move(80,20).

Hence, to address the problem of horizontal inversion, we use 
wScr-clocX instead of clocX.

Note that we use clocY as it is because there is no vertical
Inversion. We would have used hScr-clocY in case of 
Vertical inversion, where hScr is the (vertical) height of our 
screen.

## Find distance between fingers:-(when both index and middle fingers are up)

-> We extract the distance between the tips of our middle finger 
   and index finger using findDistance(). 

-> Click if distance between fingertips < 40(Generally it is almost the least possible distance between the centre of 2 fingers):-

-> if the distance obtained in previous step is less than a
   particular value, then click.
-> Also, we change the colour of the circle at the mid point of
   line joining the tips of both the fingers.

## Adjust Frame Rate and Display
		
-> adjust the frame rate and use imshow() to display.
  -> The screen shows the frame rate in the top left of the Display.

Frame rate is calculated in frames per second :-

fps = 1/(cTime - pTime)
		
where cTime = current recorded time
pTime = previously recorded time 


## Hand_Tracking_Model.py:

![Screenshot (275)](https://user-images.githubusercontent.com/68987597/123673137-bf6bc000-d85d-11eb-9230-24df21a41c03.png)

Creating Video Object:                      
-> Using openCV video capture we use our webcam through our python code.
		
Using the MediaPipe to process and detect:                             
->Using mphands.Hands() we detect the hands and also we don’t give any new configurations to the arguments taken by this module i.e. it will detect at max two hands and also if the image detection confidence if less than 50% then it will not detect so as to speed up the detection process.
Inside the loop our rgb image is sent to this function.

Displaying the points:                           
->By results.multi_hand_landmarks we get any output if there is any hand in the frame of the webcam , so if there is a hand in the frame of the webcam we run a for loop over each hand and show the hand landmarks.
 
Showing the Frame rate:                      
->Doing the reciprocal of the time difference when the last and current frame was displayed we get the frame rate , and display that on the window.

Getting Position of Landmarks:                   
We can use id’s which are already listed in the module , which shows the x ,y and z positions of the landmark in ratio of the size of image which is converted into coordinates by multiplying by the shape of image .

 Creating a class to initialize them:                   
 We have created a class named handDetector in which it stores all the parameters required for mp.hands so that we have the flexibility of changing them.
Basically we have created an object which has its own variables values which can be either given by the user manually or they already have some initial value.
Objects are :
Mode , maxHands , detectionCon(Detection Confidence) , mp.Hands
Mp.hands takes all the four of the above as params

Defining findHands:                     
This function will be taking img and self as arguments and will be doing the part described in getting position of the landmarks point . 
Also it will show coloured circles on the coordinates of the landmarks on the hands if we ask it to draw.
Defining findPosition:
This function will return a list containing the positions of all the landmarks by iterating over each hand present in the webcam frame and store them in list lmlist[] only if there is any hand present in the frame which will be checked by the if condition which checks that if any hand is even recognized.
Also this list will be printed only if it has any elements which can be also done this by an if condition.

 Restructuring code :                         
 Now we have restructured the code into functions and calling         
 functions at appropriate places which makes code cleaner and easier to understand. 






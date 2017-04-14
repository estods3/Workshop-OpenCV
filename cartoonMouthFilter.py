#import openCV functionality using cv2
import cv2

#SETUP OBJECT DETECTORS
# Setup Haar detection for mouth and face
mouthDet = cv2.CascadeClassifier('lips.xml')
faceDet = cv2.CascadeClassifier('face.xml')

#IMPORT IMAGE
#import an image "mouth.png" as data stored in variable "mouthImage"
#this is the image to be inserted over the mouth. This image should be small, less than 80 by 40 pixels. this will
# vary with the shape of the object. for instance, it may need to be smaller (say 20 by 20 pixels) for an object such as an eye
mouthImage = cv2.imread('mouth.png')

#INITIALIZE VIDEO CAPTURING
# Initialize video capture from webcam ("0")
cap = cv2.VideoCapture(0)


#loop through the frames found from the videoCapture while the video capturing webcam "cap" is opened
while(cap.isOpened()):
    # Capture frame-by-frame and store in variables "ret" and "frame". "frame" will hold the actual image
    ret, frame = cap.read()
    #if there is data in variable "frame", proceed to the body of the "if" statement. Oherwise, "frame" == None.
    if(frame != None):

        #GRAYSCALE
        #grayscale the image "frame" so we can run an object detector on it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FACE DETECTION
        # use the detector "faceDet" to detect faces in each frame identified by the webcam
        faceFound = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(1, 1), flags=cv2.CASCADE_SCALE_IMAGE)

        #the detector may find multiple faces in any given frame. loop through the faces that have been found
        #these faces are stored as X,Y,W,H coordinates in an array called "faceFound"
        for (faceX, faceY, faceW, faceH) in faceFound:
            #outline detected face with a green box
            cv2.rectangle(frame, (faceX,faceY), (faceX+faceW,faceY+faceH), (0,255,0), 2) #com

            #REGION OF INTEREST (ROI)
            #since a face has been found, look for a mouth in the region where we would expect to see a mouth on a face
            xROIstart = faceX
            xROIend = faceX+faceW
            #define the y axis ROI to be the bottom 1/3 of the face. After all, this is where we expect to find a mouth
            yROIstart = faceY + 0.666*faceH
            yROIend = faceY + faceH
            #assign the gray scaled image "gray" to be its smaller Region of Interest that we defined above
            gray = gray[(int)(yROIstart):(int)(yROIend), (int)(xROIstart):(int)(xROIend)]

            #MOUTH DETECTION
            #use the detector "mouthDet" to detect mouths in each face in array "faceFound"
            mouthFound = mouthDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(1, 1),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
            #check that mouthFound isn't of length 0. if it is, then the detector didn't find a mouth
            if(len(mouthFound) != 0):
                # Since we only have one mouth per face, we can just look at the first mouth(0) found in array "mouthFound"
                mouthX, mouthY, mouthW, mouthH = mouthFound[0]
                #the detected mouth's coordinates assume that (0,0) is the top corner of the ROI if was detecting in
                #we need to offset these values by adding the actual (x,y) coordinates (xROIstart,yROIstart)
                #try removing these line, and you will see the error on the screen
                mouthX += (int)(xROIstart)
                mouthY += (int)(yROIstart)
                #outline detected mouth with a red box
                cv2.rectangle(frame, (mouthX,mouthY), (mouthX+mouthW,mouthY+mouthH), (0,0,255), 2)

                #SET START AND END COORDINATES FOR MOUTH IMAGE
                #set start coordinates to the same at the detected mouth
                xstartofImage = mouthX
                ystartofImage = mouthY
                #set end coordinates to include the width(mouthImage.shape[1]) and height(mouthImage.shape[0]) of the
                #image "mouthImage" and not of the detected mouth's height and width because they might not match exactly
                xendofImage =  xstartofImage + mouthImage.shape[1]
                yendofImage = ystartofImage + mouthImage.shape[0]
                #paste the image "mouthImage" into the portion of "frame" outlined by the coordinates above
                frame[ystartofImage:yendofImage, xstartofImage:xendofImage] = mouthImage

        # Display frame with rectangles
        cv2.imshow('img_objdetected', frame)

    # Quit by pressing "q" on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
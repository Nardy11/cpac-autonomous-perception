import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--video', required=True, help='road video to process')
args = parser.parse_args()


def make_coordinates(image ,line_params):
    image =cv2.resize(image,(500,500))
    slope,intercept = line_params
    y1 =image.shape[0]
    y2 =int(y1*(3/5))
    x1 =int((y1-intercept)/slope) 
    x2 =int((y2-intercept)/slope) 
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image ,lines):
    image =cv2.resize(image,(500,500))
    left_fit = []
    right_fit =[]
    for line in lines:
        x1,y1,x2,y2 =line.reshape(4)
        params = np.polyfit((x1,x2),(y1,y2),1)
        slope=params[0]
        intercept=params[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    
    left_fit_average=np.average(left_fit , axis=0)
    right_fit_average=np.average(right_fit , axis=0)

    left_line=make_coordinates(image , left_fit_average)
    right_line=make_coordinates(image , right_fit_average)

    return np.array([left_line,right_line])

def canny(image):
    image =cv2.resize(image,(500,500))
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    image =cv2.resize(image,(500,500))
    line_image =np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 =line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    image =cv2.resize(image,(500,500))
    height =image.shape[0]
    polygons =np.array([
        [(100,height),(420,height),(300,282)]
    ])
    mask =np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image =cv2.bitwise_and(image,mask)
    return masked_image

cap= cv2.VideoCapture(args.video)
while(cap.isOpened):
    ret,frame=cap.read()
    if not ret :
        break
    canny_image =canny(frame)
    cropped_image =region_of_interest(canny_image)
    lines =cv2.HoughLinesP(canny_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=280)
    averaged_lines =average_slope_intercept(frame,lines)
    line_image =display_lines(frame,averaged_lines)
    final_image =cv2.addWeighted(cv2.resize(frame,(500,500)),0.8,line_image,1,1)
    print(lines)
    # plt.imshow(canny_image)
    cv2.imshow('final', cv2.resize(final_image,(500,500)))
    # plt.show()
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

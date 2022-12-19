from cv2 import cvtColor, boundingRect, threshold, findContours, COLOR_BGR2GRAY, THRESH_BINARY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from numpy import ndarray

def crop_borders(img):
  gray = cvtColor(img, COLOR_BGR2GRAY)
  _,thresh = threshold(gray, 5, 255, THRESH_BINARY)
  contours,_ = findContours(thresh, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
  cnt = contours[0]
  x,y,w,h = boundingRect(cnt)
  return img[y:y+h,x:x+w]
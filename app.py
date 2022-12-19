import os
import pickle
from io import BytesIO
from cv2 import cvtColor, imdecode, findContours, boundingRect, rectangle, COLOR_BGR2RGB, COLOR_BGR2GRAY, IMREAD_COLOR, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from numpy import asarray, ndarray, uint8
from streamlit import set_page_config, container as get_container, write

from functions.crop_borders import crop_borders

# Set webpage title
set_page_config(
    page_title="Spot the Difference",
    page_icon="🐱‍🐉",
)

# Create webpage body
body = get_container()

# Write header
header = body.container()
header.title("Spot the Difference Solver")
header.caption("13519128 Jonathan Richard Sugandhi")

# STEP 1
# Ask user to upload an image
step1 = body.container()
step1.header("Upload Images")
uploader1, uploader2 = step1.columns([1,1])
uploaded_file1: BytesIO = uploader1.file_uploader("1", label_visibility="collapsed")
uploaded_file2: BytesIO = uploader2.file_uploader("2", label_visibility="collapsed")

image_viewer = body.container()
image_viewer.header("Uploaded Images:")
image_viewer1, image_viewer2 = image_viewer.columns([1,1])


if uploaded_file1 is not None:
    rgb_image1: ndarray = cvtColor(imdecode(asarray(bytearray(uploaded_file1.read()), dtype=uint8), IMREAD_COLOR), COLOR_BGR2RGB)
    rgb_image1 = crop_borders(rgb_image1)
    image_viewer1.image(rgb_image1)
if uploaded_file2 is not None:
    rgb_image2: ndarray = cvtColor(imdecode(asarray(bytearray(uploaded_file2.read()), dtype=uint8), IMREAD_COLOR), COLOR_BGR2RGB)
    rgb_image2 = crop_borders(rgb_image2)
    image_viewer2.image(rgb_image2)

if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        result = body.container()
        result.header("Result:")
        result1, result2 = result.columns([1,1])
        result_image1 = rgb_image1.copy()
        result_image2 = rgb_image2.copy()

        gray_diff = cvtColor(rgb_image1 - rgb_image2, COLOR_BGR2GRAY)
        contours, _ = findContours(gray_diff, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        for countour in contours:
            x,y,w,h = boundingRect(countour)
            rectangle(result_image1,(x,y),(x+w,y+h),(0,255,0),2)
            rectangle(result_image2,(x,y),(x+w,y+h),(0,255,0),2)
        write("Total difference:", len(contours))
        result1.image(result_image1)
        result2.image(result_image2)
    except:
        write("Image resolutions must match!")

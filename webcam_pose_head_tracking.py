import numpy as np
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import math

import senet_feedback
import pdb

def drawBodyPart(img,start,end,color,output_2d_x,output_2d_y):
	start_line = (output_2d_x[start],output_2d_y[start])
	end_line = (output_2d_x[end],output_2d_y[end])
	#return cv2.line(img,start_line,end_line,(255,0,0),5)
	return cv2.line(img,start_line,end_line,(color[2],color[1],color[0]),5)

def webcam_pose():

	feedback_cycles = 1
	feedback_norm = False
	disable_se_layer = False
	dims_3d = 51
	dims_2d = 32
	model = senet_feedback.SEFeedback(feedback_cycles, feedback_norm, disable_se_layer,dims_2d+dims_3d)

	model_path = './model/net_run_fri_1_june_5.pyth'   # trained on h36m and mpii for 32 epochs and L1 norm cost function

	print('loading',model_path)
	model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	model.eval()	
	print('model loaded')

	cascade_root = '/Users/niallmcl/anaconda3/share/OpenCV/haarcascades/'
	face_cascade = cv2.CascadeClassifier(cascade_root + 'haarcascade_frontalface_default.xml')
	#face_cascade = cv2.CascadeClassifier(cascade_root + 'haarcascade_upperbody.xml')
	#face_cascade = cv2.CascadeClassifier(cascade_root + 'haarcascade_fullbody.xml')

	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, img = cap.read()

		#img = cv2.imread('./testdata/1.png')

		# detect persons
		# crop person
		# send to network

		# Our operations on the frame come here
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		detections = face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(10,10), maxSize=(200,200))

		if len(detections) > 0:
			for (x,y,w,h) in detections:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

				center_head_x = x + w//2
				center_head_y = y + h//2
				width_head = w
				height_head = h

				# from the head extrapolate to the full body size

				h = height_head * 9.7
				w = h

				crop_size = int(h)

				center_x = center_head_x
				center_y = y + h // 2

				y = (center_y - h // 2)
				x = (center_x - h // 2)

				cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)

				crop_img = img[y:y+h, x:x+w]

				if crop_img.shape[0] > 100 and crop_img.shape[1] > 100:
					resized_img = cv2.resize(crop_img,(128,128))

					# resized_img = cv2.resize(crop_img,(128,128))
					rawImg = torch.from_numpy(resized_img)
					rawImg = rawImg.permute((2,0,1))
					rawImg = rawImg.float() / 255

					output = model.forward(Variable(rawImg.unsqueeze(0), requires_grad=False))
					output_2d = output[0][1]

					output_2d = output_2d + 0.5
					output_2d = torch.floor(output_2d * crop_size)

					output_2d_x = output_2d[0][0:16] + int(x)
					output_2d_y = output_2d[0][16:] + int(y)

					#crop_img = cv2.resize(crop_img,(display_size,display_size))

					for i in range(16):
						cv2.circle(img,(int(output_2d_x[i].item()),int(output_2d_y[i].item())), 5, (100,255,100), -1)

					rankle = 0 
					rknee = 1 
					rhip = 2 
					lhip = 3 
					lknee = 4 
					lankle = 5 
					pelvis = 6 
					thorax = 7 
					upperneck = 8 
					headtop = 9 
					rwrist = 10 
					relbow = 11 
					rshoulder = 12 
					lshoulder = 13 
					lelbow = 14 
					lwrist = 15 

					colors = [(165,0,38),(215,48,39),(244,109,67),(253,174,97),(254,224,144),(255,255,191),(224,243,248),(171,217,233),(116,173,209),(69,117,180),(49,54,149)]

					img = drawBodyPart(img,upperneck,headtop,colors[5],output_2d_x,output_2d_y)
					img = drawBodyPart(img,thorax,upperneck,colors[5],output_2d_x,output_2d_y)

					img = drawBodyPart(img,thorax,pelvis,colors[5],output_2d_x,output_2d_y)
					img = drawBodyPart(img,rhip,rknee,colors[1],output_2d_x,output_2d_y)
					img = drawBodyPart(img,rknee,rankle,colors[0],output_2d_x,output_2d_y)
					img = drawBodyPart(img,lhip,lknee,colors[10],output_2d_x,output_2d_y)
					img = drawBodyPart(img,lknee,lankle,colors[9],output_2d_x,output_2d_y)

					img = drawBodyPart(img,lshoulder,lelbow,colors[9],output_2d_x,output_2d_y)
					img = drawBodyPart(img,lelbow,lwrist,colors[10],output_2d_x,output_2d_y)

					img = drawBodyPart(img,rshoulder,relbow,colors[1],output_2d_x,output_2d_y)
					img = drawBodyPart(img,relbow,rwrist,colors[0],output_2d_x,output_2d_y)

					img = drawBodyPart(img,pelvis,lhip,colors[7],output_2d_x,output_2d_y)
					img = drawBodyPart(img,pelvis,rhip,colors[4],output_2d_x,output_2d_y)

					img = drawBodyPart(img,thorax,thorax,colors[5],output_2d_x,output_2d_y)
					img = drawBodyPart(img,thorax,rshoulder,colors[4],output_2d_x,output_2d_y)
					img = drawBodyPart(img,thorax,lshoulder,colors[7],output_2d_x,output_2d_y)

					# cv2.line(crop_img,(0,0),(100,100),(255,0,0),5)
					# cv2.circle(crop_img,(100,100), 5, (0,255,0), -1)

					# Display the resulting frame
					#cv2.imshow('frame',gray)
					# cv2.imshow('frame',img)
					# if cv2.waitKey(1) & 0xFF == ord('q'):
					# 	break

		cv2.imshow('frame',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

webcam_pose()
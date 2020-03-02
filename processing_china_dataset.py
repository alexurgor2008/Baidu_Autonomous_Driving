from math import sin, cos,acos,pi,sqrt
import numpy as  np
import cv2 
import os
import pandas as pd
import matplotlib.pylab as plt
# Utility functions
################################***DECLARATIONS***######################################################################
IMAGE_DIR = 'Dataset/train_images'
BOX2D_LOC_DIR = 'Dataset/train_images/box_2d'
BOX3D_LOC_DIR = 'Dataset/train_images/box_3d'

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)
########################################################################################################################

def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))
def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image
        ys: y coordinates in the image
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    return img_xs, img_ys
   
def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image
    
def calc_angle(r):
	x = r[0]
	y = r[1]
	z = r[2]
	
	
	return acos(z/sqrt(x**2 + y**2 + z**2))   
def calc_angle_by_data(point):
	x, y, z = point['x'], point['y'], point['z']
	#return int(180*acos(z/sqrt(x**2 + y**2 + z**2)) /pi) # градусы
	return acos(z/sqrt(x**2 + y**2 + z**2)) # радианы
	    
def draw_3dbox(image,points):
    color = (0, 255, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 4)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 4)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 4)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 4)
    
    cv2.line(image, tuple(points[4][:2]), tuple(points[7][:2]), color, 4)
    cv2.line(image, tuple(points[4][:2]), tuple(points[5][:2]), color, 4)
    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 4)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 4)
    
    cv2.line(image, tuple(points[0][:2]), tuple(points[4][:2]), color, 4)
    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 4)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 4)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 4)
    
    return image
def draw_rects(image,points):
	X = np.array([p[0] for p in points])
	Y = np.array([p[1] for p in points])
	color = (255, 0, 0)
	xmin = X.min()
	xmax = X.max()
	ymin = Y.min()
	ymax = Y.max()
	cv2.line(image,(xmin,ymin),(xmax,ymin),color,8)
	cv2.line(image,(xmax,ymin),(xmax,ymax),color,8)
	cv2.line(image,(xmax,ymax),(xmin,ymax),color,8)
	cv2.line(image,(xmin,ymax),(xmin,ymin),color,8)
	return image
def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    return image
def get_2dbox(image,points):
	X = np.array([p[0] for p in points])
	Y = np.array([p[1] for p in points])
	
	xmin = X.min()
	xmax = X.max()
	ymin = Y.min()
	ymax = Y.max()
	return np.array([xmin,xmax,ymin,ymax])



def visualize(img, coords):
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [x_l, y_l, -z_l, 1],
                      [x_l, y_l, z_l, 1],
                      [-x_l, y_l, z_l, 1],
                      [-x_l, y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        #print(img_cor_points)
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        #img = draw_line(img, img_cor_points)
        
        #рисуем 3D  боксы
        img = draw_3dbox(img, img_cor_points)
        
        # рисуем прямоугольники
        draw_rects(img,img_cor_points)
        
        #печатаем величины углов theta_ray в радианах
        print(calc_angle_by_data(point))
        #img = draw_points(img, img_cor_points[-1:])
        #cv2.putText(img, str(int(z)), (img_cor_points[-1,0], img_cor_points[-1,1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
    
    return img
DATASET_DIR = "./"#"/media/klchnv/LinuxExt4/Peking University_Baidu - Autonomous Driving/"
train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
fname = 'ID_0a1d250a1' 
fpath = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(fname, 'jpg'))
img = cv2.imread(fpath)
print(img.shape)
image_idx = train.loc[train['ImageId'] == fname].index[0]
img_vis = visualize(img, str2coords(train['PredictionString'][image_idx]))

plt.figure(figsize=(20,20))
plt.imshow(img_vis)
plt.title('2D (3D) боксы')
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
def image_processing(image, coords):

	all_image = sorted(os.listdir(IMAGE_DIR))
	for f in all_image:
		image_file = IMAGE_DIR + f
		box2d_file = BOX2D_LOC_DIR + f.replace('jpg', 'txt')
		box3d_file = BOX3D_LOC_DIR + f.replace('jpg', 'txt')

		with open(box3d_file, 'w') as box3d:
			img = cv2.imread(image_file)
			img = img.astype(np.float32, copy=False)

			for line in open(box2d_file):
				line = line.strip().split(' ')
				truncated = np.abs(float(line[1]))
				occluded = np.abs(float(line[2]))

				obj = {'xmin': int(float(line[4])),
				       'ymin': int(float(line[5])),
				       'xmax': int(float(line[6])),
				       'ymax': int(float(line[7])),
				       }

				patch = img[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
				patch = cv2.resize(patch, (NORM_H, NORM_W))
				patch = patch - np.array([[[103.939, 116.779, 123.68]]])
				patch = np.expand_dims(patch, 0)

				prediction = model.predict(patch)

				# Transform regressed angle
				max_anc = np.argmax(prediction[2][0])
				anchors = prediction[1][0][max_anc]

				if anchors[1] > 0:
					angle_offset = np.arccos(anchors[0])
				else:
					angle_offset = -np.arccos(anchors[0])

				wedge = 2. * np.pi / BIN
				angle_offset = angle_offset + max_anc * wedge
				angle_offset = angle_offset % (2. * np.pi)

				angle_offset = angle_offset - np.pi / 2
				if angle_offset > np.pi:
					angle_offset = angle_offset - (2. * np.pi)

				line[3] = str(angle_offset)

				# Transform regressed dimension
				dims = dims_avg['Car'] + prediction[0][0]

				line = line + list(dims)

				# Write regressed 3D dim and oritent to file
				line = ' '.join([str(item) for item in line]) + '\n'
				box3d.write(line)

	return img


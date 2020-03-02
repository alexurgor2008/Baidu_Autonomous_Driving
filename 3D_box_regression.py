from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from Models import VGGnet_regr
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import copy
import cv2, os
import numpy as np
from random import shuffle
#get_ipython().magic(u'matplotlib inline')

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
CHAN = 3
VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
BATCH_SIZE = 8

IMAGE_DIR = 'Dataset/train_images'
BOX2D_LOC_DIR = 'Dataset/train_images/box_2d'
BOX3D_LOC_DIR = 'Dataset/train_images/box_3d'
#label_dir = '/home/husky/data/kitti_object/data_object_image_2/training/label_2/'
#image_dir = '/home/husky/data/kitti_object/data_object_image_2/training/image_2/'
visual_architect_dir = 'Models/Visualize_architect'
label_dir = BOX2D_LOC_DIR
image_dir = IMAGE_DIR

#-----------------------------Определение шаблонов выделения------------------------------------------------------------
def compute_anchors(angle):
	anchors = []

	wedge = 2. * np.pi / BIN
	l_index = int(angle / wedge)
	r_index = l_index + 1

	if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
		anchors.append([l_index, angle - l_index * wedge])

	if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
		anchors.append([r_index % BIN, angle - r_index * wedge])

	return anchors

#-----------------------------Вычисление угла---------------------------------------------------------------------------
def compute_angle(anchors):
	pass

#-----------------------------Парсинг аннотаций-------------------------------------------------------------------------
def parse_annotation(label_dir, image_dir):
	all_objs = []
	dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
	dims_cnt = {key: 0 for key in VEHICLES}

	for label_file in os.listdir(label_dir):
		image_file = label_file.replace('txt', 'png')

		for line in open(label_dir + label_file).readlines():
			line = line.strip().split(' ')
			truncated = np.abs(float(line[1]))
			occluded = np.abs(float(line[2]))

			if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
				new_alpha = float(line[3]) + np.pi / 2.
				if new_alpha < 0:
					new_alpha = new_alpha + 2. * np.pi
				new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

				obj = {'name': line[0],
				       'image': image_file,
				       'xmin': int(float(line[4])),
				       'ymin': int(float(line[5])),
				       'xmax': int(float(line[6])),
				       'ymax': int(float(line[7])),
				       'dims': np.array([float(number) for number in line[8:11]]),
				       'new_alpha': new_alpha
				       }

				dims_avg[obj['name']] = dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']
				dims_cnt[obj['name']] += 1
				dims_avg[obj['name']] /= dims_cnt[obj['name']]

				all_objs.append(obj)

	return all_objs, dims_avg

#-----------------------------Подготовка входного и выходного обучающего наборов----------------------------------------
def prepare_input_and_output(train_inst):
	### Prepare image patch
	xmin = train_inst['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
	ymin = train_inst['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
	xmax = train_inst['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
	ymax = train_inst['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)

	img = cv2.imread(image_dir + train_inst['image'])
	img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)

	# re-color the image
	# img += np.random.randint(-2, 3, img.shape).astype('float32')
	# t  = [np.random.uniform()]
	# t += [np.random.uniform()]
	# t += [np.random.uniform()]
	# t = np.array(t)

	# img = img * (1 + t)
	# img = img / (255. * 2.)

	# flip the image
	flip = np.random.binomial(1, .5)
	if flip > 0.5: img = cv2.flip(img, 1)

	# resize the image to standard size
	img = cv2.resize(img, (NORM_H, NORM_W))
	img = img - np.array([[[103.939, 116.779, 123.68]]])
	# img = img[:,:,::-1]

	### Fix orientation and confidence
	if flip > 0.5:
		return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
	else:
		return img, train_inst['dims'], train_inst['orient'], train_inst['conf']

#-------------------------------------------------Генерация данных------------------------------------------------------
def data_gen(all_objs, batch_size):
	num_obj = len(all_objs)

	keys = range(num_obj)
	np.random.shuffle(keys)

	l_bound = 0
	r_bound = batch_size if batch_size < num_obj else num_obj

	while True:
		if l_bound == r_bound:
			l_bound = 0
			r_bound = batch_size if batch_size < num_obj else num_obj
			np.random.shuffle(keys)

		currt_inst = 0
		x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
		d_batch = np.zeros((r_bound - l_bound, 3))
		o_y_batch = np.zeros((r_bound - l_bound, BIN, 2))
		c_y_batch = np.zeros((r_bound - l_bound, BIN))
		o_p_batch = np.zeros((r_bound - l_bound, BIN, 2))
		c_p_batch = np.zeros((r_bound - l_bound, BIN))
		o_r_batch = np.zeros((r_bound - l_bound, BIN, 2))
		c_r_batch = np.zeros((r_bound - l_bound, BIN))
		"""
		o_batch = np.zeros((r_bound - l_bound, BIN, 2))
		c_batch = np.zeros((r_bound - l_bound, BIN))
		"""

		for key in keys[l_bound:r_bound]:
			# augment input image and fix object's orientation and confidence
			image, orientation_yaw, confidence_yaw, orientation_pitch, \
			confidence_pitch, orientation_roll, confidence_roll = prepare_input_and_output(all_objs[key])
			#image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

			# plt.figure(figsize=(5,5))
			# plt.imshow(image/255./2.); plt.show()
			# print dimension
			# print orientation
			# print confidence

			x_batch[currt_inst, :] = image
			#d_batch[currt_inst, :] = dimension
			o_y_batch[currt_inst, :] = orientation_yaw
			c_y_batch[currt_inst, :] = confidence_yaw
			o_p_batch[currt_inst, :] = orientation_pitch
			c_p_batch[currt_inst, :] = confidence_pitch
			o_r_batch[currt_inst, :] = orientation_roll
			c_r_batch[currt_inst, :] = confidence_roll
			"""
			o_batch[currt_inst, :] = orientation
			c_batch[currt_inst, :] = confidence
			"""

			currt_inst += 1

		#yield x_batch, [d_batch, o_batch, c_batch]
		yield x_batch, [o_y_batch, c_y_batch, o_p_batch, c_p_batch, o_r_batch, c_r_batch]

		l_bound = r_bound
		r_bound = r_bound + batch_size
		if r_bound > num_obj: r_bound = num_obj

#--------------------------------------------Нормализация двумерного вектора--------------------------------------------
def l2_normalize(x):
	return tf.nn.l2_normalize(x, dim=2)
#-----------------------------------------Вычисление ошибки ориентации--------------------------------------------------
def orientation_loss(y_true, y_pred):
	# Find number of anchors
	anchors = tf.reduce_sum(tf.square(y_true), axis=2)
	anchors = tf.greater(anchors, tf.constant(0.5))
	anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

	# Define the loss
	loss = -(y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
	loss = tf.reduce_sum(loss, axis=1)
	loss = loss / anchors

	return tf.reduce_mean(loss)

########################################################################################################################
# check to see if the main thread should be started
if __name__ == "__main__":
	all_objs, dims_avg = parse_annotation(label_dir, image_dir)

	for obj in all_objs:
		# Fix dimensions
		obj['dims'] = obj['dims'] - dims_avg[obj['name']]

		# Fix orientation and confidence for no flip
		orientation = np.zeros((BIN, 2))
		confidence = np.zeros(BIN)

		anchors = compute_anchors(obj['new_alpha'])

		for anchor in anchors:
			orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
			confidence[anchor[0]] = 1.

		confidence = confidence / np.sum(confidence)

		obj['orient'] = orientation
		obj['conf'] = confidence

		# Fix orientation and confidence for flip
		orientation = np.zeros((BIN, 2))
		confidence = np.zeros(BIN)

		anchors = compute_anchors(2. * np.pi - obj['new_alpha'])

		for anchor in anchors:
			orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
			confidence[anchor[0]] = 1

		confidence = confidence / np.sum(confidence)

		obj['orient_flipped'] = orientation
		obj['conf_flipped'] = confidence

	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
	checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',
	                             period=1)
	tensorboard = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True, write_images=False)

	all_exams = len(all_objs)
	trv_split = int(0.9 * all_exams)
	batch_size = 8
	np.random.shuffle(all_objs)

	train_gen = data_gen(all_objs[:trv_split], batch_size)
	valid_gen = data_gen(all_objs[trv_split:all_exams], batch_size)

	train_num = int(np.ceil(trv_split / batch_size))
	valid_num = int(np.ceil((all_exams - trv_split) / batch_size))

	#minimizer = SGD(lr=0.0001)
	OPTIMIZER = Adam(lr=0.001)
	LOSS = {'orientation_yaw': orientation_loss, 'confidence_yaw': 'mean_squared_error',
	        'orientation_pitch': orientation_loss, 'confidence_pitch': 'mean_squared_error',
	        'orientation_roll': orientation_loss, 'confidence_roll': 'mean_squared_error'}
	"""
	LOSS = {'dimension': 'mean_squared_error', 'orientation_yaw': orientation_loss, 'confidence_yaw': 'mean_squared_error',
	        'orientation_pitch': orientation_loss, 'confidence_pitch': 'mean_squared_error',
	        'orientation_roll': orientation_loss, 'confidence_roll': 'mean_squared_error'}
	"""
	LOSS_WEIGHTS = {'orientation_yaw': 1., 'confidence_yaw': 1.,
	                'orientation_pitch': 0.9, 'confidence_pitch': 0.9,
	                'orientation_roll': 0.9, 'confidence_roll': 0.9}
	"""
	LOSS_WEIGHTS = {'dimension': 1., 'orientation_yaw': 1., 'confidence_yaw': 1.,
	                'orientation_pitch': 0.9, 'confidence_pitch': 0.9,
	                'orientation_roll': 0.9, 'confidence_roll': 0.9}
	"""
	METRICS = 'accuracy'
	model = VGGnet_regr.build_model(width=NORM_W, height=NORM_H, depth=CHAN, BIN_YAW=BIN, BIN_PITCH=BIN, BIN_ROLL=BIN,
	                                normalization=l2_normalize)
	model.compile(loss=LOSS, loss_weights=LOSS_WEIGHTS, optimizer=OPTIMIZER, metrics=[METRICS])
	plot_model(model, to_file=os.path.join(visual_architect_dir, 'VGGNet_3axes_regr_model.png'), show_shapes=True)
	"""
	model.compile(optimizer='adam',  # minimizer,
	              loss={'dimension': 'mean_squared_error', 'orientation': orientation_loss,
	                    'confidence': 'mean_squared_error'},
	              loss_weights={'dimension': 1., 'orientation': 1., 'confidence': 1.})
	"""
	model.fit_generator(generator=train_gen,
	                    steps_per_epoch=train_num,
	                    epochs=500,
	                    verbose=1,
	                    validation_data=valid_gen,
	                    validation_steps=valid_num,
	                    callbacks=[early_stop, checkpoint, tensorboard],
	                    max_q_size=3)
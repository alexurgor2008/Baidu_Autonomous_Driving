# импорт библиотек Keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import regularizers

########################################################################################################################
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
########################################################################################################################

class VGGnet_regr:
	@staticmethod
	def build_model(width, height, depth, BIN_YAW, BIN_PITCH, BIN_ROLL, normalization):
		# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=‌​2, inter_op_parallelism_threads=2)))
		conf = K.tf.ConfigProto(device_count={'GPU': 1}, intra_op_parallelism_threads=2048,
		                        inter_op_parallelism_threads=2048)
		K.set_session(K.tf.Session(config=conf))

		model = Sequential()

		inputShape = (height, width, depth)

		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		# print(inputShape)
		# Construct the network
		#inputs = Input(shape=(224, 224, 3))
		inputs = Input(shape=inputShape)
		# Block 1
		x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block1_conv1')(inputs)
		x = BatchNormalization(axis=chanDim)(x) ####!!!!!!!!!!!!!!
		x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		# Block 2
		x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		# Block 3
		x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block3_conv3')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		# Block 4
		x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block4_conv1')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block4_conv2')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block4_conv3')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		# Block 5
		x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block5_conv1')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block5_conv2')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal", name='block5_conv3')(x)
		x = BatchNormalization(axis=chanDim)(x)  ####!!!!!!!!!!!!!!
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

		x = Flatten()(x)

		"""
		dimension = Dense(512)(x)
		dimension = LeakyReLU(alpha=0.1)(dimension)
		dimension = Dropout(0.5)(dimension)
		dimension = Dense(3)(dimension)
		dimension = LeakyReLU(alpha=0.1, name='dimension')(dimension)
		"""

		#ORIENTATION YAW
		orientation_yaw = Dense(256)(x)
		orientation_yaw = LeakyReLU(alpha=0.1)(orientation_yaw)
		orientation_yaw = Dropout(0.5)(orientation_yaw)
		orientation_yaw = Dense(BIN_YAW * 2)(orientation_yaw)
		orientation_yaw = LeakyReLU(alpha=0.1)(orientation_yaw)
		orientation_yaw = Reshape((BIN_YAW, -1))(orientation_yaw)
		orientation_yaw = Lambda(normalization, name='orientation_yaw')(orientation_yaw)
		#orientation = Lambda(l2_normalize, name='orientation')(orientation)

		#ORIENTATION PITCH
		orientation_pitch = Dense(128)(x)
		orientation_pitch = LeakyReLU(alpha=0.1)(orientation_pitch)
		orientation_pitch = Dropout(0.5)(orientation_pitch)
		orientation_pitch = Dense(BIN_PITCH * 2)(orientation_pitch)
		orientation_pitch = LeakyReLU(alpha=0.1)(orientation_pitch)
		orientation_pitch = Reshape((BIN_PITCH, -1))(orientation_pitch)
		orientation_pitch = Lambda(normalization, name='orientation_pitch')(orientation_pitch)

		#ORIENTATION ROLL
		orientation_roll = Dense(128)(x)
		orientation_roll = LeakyReLU(alpha=0.1)(orientation_roll)
		orientation_roll = Dropout(0.5)(orientation_roll)
		orientation_roll = Dense(BIN_ROLL * 2)(orientation_roll)
		orientation_roll = LeakyReLU(alpha=0.1)(orientation_roll)
		orientation_roll = Reshape((BIN_ROLL, -1))(orientation_roll)
		orientation_roll = Lambda(normalization, name='orientation_roll')(orientation_roll)

		#CONFIDENCE YAW
		confidence_yaw = Dense(256)(x)
		confidence_yaw = LeakyReLU(alpha=0.1)(confidence_yaw)
		confidence_yaw = Dropout(0.5)(confidence_yaw)
		confidence_yaw = Dense(BIN_YAW, activation='softmax', name='confidence_yaw')(confidence_yaw)

		#CONFIDENCE PITCH
		confidence_pitch = Dense(128)(x)
		confidence_pitch = LeakyReLU(alpha=0.1)(confidence_pitch)
		confidence_pitch = Dropout(0.5)(confidence_pitch)
		confidence_pitch = Dense(BIN_PITCH, activation='softmax', name='confidence_pitch')(confidence_pitch)

		#CONFIDENCE ROLL
		confidence_roll = Dense(128)(x)
		confidence_roll = LeakyReLU(alpha=0.1)(confidence_roll)
		confidence_roll = Dropout(0.5)(confidence_roll)
		confidence_roll = Dense(BIN_ROLL, activation='softmax', name='confidence_roll')(confidence_roll)

		#model = Model(inputs, outputs=[dimension, orientation_yaw, confidence_yaw, orientation_pitch, confidence_pitch, orientation_roll, confidence_roll])
		model = Model(inputs, outputs=[orientation_yaw, confidence_yaw, orientation_pitch, confidence_pitch,
		                               orientation_roll, confidence_roll])
		
		return model

		# model.load_weights('initial_weights.h5')
		"""
		model.add(Conv2D(24, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=inputShape))
		# model.add(Activation("relu"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))  ###???

		model.add(Conv2D(48, (3, 3), padding="same", kernel_initializer="he_normal"))
		# model.add(Activation("relu"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))  ###???
		"""

# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.
from pprint import pprint as pp
import tensorflow as tf
import numpy as np
import scipy.io
import sys

# lista de camadas do modelo VGG19
VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4'
)

# função que carrega o modelo treinado em formato .mat
def load_net(data_path):

	data = scipy.io.loadmat(data_path) # lê o modelo VGG19 em formato .mat

	# verifica se os métodos estão no modelo carregado 
	if not all(i in data for i in ('layers', 'classes', 'normalization')):
		raise ValueError("O modelo baixado não é o correto!")

	mean = data['normalization'][0][0][0] # tensor de médias no formato (224, 224, 3)
	mean_pixel = np.mean(mean, axis=(0, 1)) # médias dos pixeis (uma por cada canal de cor)
	
	# carrega os parâmetros do modelo treinado
	weights = data['layers'][0]
	return weights, mean_pixel


def net_preloaded(weights, input_image):
	net = {} # cria dicionário vazio para colocar o modelo
	current = input_image
	
	# print(weights[0][0][0][0][0])
	# itera pelos nomeas das camadas
	for i, name in enumerate(VGG19_LAYERS):
		
		kind = name[:4] # pega o tipo de camada
		if kind == 'conv': # se a camada for convolucional
			
			# matconvnet: weights são no formato [width, height, in_channels, out_channels]
			# tensorflow: weights são no formato [height, width, in_channels, out_channels]
			# kernels será a matriz W de parâmetros da camada convolucional
			kernels, bias = weights[i][0][0][0][0]

			# transopõe o tensor de parâmetros trocando eixo 1 e 2
			# assim, fica consistente com formato tf
			kernels = np.transpose(kernels, (1, 0, 2, 3))
			bias = bias.reshape(-1) # reformata os bs para tensor unidimensional
			
			# adiciona um nó de camada convolucional no grafo tensorflow
			# atualiza current para para a próxima camada, já que o output
			# de uma camada será o input da próxima
			current = _conv_layer(current, kernels, bias)
		
		elif kind == 'relu':
			# adiciona camada relu
			# atualiza current para o output da camada adicionada
			current = tf.nn.relu(current)
		
		elif kind == 'pool':
			# adiciona camada de pooling
			# atualiza current para o output da camada adicionada
			current = _pool_layer(current)
		
		# adiciona camada ao dicionário da rede neural
		net[name] = current

	assert len(net) == len(VGG19_LAYERS)
	
	# retorna um dicionário {nome_da_camada: camada (tensor)}
	return net 

def _conv_layer(input, weights, bias):
	'''função para adicionar uma camada convolucional ao grafo tf.
	Argumentos são os inputs da camada, Ws e bs'''
	conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
			padding='SAME')
	return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
	'''Função para adicionar camada de pooling'''
	return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
				padding='SAME')

def preprocess(image, mean_pixel):
	return image - mean_pixel


def unprocess(image, mean_pixel):
	return image + mean_pixel
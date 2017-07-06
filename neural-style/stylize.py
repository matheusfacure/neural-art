# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.
from sys import exit
import vgg19 as vgg

import tensorflow as tf
import numpy as np

from sys import stderr

from PIL import Image

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
	reduce
except NameError:
	from functools import reduce


def stylize(network, # path do modelo VGG19
	 content, # img de conteúdo
	 styles, # imgs de estilo
	 iterations, # numero de iterações
	 content_weight, # ponderação do conteúdo
	 content_weight_blend,
	 style_weight, # ponderação do estilo
	 style_layer_weight_exp, # multiplicação da ponderação da camada
	 style_blend_weights,
	 tv_weight,
	 learning_rate, # tx de aprendizado
	 beta1, beta2, epsilon):# parâmetros do otimizador Adam

	
	# formato da imagem de conteúdo
	# add eixo de observaçoes no começo do tensor
	shape = (1,) + content.shape

	# formato da imagem de estilo
	# add eixo de observaçoes no começo do tensor
	style_shapes = [(1,) + style.shape for style in styles]
	
	# inicia dicionários vazios
	content_features = {} # dicionário de valor de ativação das camadas de conteúdo
	style_features = [{} for _ in styles] # dicionários de valor de ativação das camadas de estilo

	# carrega os parâmetros do modelo treinado (W) em vgg_weights
	# carrega estatisticas para batch normalizations
	vgg_weights, vgg_mean_pixel = vgg.load_net(network)

	layer_weight = 1.0 # ponderação da camada padrão
	
	# pondera as camadas de estilo de forma exponencial
	# camadas mais para cima receberão ponderação maior
	style_layers_weights = {} # dicionário de ponderação das camadas de estilo
	for style_layer in STYLE_LAYERS: # itera pelas camadas de conteúdo
		style_layers_weights[style_layer] = layer_weight 
		layer_weight *= style_layer_weight_exp # atualiza a ponderação da camada

	# normaliza as ponderações para que elas sejam relativas
	layer_weights_sum = 0
	for style_layer in STYLE_LAYERS: # calcula o peso total
		layer_weights_sum += style_layers_weights[style_layer] 
	
	for style_layer in STYLE_LAYERS: # divide pela soma dos pesos para ter ponderação relativa
		style_layers_weights[style_layer] /= layer_weights_sum


	g = tf.Graph() # cria um grafo tf para pegar ativação das camadas de conteúdo
	# abre o grafo para adicionar nós
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		
		# cria um placeholder de input para imagem de conteúdo
		image = tf.placeholder('float', shape=shape)
		
		# adiciona o modelo VGG16 ao grafo
		# Guarda as referencia aos tensores do modelo no dicionário net
		net = vgg.net_preloaded(vgg_weights, image)
		
		# subtrai a média dos píxeis das imagens (preprocessamento)
		content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
		
		for layer in CONTENT_LAYERS:
			# para cada camada de conteúdo, parramos a imagem pela rede neural
			# e coletamos a ativação na camada. Nós adicionamos o conteúdo ao
			# dicionário content_features
			content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

	# itera por cada camada de estilo
	for i in range(len(styles)):

		# cria um grafo tf para pegar as atividades nas camadas de estilo
		g = tf.Graph()
		with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
			
			# placeholder para a imagem de estilo
			image = tf.placeholder('float', shape=style_shapes[i])
			
			# Guarda as referencia aos tensores do modelo no dicionário net
			net = vgg.net_preloaded(vgg_weights, image)

			# processa imagem de estilo subtraindo a média
			style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
			
			# itera pelas camadas de estilo
			for layer in STYLE_LAYERS:
				# computa as ativações na camada de estilo
				features = net[layer].eval(feed_dict={image: style_pre})
				
				# achata as ativações em uma matriz em que cada
				# coluna é um filtro de ativação
				features = np.reshape(features, (-1, features.shape[3]))
				
				# calcula a matriz de covariância entre os filtros de ativação da camada
				gram = np.matmul(features.T, features) / features.size
				
				# adiciona a matriz de covariância ao dicionário de 
				# ativações das camadas de estilo
				style_features[i][layer] = gram

	# make stylized image using backpropogation
	with tf.Graph().as_default():
		
		# cria uma imagem inicial a partir de valores aleatórios
		initial = tf.random_normal(shape) * 0.256
		
		# adiciona um nó da imagem inicial
		# esse nó é uma variável, então o tensorflow pode atualizá-la
		# durante a otimização
		image = tf.Variable(initial)
		
		# adicona os nós do modelo, tenso a imagem inicial como input
		net = vgg.net_preloaded(vgg_weights, image)

		# adiciona as ponderações de cada camada de conteúdo à um dicionários
		content_layers_weights = {}
		content_layers_weights['relu4_2'] = content_weight_blend
		content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

		# calcula o custo de conteúdo
		content_losses = [] # inicial lista vazia com custo de estilo
		# para cada camada de conteúdo
		for content_layer in CONTENT_LAYERS:
			
			# adiciona o custo de conteúdo à lista de custos de conteúdo
			content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
					net[content_layer] - content_features[content_layer]) /
					content_features[content_layer].size))
		
		# soma o custo de conteúdo de todas as camadas
		content_loss = tf.reduce_sum(content_losses)

		# custo de estilo
		style_losses = [] # cria lista de custos de estilo
		for i in range(len(styles)): # itera pelas imagens de estilo
			for style_layer in STYLE_LAYERS: # itera pelas camadas de estilo
				layer = net[style_layer] # pega tensor referência da imagem
				_, height, width, number = map(lambda i: i.value, layer.get_shape()) # desempacota formatos da camada
				size = height * width * number # calcula número de filtros de ativação
				feats = tf.reshape(layer, [-1, number]) # achata camada, com filtros de ativação nas colunas
				
				# calcula matriz de covariância; divide pelo número total de filtros
				gram = tf.matmul(tf.transpose(feats), feats) / size # gram com respeito a img misturada
				style_gram = style_features[i][style_layer] # gram (fixa) com respeito as imgs de estilo
				
				# adiciona o custo de estilo da camada nos custos na lista de custos de estilo
				style_losses.append(style_layers_weights[style_layer] * style_blend_weights[i] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
		
		# soma todos os custos de estilo e pondera pelos pesos de estilo
		style_loss = style_weight * tf.reduce_sum(style_losses)

		# custo de denoising (delsoca a imagem e tira a distancia L2)
		tv_y_size = _tensor_size(image[:,1:,:,:])
		tv_x_size = _tensor_size(image[:,:,1:,:])
		tv_loss = tv_weight * 2 * (
				(tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
					tv_y_size) +
				(tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
					tv_x_size))
		
		# custos totais
		loss = content_loss + style_loss + tv_loss

		# otimizador e uma iteração de treino
		train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

		# para mostrar o progresso
		def print_progress():
			stderr.write('  content loss: %g\n' % content_loss.eval())
			stderr.write('    style loss: %g\n' % style_loss.eval())
			stderr.write('       tv loss: %g\n' % tv_loss.eval())
			stderr.write('    total loss: %g\n' % loss.eval())


		# abre sessão para otimização
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer()) # inicializa variáveis
			stderr.write('Optimization started...\n')
	
			# loop de otimização
			for i in range(iterations):
				stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
				
				# roda uma itração de treino
				train_step.run()

			# printa as estatísticas de treino
			print_progress()
			img_out = image.eval()

			# adiciona pixel médio retirado no preprocessamento
			img_out = vgg.unprocess(img_out.reshape(shape[1:]), vgg_mean_pixel)

			# retorna a imagem misturada
			return img_out
			


def _tensor_size(tensor):
	from operator import mul
	return reduce(mul, (d.value for d in tensor.get_shape()), 1)

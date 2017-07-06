# Autor original:  Magnus Erik Hvass Pedersen
# https://github.com/Hvass-Labs/TensorFlow-Tutorials

import numpy as np
import tensorflow as tf
import os
import PIL.Image
import urllib.request
import zipfile
import math
import random
import PIL.Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

# URL para a rede neural treinada
data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
data_dir = './data/' # onde vamos salvar essa rede neural
model_name = os.path.split(data_url)[-1] # inception5h.zip
	
# Arquivo com o grafo tensorflow
path_graph_def = "tensorflow_inception_graph.pb"
local_zip_file = os.path.join(data_dir, model_name) # path completo do local para salvar o arquivo zipado
model_path = os.path.join(data_dir, path_graph_def) # path do modelo

def maybe_download():
	"""
	Faz o download do inception se ele já não tiver sido feito.
	O arquivo contem 50 MB.
	"""
	print("Fazendo o download do inception...")

	# caso ainda não tivermos feito o download
	if not os.path.exists(local_zip_file): 

		# faz o download
		model_url = urllib.request.urlopen(data_url)
		with open(local_zip_file, 'wb') as output:
			output.write(model_url.read())
		
		# Extrai o arquivo zipado
		with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
			zip_ref.extractall(data_dir)


class Inception5h:
	"""
	Wraper para facilitar manipulação do inception.
	"""

	# Nome do tensor de input, por onde vamos alimentar as imagens
	tensor_name_input_image = "input:0"

	def __init__(self):
		
		# Aqui, vamos carregar o inception

		# Primeiro, criamos um grafo tensorflow. Nos vamos importar o inception nele.
		self.graph = tf.Graph()

		# Tornamos o grafo como o nosso default
		with self.graph.as_default():

			
			# Abre o arquivo do modelo para leitura de binarios.
			with tf.gfile.FastGFile(model_path, 'rb') as file:

				graph_def = tf.GraphDef() # criamos uma definição de grafo como um buffer

				graph_def.ParseFromString(file.read()) # carregamos o inception nesse buffer

				tf.import_graph_def(graph_def, name='') # finalmente, carregamos o inception a partir do buffer

				# Agora, self.graph contém todo o grafo do inception

			# pegamos a referência do tensor de input
			self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

			# pegamos a o nome de todas as camadas convolucionais
			self.layer_names = [op.name for op in self.graph.get_operations() if op.type=='Conv2D']

			# pegamos a referência de todas as camadas convolucionais
			self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

			# pegamos referência das dimensões de cada camada
			self.feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in self.layer_names]



	def create_feed_dict(self, image=None):
		"""
		Cria um feed_dict para acessar passar dados para a sessão do tensorflow

		:image:
			A imagem deve ser um array 3D, cujos pixeis variam de 0 até 255.

		:retorna:
			Um dicionário feed_dict para passar a sessão tf
		"""

		# Aumenta a dimensão do array para 4
		# Isso pq o modelo recebe como input um tensor de mini-lotes de imagem:
		# [n_amostras, largura, altura, cor]
		image = np.expand_dims(image, axis=0)

		# cria o dicionário
		feed_dict = {self.tensor_name_input_image: image}

		return feed_dict


	def get_gradient(self, tensor):
		"""
		Retorna o gradiente do tensor (camada) passado com respeito ao
		tensor de input.

		:param tensor:
			O tensor (camada) para pegar o gradiente

		:retorna:
			o gradiente do tensor com respeito ao input
			
		"""

		# torna o grafo como padrão para adicionar ops nele
		with self.graph.as_default():

			# Eleva o tensor de input ao quadrado (aumenta o sinal)
			tensor = tf.square(tensor)

			# tira a média do tensor, para que o gradiente seja um único número, a derivada
			tensor_mean = tf.reduce_mean(tensor)

			# pega o gradiente da média do tensor com respeito ao input.
			gradient = tf.gradients(tensor_mean, self.input)[0]

		return gradient


def load_image(filename):
	'''Carrega uma imagem e converte para um array de float32'''
	image = PIL.Image.open(filename)

	return np.float32(image)


def save_image(image, filename):
	'''Salva uma imagem'''
	# garante que todos os pixeis estejam entre 0 e 255
	image = np.clip(image, 0.0, 255.0)
	
	# converte imagem para bytes
	image = image.astype(np.uint8)
	
	# salva imagem em jpeg
	with open(filename, 'wb') as file:
		PIL.Image.fromarray(image).save(file, 'jpeg')


def normalize_image(x):
	'''normaliza imagem com min-max. Assume que imagem seja array'''
	x_min = x.min()
	x_max = x.max()
	x_norm = (x - x_min) / (x_max - x_min)
	return x_norm


def plot_image(image):
	'''Plota uma imagem (assume que pixeis estarão entre 0 e 255)'''
	
	# Garante que os pixeis estejam entre 0 e 255.
	image = np.clip(image, 0.0, 255.0)
	
	# converte imagem para bytes
	image = image.astype(np.uint8)

	# converte para objeto PIL
	PIL.Image.fromarray(image).show()


def plot_gradient(gradient):
	'''plota gradiente'''
	gradient_normalized = normalize_image(gradient) # normaliza o gradiente
	plt.imshow(gradient_normalized, interpolation='bilinear')
	plt.show()


def resize_image(image, size=None, factor=None):
	'''Reescalona uma imagem.'''

	# Caso use o fator de reescalonamento 
	if factor is not None:
		# calcula tamanho final
		size = np.array(image.shape[0:2]) * factor
		size = size.astype(int) # converte tamanho para inteiros
	
	else:
		# garante que o tamanho seja 2D
		size = size[0:2]
	
	# inverte dim. PIL e matplotlib usam dimensões inversas :/
	size = tuple(reversed(size))

	# garante que pixeis sejam entre 0 e 255
	img = np.clip(image, 0.0, 255.0)
	
	# converte pixeis para bytes
	img = img.astype(np.uint8)
	
	# cria objeto PIL
	img = PIL.Image.fromarray(img)
	
	# reescalona a imagem
	img_resized = img.resize(size, PIL.Image.LANCZOS)
	
	# converte de volta de bytes para array de floats32
	img_resized = np.float32(img_resized)

	return img_resized


# calcular o gradiente de toda a imagem de uma tacada só pode consumir mt RAM.
# vamos então calcular o gradiente de pequenos pedaços separadamente

def get_tile_size(num_pixels, tile_size=400):
	"""
	Acha a dimensão do pedaço.
	
	num_pixels: n de pixeis em uma das dimensões da imagem
	tile_size: tamanho desejado do pedaço
	"""

	# qtd de vezes q podemos repetir um pedaço na imagem
	num_tiles = int(round(num_pixels / tile_size))
	
	# garante que haja pelo menos 1 pedaço
	num_tiles = max(1, num_tiles)
	
	# tamanho real do pedaço
	actual_tile_size = math.ceil(num_pixels / num_tiles)
	
	return actual_tile_size


def tiled_gradient(model, gradient, image, tile_size=400):
	'''calcula o gradiente por pedaços'''

	grad = np.zeros_like(image) # aloca um array vazio do tamanho da imagem

	# N de pixeis por dimensão
	x_max, y_max, _ = image.shape

	# tamanho do pedaço para dim-x
	x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
	x_tile_size4 = x_tile_size // 4 # 1/4 do tamanho do pedaço

	# tamanho do pedaço para dim-y
	y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
	y_tile_size4 = y_tile_size // 4 # 1/4 do tamanho do pedaço

	# lugar para começar a calcular na dim-x
	x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

	while x_start < x_max:
		# posição final do pedaço atual (dim-x)
		x_end = x_start + x_tile_size
		
		# garante que local de inicio e fim do pedaço sejam contidos na imagem
		x_start_lim = max(x_start, 0)
		x_end_lim = min(x_end, x_max)

		# lugar para começar a calcular na dim-y
		y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

		while y_start < y_max:
			# posição final do pedaço atual (dim-x)
			y_end = y_start + y_tile_size

			# garante que local de inicio e fim do pedaço sejam contidos na imagem
			y_start_lim = max(y_start, 0)
			y_end_lim = min(y_end, y_max)

			# finalmente, monta o pedaço nas duas dims
			img_tile = image[x_start_lim:x_end_lim,
							 y_start_lim:y_end_lim, :]

			# cria um o feed_dict
			feed_dict = model.create_feed_dict(image=img_tile)

			# Usa o TensorFlow para calcular o valor numético do gradiente
			g = session.run(gradient, feed_dict=feed_dict)

			# normaliza o gradiente para o pedaço
			g /= (np.std(g) + 1e-8)

			# armazena o gradiente no pedaço no array antes vazio
			grad[x_start_lim:x_end_lim,
				 y_start_lim:y_end_lim, :] = g
			
			# avança para a próxima posição no eixo y
			y_start = y_end

		# avança para a próxima posição no eixo x
		x_start = x_end

	return grad


def optimize_image(model, layer_tensor, image,
				   num_iterations=10, step_size=3.0, tile_size=400,
				   show_gradient=False):
	"""
	Uma gradiente ascendente para maximizar a imagem com respeito a camada
	
	Parameters:
	model: modelo inception
	layer_tensor: referência da camada para a otimização
	image: imagem
	num_iterations: número de iterações
	step_size: escala para cada passo do gradiente descendente
	tile_size: tamanho do pedaço para calcular o gradiente
	show_gradient: se mostra ou não o gradiente
	"""

	# cria uma cópia da imagem
	img = image.copy()
	
	print("Processando imagem: ", end="")

	# calcula o gradiente. Pode consumir mt RAM
	gradient = model.get_gradient(layer_tensor)
	
	for i in range(num_iterations):

		# calcula o gradiente por pedaços
		grad = tiled_gradient(model=model, gradient=gradient, image=img)
		
		# deixa o gradiente mais suave
		sigma = (i * 4.0) / num_iterations + 0.5
		grad_smooth1 = gaussian_filter(grad, sigma=sigma)
		grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
		grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
		grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

		# escalona o gradiente
		step_size_scaled = step_size / (np.std(grad) + 1e-8)

		# atualiza a imagem com o gradiente
		img += grad * step_size_scaled

		# mostra o gradiente, se for o caso
		if show_gradient:
			plot_gradient(grad)
		else:
			# mostra uma barra de progresso
			print(". ", end="")

	return img


def recursive_optimize(model, layer_tensor, image,
					   num_repeats=4, rescale_factor=0.7, blend=0.2,
					   num_iterations=10, step_size=3.0,
					   tile_size=400):
	"""
	Aplica o DeepDream de forma recursiva para ficar bom mesmo em imagens grandes
	
	modelo: modelo inception
	image: imagem de input
	rescale_factor: fator de escalonamento
	num_repeats: qtd de vezes para reescalonar
	blend: fator de mistura com os diversos tamanhos

	parametros de optimize_image():
	layer_tensor: referência da camada para a otimização
	num_iterations: número de iterações
	step_size: escala para cada passo do gradiente descendente
	tile_size: tamanho do pedaço para calcular o gradiente
	"""

	# para fazer um passo iterativo até o limite estabelecido
	if num_repeats>0:

		# suaviza a imagem
		sigma = 0.5
		img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

		# diminui a imagem
		img_downscaled = resize_image(image=img_blur,
									  factor=rescale_factor)
			
		# chamada recursiva da função
		img_result = recursive_optimize(layer_tensor=layer_tensor,
										model=model,
										image=img_downscaled,
										num_repeats=num_repeats-1,
										rescale_factor=rescale_factor,
										blend=blend,
										num_iterations=num_iterations,
										step_size=step_size,
										tile_size=tile_size)
		
		# aumenta imagem ao tamanho original
		img_upscaled = resize_image(image=img_result, size=image.shape)

		# mescla as imagens de tamanhos diferentes
		image = blend * image + (1.0 - blend) * img_upscaled

	print("nível da recursão:", num_repeats)

	# processa a imagem usando o deepdream
	img_result = optimize_image(layer_tensor=layer_tensor,
								model=model,
								image=image,
								num_iterations=num_iterations,
								step_size=step_size,
								tile_size=tile_size)
	
	return img_result


def DeepDream(model, image, img_file='after.jpg', layer=None):

	# uma camada for selecionada
	if layer is not None:
		
		layer_tensor = model.layer_tensors[layer] # seleciona camada

		# aplica o deepdream com recursão
		img_result = recursive_optimize(model, layer_tensor=layer_tensor, image=image,
					 num_iterations=15, step_size=3.0, rescale_factor=0.7,
					 num_repeats=4, blend=0.2)
		
		# salva a imagem
		save_image(img_result, filename=str(layer)+'_DD_'+img_file)

	else:

		# itera por cada camada
		for c in range(56):

			layer_tensor = model.layer_tensors[c]
			print('\nCamada %d' % c, layer_tensor)

			# aplica o deepdream com recursão
			img_result = recursive_optimize(model, layer_tensor=layer_tensor, image=image,
						 num_iterations=15, step_size=3.0, rescale_factor=0.7,
						 num_repeats=4, blend=0.2)

			# salva a imagem
			save_image(img_result, filename=str(c)+'_DD_'+img_file)

	return img_result



def zoom_in(img, zoom_factor=1.1):
	'''Amplia a imagem no centro, mantendo dimensões;'''
	
	img_shape = img.shape # fixa o tamanho original

	img = zoom(img, [zoom_factor, zoom_factor, 1]) # amplia a imagem

	y, x = img.shape[:2] # pega o novo tamanho

	# corta a imagem no centro
	startx = x//2-(img_shape[1]//2)
	starty = y//2-(img_shape[0]//2)    
	return img[starty:starty+img_shape[0],startx:startx+img_shape[1]]


def DeepDreamZoom(model, image, frames, layer=50):

	for frame in range(frames):

		# aplica o DeepDream
		image = DeepDream(model, image, 'frame_'+str(frame)+'.jpg', layer=layer)

		image = zoom_in(image, 1.1) # aplica o zoom de 5%




########################################################################


if __name__ == '__main__':
	
	# Baixa a rede neural
	maybe_download()

	# carrega a rede neural
	model = Inception5h() 

	# inicia uma sessão tf
	session = tf.InteractiveSession(graph=model.graph)

	# carrega a imagm
	img_file = './data/festa_da_lanterna.jpg' 
	image = load_image(filename=img_file)
	
	# aplica o deepdream e salva a imagem gerada
	image = DeepDream(model, image, img_file, layer=1)
	
	# aplica deepdream iterativo com zoom
	# DeepDreamZoom(model, image, 5, 50)
	
	
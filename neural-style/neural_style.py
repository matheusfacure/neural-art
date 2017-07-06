# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os
import sys
import numpy as np
import scipy.misc
import urllib.request 
from stylize import stylize
import math
from argparse import ArgumentParser
from PIL import Image

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def _print_download_progress(count, block_size, total_size):
	"""
	Para mostrar o progersso do download
	"""

	# percentagem completa
	pct_complete = float(count * block_size) / total_size

	# mensagem
	msg = "\r- Progresso de Download: {0:.1%}".format(pct_complete)

	# mostra a mensagem
	sys.stdout.write(msg)
	sys.stdout.flush()


def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--content',
			dest='content', help='content image',
			metavar='CONTENT', required=True)
	parser.add_argument('--styles',
			dest='styles',
			nargs='+', help='one or more style images',
			metavar='STYLE', required=True)
	parser.add_argument('--output',
			dest='output', help='output path',
			metavar='OUTPUT', required=True)
	parser.add_argument('--iterations', type=int,
			dest='iterations', help='iterations (default %(default)s)',
			metavar='ITERATIONS', default=ITERATIONS)
	parser.add_argument('--style-scales', type=float,
			dest='style_scales',
			nargs='+', help='one or more style scales',
			metavar='STYLE_SCALE')
	parser.add_argument('--network',
			dest='network', help='path to network parameters (default %(default)s)',
			metavar='VGG_PATH', default=VGG_PATH)
	parser.add_argument('--content-weight-blend', type=float,
			dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
			metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
	parser.add_argument('--content-weight', type=float,
			dest='content_weight', help='content weight (default %(default)s)',
			metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
	parser.add_argument('--style-weight', type=float,
			dest='style_weight', help='style weight (default %(default)s)',
			metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
	parser.add_argument('--style-layer-weight-exp', type=float,
			dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
			metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
	parser.add_argument('--style-blend-weights', type=float,
			dest='style_blend_weights', help='style blending weights',
			nargs='+', metavar='STYLE_BLEND_WEIGHT')
	parser.add_argument('--tv-weight', type=float,
			dest='tv_weight', help='total variation regularization weight (default %(default)s)',
			metavar='TV_WEIGHT', default=TV_WEIGHT)
	parser.add_argument('--learning-rate', type=float,
			dest='learning_rate', help='learning rate (default %(default)s)',
			metavar='LEARNING_RATE', default=LEARNING_RATE)
	parser.add_argument('--beta1', type=float,
			dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
			metavar='BETA1', default=BETA1)
	parser.add_argument('--beta2', type=float,
			dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
			metavar='BETA2', default=BETA2)
	parser.add_argument('--eps', type=float,
			dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
			metavar='EPSILON', default=EPSILON)
	return parser


def main():
	# lida com os argumentos da linha de comando
	parser = build_parser()
	options = parser.parse_args()

	# verifica se a rede neural utilizada é o modelo correto
	if not os.path.isfile(options.network):
		urllib.request.urlretrieve(url='http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat',
									filename=options.network,
									 reporthook=_print_download_progress)
	
	content_image = imread(options.content) # lê a imagem de conteúdo
	style_images = [imread(style) for style in options.styles] # lê as imagens de estilo

	# imagem de saída terá mesmo formato da de conteúdo
	target_shape = content_image.shape

	# itera pelas imagens de estilo
	for i in range(len(style_images)):

		# reescalona as imagens de estilo
		style_scale = STYLE_SCALE
		if options.style_scales is not None:
			style_scale = options.style_scales[i]
		
		style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
				target_shape[1] / style_images[i].shape[1])

	# define a ponderação para cada estilo passado
	style_blend_weights = options.style_blend_weights
	if style_blend_weights is None: # padrão sendo peso igual para cada estilo
		style_blend_weights = [1.0/len(style_images) for _ in style_images]
	
	else:
		total_blend_weight = sum(style_blend_weights)
		style_blend_weights = [weight/total_blend_weight
							   for weight in style_blend_weights]
	combined_rgb = stylize(
		network=options.network, # path do modelo VGG19
		content=content_image, # imagem de conteúdo
		styles=style_images, # imagens de estilo
		iterations=options.iterations,
		content_weight=options.content_weight,
		content_weight_blend=options.content_weight_blend,
		style_weight=options.style_weight,
		style_layer_weight_exp=options.style_layer_weight_exp,
		style_blend_weights=style_blend_weights,
		tv_weight=options.tv_weight,
		learning_rate=options.learning_rate,
		beta1=options.beta1,
		beta2=options.beta2,
		epsilon=options.epsilon)


	output_file = options.output
	imsave(output_file, combined_rgb)


def imread(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2:
		# grayscale
		img = np.dstack((img,img,img))
	elif img.shape[2] == 4:
		# PNG with alpha channel
		img = img[:,:,:3]
	return img


def imsave(path, img):
	img = np.clip(img, 0, 255).astype(np.uint8)
	Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
	main()

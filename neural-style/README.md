usage: neural_style.py [-h] --content CONTENT --styles STYLE [STYLE ...]
                       --output OUTPUT [--iterations ITERATIONS]
                       [--style-scales STYLE_SCALE [STYLE_SCALE ...]]
                       [--network VGG_PATH]
                       [--content-weight-blend CONTENT_WEIGHT_BLEND]
                       [--content-weight CONTENT_WEIGHT]
                       [--style-weight STYLE_WEIGHT]
                       [--style-layer-weight-exp STYLE_LAYER_WEIGHT_EXP]
                       [--style-blend-weights STYLE_BLEND_WEIGHT [STYLE_BLEND_WEIGHT ...]]
                       [--tv-weight TV_WEIGHT] [--learning-rate LEARNING_RATE]
                       [--beta1 BETA1] [--beta2 BETA2] [--eps EPSILON]

Exemplo: 

python3 neural_style.py --content barolo.jpg --styles davinci.jpg --output out.jpg


Esse código foi copiado de anishathalye. Aqui está a versão original: https://github.com/anishathalye/neural-style

As únicas modificações são que a versão aqui presente é simplificada e documentada em português. 
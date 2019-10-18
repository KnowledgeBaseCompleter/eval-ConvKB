CUDA_VISIBLE_DEVICES=0 python -u train.py --embedding_dim 100 --num_filters 50 --learning_rate 0.000005 --name FB15k-237 --useConstantInit --model_name fb15k237
CUDA_VISIBLE_DEVICES=0 python -u eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237 --num_splits 1 --testIdx 0


CUDA_VISIBLE_DEVICES=0 python -u train.py --embedding_dim 100 --num_filters 50 --learning_rate 0.000005 --name FB15k-237 --useConstantInit --model_name fb15k237_ln --add_layer_norm
CUDA_VISIBLE_DEVICES=0 python -u eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237_ln --num_splits 1 --testIdx 0 --add_layer_norm

CUDA_VISIBLE_DEVICES=5 python -u eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237_ln --num_splits 1 --testIdx 0 --add_layer_norm --eval_type org

CUDA_VISIBLE_DEVICES=6 python -u eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237_ln --num_splits 1 --testIdx 0 --add_layer_norm --eval_type random

CUDA_VISIBLE_DEVICES=7 python -u eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237_ln --num_splits 1 --testIdx 0 --add_layer_norm --eval_type last
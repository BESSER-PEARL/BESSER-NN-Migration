"""
It defines mapping dictionaries for transforming PyTorch code
to BUML code.
"""

layers_mapping = {
    "Conv1d": "Conv1D", "Conv2d": "Conv2D", "Conv3d": "Conv3D",           
    "MaxPool1d": "PoolingLayer", "MaxPool2d": "PoolingLayer", 
    "MaxPool3d": "PoolingLayer", "AvgPool1d": "PoolingLayer", 
    "AvgPool2d": "PoolingLayer", "AvgPool3d": "PoolingLayer",
    "AdaptiveAvgPool1d": "PoolingLayer", "AdaptiveAvgPool2d": "PoolingLayer", 
    "AdaptiveAvgPool3d": "PoolingLayer", "AdaptiveMaxPool1d": "PoolingLayer", 
    "AdaptiveMaxPool2d": "PoolingLayer", "AdaptiveMaxPool3d": "PoolingLayer", 
    "Flatten": "FlattenLayer", "Linear": "LinearLayer", 
    "Embedding": "EmbeddingLayer", "BatchNorm1d": "BatchNormLayer",
    "BatchNorm2d": "BatchNormLayer", "BatchNorm3d": "BatchNormLayer",
    "LayerNorm": "LayerNormLayer", "Dropout": "DropoutLayer", "Dropout1d": "DropoutLayer",
    "Dropout2d": "DropoutLayer", "Dropout3d": "DropoutLayer", 
    "RNN": "SimpleRNNLayer", "LSTM": "LSTMLayer", "GRU": "GRULayer"
}

actv_fun_mapping = {
    "ReLU": "relu", "LeakyReLU": "leaky_relu", "Sigmoid": "sigmoid",
    "Softmax": "softmax", "Tanh": "tanh", "GELU": "gelu"
}

functional_to_module_mapping = {
    "relu": "ReLU", "leaky_relu": "LeakyReLU", "sigmoid": "Sigmoid",
    "softmax": "Softmax", "tanh": "Tanh", "gelu": "GELU",
    "adaptive_max_pool1d": "AdaptiveMaxPool1d",
    "adaptive_avg_pool1d": "AdaptiveAvgPool1d",
    "adaptive_max_pool2d": "AdaptiveMaxPool2d",
    "adaptive_avg_pool2d": "AdaptiveAvgPool2d",
    "adaptive_max_pool3d": "AdaptiveMaxPool3d",
    "adaptive_avg_pool3d": "AdaptiveAvgPool3d",
    "max_pool1d": "MaxPool1d", "max_pool2d": "MaxPool2d", "max_pool3d": "MaxPool3d",
    "avg_pool1d": "AvgPool1d", "avg_pool2d": "AvgPool2d", "avg_pool3d": "AvgPool3d",
    "flatten": "Flatten"
}

params_mapping = {
    "in_channels": "in_channels", "out_channels": "out_channels",
    "kernel_size": "kernel_dim", "stride": "stride_dim",
    "padding": "padding_amount", "output_size": "output_dim",
    "dilation": "dilation", "groups": "groups", "bias": "bias",
    "input_size": "input_size", "hidden_size": "hidden_size",
    "bidirectional": "bidirectional", "dropout": "dropout",
    "batch_first": "batch_first", "normalized_shape": "normalized_shape",
    "p": "rate", "num_features": "num_features",
    "eps": "eps", "momentum": "momentum", "affine": "affine",
    "elementwise_affine": "affine",  # LayerNorm uses elementwise_affine
    "track_running_stats": "track_running_stats",
    "num_embeddings": "num_embeddings", "embedding_dim": "embedding_dim",
    "padding_idx": "padding_idx",
    "start_dim": "start_dim", "end_dim": "end_dim",
    "in_features": "in_features", "out_features": "out_features",
    "return_type": "return_type", "permute_in": "permute_in",
    "permute_out": "permute_out", "nonlinearity": "actv_func"
}

static_params = {
    "MaxPool1d": {"pooling_type": "max", "dimension": "1D"},
    "MaxPool2d": {"pooling_type": "max", "dimension": "2D"},
    "MaxPool3d": {"pooling_type": "max", "dimension": "3D"},
    "AvgPool1d": {"pooling_type": "avg", "dimension": "1D"},
    "AvgPool2d": {"pooling_type": "avg", "dimension": "2D"},
    "AvgPool3d": {"pooling_type": "avg", "dimension": "3D"},
    "AdaptiveAvgPool1d": {"pooling_type": "adaptive_average", 
                          "dimension": "1D"},
    "AdaptiveAvgPool2d": {"pooling_type": "adaptive_average", 
                          "dimension": "2D"},
    "AdaptiveAvgPool3d": {"pooling_type": "adaptive_average", 
                          "dimension": "3D"},
    "AdaptiveMaxPool1d": {"pooling_type": "adaptive_max", "dimension": "1D"},
    "AdaptiveMaxPool2d": {"pooling_type": "adaptive_max", "dimension": "2D"},
    "AdaptiveMaxPool3d": {"pooling_type": "adaptive_max", "dimension": "3D"},
    "BatchNorm1d": {"dimension": "1D"},
    "BatchNorm2d": {"dimension": "2D"},
    "BatchNorm3d": {"dimension": "3D"},
}


channels_first_layers = ["Conv1D", "Conv2D", "Conv3D", "PoolingLayer", "BatchNormLayer"]


loss_func_mapping = {"CrossEntropyLoss": "crossentropy",
                     "BCELoss": "binary_crossentropy",
                     "MSELoss": "mse"}

pos_params = {"Linear": ["in_features", "out_features"],
              "Embedding": ["num_embeddings", "embedding_dim"],
              "Flatten": ["start_dim", "end_dim"],
              "RNN": ["input_size", "hidden_size"],
              "LSTM": ["input_size", "hidden_size"],
              "GRU": ["input_size", "hidden_size"],
              "Conv": ["in_channels", "out_channels", "kernel_size",
                       "stride", "padding"],
              "AvgPool": ["kernel_size", "stride", "padding"],
              "MaxPool": ["kernel_size", "stride", "padding"],
              "AdaptiveAvgPool": ["output_size"],
              "AdaptiveMaxPool": ["output_size"],
              "Dropout": ["p"], "Dropout1d": ["p"], "Dropout2d": ["p"], "Dropout3d": ["p"],
              "LayerNorm": ["normalized_shape"],
              "BatchNorm": ["num_features"]}


int2list_params = ["kernel_size", "stride", "output_size", "normalized_shape", "dilation"]
lyrs_of_int2list_params = ["Conv1d", "Conv2d", "Conv3d", "MaxPool1d",
                           "AvgPool1d", "MaxPool2d", "AvgPool2d", 
                           "MaxPool3d", "AvgPool3d", "LayerNorm",
                           "AdaptiveAvgPool1d", "AdaptiveMaxPool1d",
                           "AdaptiveAvgPool2d", "AdaptiveMaxPool2d",
                           "AdaptiveAvgPool3d", "AdaptiveMaxPool3d"]


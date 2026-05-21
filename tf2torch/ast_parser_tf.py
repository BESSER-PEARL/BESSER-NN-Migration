"""
Module to extract information from the AST of a neural network
written in TensorFlow and transforms it to a BUML model.
It also extracts data and model configuration attributes.
"""


import ast
import sys
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
import besser.BUML.metamodel.nn as mm_classes
from besser.BUML.metamodel.nn import NN, Layer
from ast_parser_nn import ASTParser
from definitions import (
    layers_mapping, params_mapping, static_params, rnn_layers,
    pos_params, int2list_params, lyrs_of_int2list_params, loss_func_mapping
)
from transform_code import (
    process_positional_params, param_to_list, set_static_params
)

# TensorFlow activation function mapping (tf.nn.* to BUML)
tf_actv_func_mapping = {
    "relu": "relu",
    "tanh": "tanh",
    "sigmoid": "sigmoid",
    "softmax": "softmax",
    "leaky_relu": "leaky_relu",
    "elu": "elu",
    "selu": "selu",
    "gelu": "gelu",
}

def infer_batchnorm_params(buml_model):
    """
    Infer BatchNormalization dimension and num_features from previous layers.

    Look backwards through layers to find the most recent CNN layer.
    If found, use its dimension. Otherwise default to 1D.

    Parameters:
        buml_model: The BUML model with already added layers.

    Returns:
        dict with 'dimension' and 'num_features' params.
    """
    # Search backwards through layers for the most recent conv layer
    for layer in reversed(buml_model.layers):
        layer_class = layer.__class__.__name__

        # Check if it's a conv layer
        if layer_class in ["Conv1D", "Conv2D", "Conv3D"]:
            # Extract dimension from class name (Conv1D -> 1D)
            dim = layer_class[-2:]  # Gets "1D", "2D", or "3D"
            num_features = layer.out_channels
            return {"dimension": dim, "num_features": num_features}

        # If we hit a Flatten or Dense, the data is now 1D
        if layer_class in ["FlattenLayer", "LinearLayer"]:
            # After flatten/dense, we're in 1D space
            if layer_class == "LinearLayer":
                num_features = layer.out_features
            else:
                num_features = 1  # Will need to infer dynamically
            return {"dimension": "1D", "num_features": num_features}

    # Default if no conv/dense found
    return {"dimension": "1D", "num_features": 1}


class ASTParserTF(ASTParser):
    """
    Class visiting and parsing TensorFlow code AST
    
    Attributes:
        input_nn_type (str): The type of the nn input architecture.
        only_nn (str): Whether to process only the model definition or also
            its configuration and dataset.
        padding_amount (int | None): It  keeps track of padding in 
            ZeroPadding layer. In TF, padding is added to conv layers 
            using a separate layer, but in PyTorch it is defined as an 
            attribute of the conv layer.
    """

    def __init__(self, input_nn_type: str, only_nn:bool):
        super().__init__(input_nn_type, only_nn)

        self.padding_amount: int | None = None

    def handle_init(self, node: ast.Assign):
        """
        It retrieves the sub_nn layers and stores them in the 'sub_nn' 
        dict. It also retreives the layers and their parameters and 
        stores them in the 'layers' dict.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name)):
            module_type = node.value.func.id
            if module_type == "Sequential":
                self.handle_sequential_layers(node, module_name)

        #simple calls to layers
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            lyr_type, lyr_params = self.extract_layer(node.value)

            if len(node.value.args)>0: #rnn bidirectional
                if (isinstance(node.value.args[0], ast.Call) and
                    node.value.func.attr == "Bidirectional"):
                    lyr = node.value.args[0]
                    lyr_type, lyr_params = self.extract_layer(lyr)
                    lyr_params["bidirectional"] = True

            lyr_type, lyr_params, padding_amount = transform_layer(
                lyr_type, lyr_params, self.padding_amount, module_name
            )

            self.padding_amount = padding_amount
            if not lyr_type.startswith("ZeroPadding"):
                # Infer BatchNorm params from previous layers if needed
                if lyr_type == "BatchNormLayer":
                    inferred_params = infer_batchnorm_params(self.buml_model)
                    lyr_params.update(inferred_params)

                buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
                self.buml_model.add_layer(buml_layer)

        #to get the proper order from forward the method
        self.buml_model.modules.clear()


    def handle_sequential_layers(self, node: ast.Assign, seq_name: str):
        """
        It retrieves layers of a sequential model.
        It can be used for the main sequential nn and sub-nns.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.
            seq_name (str): The name of the sequential model.

        Returns:
            None, but populates the BUML model.

        """
        subnn: NN = NN(name=seq_name)
        layer_id = 1
        # Extract layers within Sequential
        for elt in node.value.args[0].elts:
            if isinstance(elt, ast.Call):
                lyr_type, lyr_params = self.extract_layer(elt)
                if len(elt.args)>0: #rnn bidirectional
                    if (isinstance(elt.args[0], ast.Call) and
                        isinstance(elt.func, ast.Attribute)):
                        if  elt.func.attr == "Bidirectional":
                            lyr = elt.args[0]
                            lyr_type, lyr_params = self.extract_layer(lyr)
                            lyr_params["bidirectional"] = True

                lyr_type, lyr_params, padding_amount = transform_layer(
                    lyr_type, lyr_params, self.padding_amount
                )
                self.padding_amount = padding_amount
                if not lyr_type.startswith("ZeroPadding"):
                    lyr_params["name"] = f"layer_{layer_id}"

                    # Infer BatchNorm params from previous layers if needed
                    if lyr_type == "BatchNormLayer":
                        inferred_params = infer_batchnorm_params(subnn)
                        lyr_params.update(inferred_params)

                    subnn_layer = getattr(mm_classes, lyr_type)(**lyr_params)
                    subnn.add_layer(subnn_layer)
                    layer_id+=1

            elif isinstance(elt, ast.Name):
                subnn_obj = next((obj for obj in self.buml_model.sub_nns if
                                  obj.name == elt.id), None)
                subnn.add_sub_nn(subnn_obj)

        self.buml_model.add_sub_nn(subnn)


    def decompose_chained_call(self, node):
        """Decompose chained calls into list of individual operations in execution order."""
        chain = []
        current = node.value
        while isinstance(current, ast.Call) and hasattr(current.func, 'value'):
            chain.append(current)
            if isinstance(current.func.value, ast.Call):
                current = current.func.value
            else:
                break
        chain.reverse()
        return chain

    def handle_forward_simple_call(self, node: ast.Assign):
        """
        Processes forward method assignments including chained calls.
        """
        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'value'):
            if isinstance(node.value.func.value, ast.Call):
                chain = self.decompose_chained_call(node)
                for i, call in enumerate(chain):
                    is_last = (i == len(chain) - 1)
                    if is_last:
                        synthetic_node = ast.Assign(targets=node.targets, value=call)
                    else:
                        temp_name = f"_chain_temp_{self.tensor_op_counter}_{i}"
                        temp_target = ast.Name(id=temp_name, ctx=ast.Store())
                        synthetic_node = ast.Assign(targets=[temp_target], value=call)

                        # Update next call based on whether it's functional or method call
                        if i + 1 < len(chain):
                            next_call = chain[i + 1]
                            next_func_value = next_call.func.value

                            if isinstance(next_func_value, ast.Name):
                                # Functional call like layers.Activation(x) where x is in args[0]
                                if next_call.args:
                                    next_call.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
                            elif isinstance(next_func_value, ast.Call):
                                # Method call like result.transpose(...) where result is func.value
                                next_call.func.value = ast.Name(id=temp_name, ctx=ast.Load())
                    synthetic_node.lineno = node.lineno
                    synthetic_node.col_offset = node.col_offset
                    self.process_single_call(synthetic_node)
                    self.previous_assign = synthetic_node
                return

        # Handle nested calls
        is_nested_call = False
        if isinstance(node.value, ast.Call) and node.value.args:
            if isinstance(node.value.args[0], ast.Call):
                # Nested call detected: process inner call first
                inner_call = node.value.args[0]
                temp_name = f"_nested_temp_{self.tensor_op_counter}"
                self.tensor_op_counter += 1
                temp_target = ast.Name(id=temp_name, ctx=ast.Store())

                # Create synthetic node for inner call
                inner_node = ast.Assign(targets=[temp_target], value=inner_call)
                inner_node.lineno = node.lineno
                inner_node.col_offset = node.col_offset
                self.process_single_call(inner_node)
                self.previous_assign = inner_node

                # Replace inner call with temp variable in outer call
                node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
                is_nested_call = True

        # Mark if this is a nested call's outer part
        if is_nested_call:
            self.is_processing_nested_outer = True

        self.process_single_call(node)
        self.previous_assign = node

        if is_nested_call:
            self.is_processing_nested_outer = False

    def process_single_call(self, node: ast.Assign):
        """
        Process a single (non-chained) call operation.
        This contains the original logic from handle_forward_simple_call.
        """
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "self":
                module_name = node.value.func.attr
                #populate inputs_outputs and module_of_output
                self.inputs_outputs[module_name] = [node.value.args[0].id,
                                                    node.targets[0].id]
                self.module_of_output[node.targets[0].id] = module_name
                module_obj = next((obj for obj in self.buml_model.layers if
                                   obj.name == module_name), None)
                if not module_obj:
                    subnns = self.buml_model.sub_nns
                    module_obj = next((obj for obj in subnns if
                                       obj.name == module_name), None)

                self.buml_model.modules.append(module_obj)
            else:
                #tensorops or tf.nn.* activations
                # Check if it's tf.nn.<activation>
                if (isinstance(node.value.func.value, ast.Attribute) and
                    node.value.func.value.attr == "nn" and
                    isinstance(node.value.func.value.value, ast.Name) and
                    node.value.func.value.value.id == "tf"):
                    # This is tf.nn.* - check if activation
                    func_name = node.value.func.attr
                    if func_name in tf_actv_func_mapping:
                        self.handle_tf_activation(node, func_name)
                    else:
                        # Not an activation, treat as tensorop
                        self.extract_tensorop(node)
                else:
                    self.extract_tensorop(node)
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.value.id == "tf":
                # Check if it's tf.nn.<activation>
                if (node.value.func.value.attr == "nn" and
                    node.value.func.attr in tf_actv_func_mapping):
                    self.handle_tf_activation(node, node.value.func.attr)
                else:
                    self.extract_tensorop(node)


    def handle_tf_activation(self, node: ast.Assign, func_name: str):
        """
        Handle tf.nn.* activation functions (like tf.nn.relu).

        Parameters:
            node (ast.Assign): The AST node representing the assignment.
            func_name (str): The activation function name (e.g., 'relu').

        Returns:
            None, but attaches activation to previous layer or creates standalone.
        """
        input_var = node.value.args[0].id
        prev_is_tensorop = False
        prev_lyr_obj = None

        # Check if previous module is a TensorOp
        if input_var in self.module_of_output:
            prev_lyr_name = self.module_of_output[input_var]
            prev_lyr_obj = next((obj for obj in self.buml_model.modules if
                               obj.name == prev_lyr_name), None)
            prev_cls = prev_lyr_obj.__class__.__name__ if prev_lyr_obj else "None"
            if prev_lyr_obj and prev_cls == "TensorOp":
                prev_is_tensorop = True

        if prev_is_tensorop:
            # Create standalone activation layer
            actv_lyr_name = f"activ_{func_name}_{self.tensor_op_counter}"
            self.tensor_op_counter += 1
            actv_lyr = mm_classes.GeneralLayer(
                name=actv_lyr_name,
                actv_func=tf_actv_func_mapping[func_name]
            )
            self.buml_model.add_layer(actv_lyr)
            self.buml_model.modules.append(actv_lyr)

            # Track inputs/outputs
            output_var = node.targets[0].id
            self.inputs_outputs[actv_lyr_name] = [input_var, output_var]
            self.module_of_output[output_var] = actv_lyr_name
        else:
            # Attach activation to previous layer
            if prev_lyr_obj:
                prev_lyr_obj.actv_func = tf_actv_func_mapping[func_name]
                # Update layer output for nested calls
                if hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer:
                    if prev_lyr_name in self.inputs_outputs:
                        self.inputs_outputs[prev_lyr_name][1] = node.targets[0].id
            # Update module_of_output
            if input_var in self.module_of_output:
                self.module_of_output[node.targets[0].id] = self.module_of_output[input_var]

    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """
        op_type = node.value.func.attr
        op_args = node.value.args
        tensorop_param = None
        if op_type == "concat":
            op_args = node.value.args[0].elts
            if (op_args[0].id in self.module_of_output and
                op_args[1].id in self.module_of_output):
                lyr1 = self.module_of_output[op_args[0].id]
                lyr2 = self.module_of_output[op_args[1].id]
                if lyr1 != lyr2:
                    layers_of_tensors = [lyr1, lyr2]
                    cat_dim = self.param_value(node.value.keywords[0].value)
                    tensorop_param = {"tns_type": "concatenate",
                                      "layers_of_tensors": layers_of_tensors,
                                      "concatenate_dim": cat_dim}
        elif op_type == "matmul" or op_type == "multiply":
            op_type = "matmultiply" if op_type == "matmul" else "multiply"
            layers_of_tensors = [self.module_of_output[op_args[0].id],
                                 self.module_of_output[op_args[1].id]]
            tensorop_param = {"tns_type": op_type,
                              "layers_of_tensors": layers_of_tensors}
        elif op_type == "transpose":
            op_args = node.value.keywords[0].value.elts
            transpose_dim = [op_args[0].value, op_args[1].value,
                             op_args[2].value]
            tensorop_param = {"tns_type": op_type,
                              "transpose_dim": transpose_dim}
        elif op_type == "reshape":
            reshape_dim = [op_args[i].value for i in range(1, len(op_args))]
            tensorop_param = {"tns_type": op_type,
                              "reshape_dim": reshape_dim}
        else:
            print(f"{op_type} is not recognized!")

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1

    def handle_outer_attribute_assignment(self, node: ast.Assign):
        """
        It visits and extracts information from assignment statements
        (node attributes) called outside the NN class or the sequentail model.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the data_config dictionary.
        """
        config = self.data_config["config"]
        if (node.value.func.attr == "batch" and node.value.args and
                isinstance(node.value.args[0], ast.Constant)):
            config["batch_size"] = node.value.args[0].value
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.attr == "optimizers":
                self.get_params_from_optimizer(node)
            elif node.value.func.value.attr == "losses":
                loss = node.value.func.attr
                config["loss_function"] = loss_func_mapping[loss]
            elif node.value.func.attr == "image_dataset_from_directory":
                batch_size = next(
                    (k.value.value for k in node.value.keywords
                     if k.arg == "batch_size"),
                    None
                )
                if batch_size is not None:
                    config["batch_size"] = batch_size


    def handle_outer_assignments(self, node: ast.Assign):
        """
        It visits and extracts information from assignment statements
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config and data in 
                data_config dict.
        """
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            self.handle_outer_attribute_assignment(node)

        elif (isinstance(node.value, ast.Call) and
              isinstance(node.value.func, ast.Name)):
            self.handle_outer_simple_assignment(node)
        elif (isinstance(node.value, ast.List) and
              isinstance(node.targets[0], ast.Name)):
            if node.targets[0].id == "metrics":
                elts = node.value.elts
                config = self.data_config["config"]
                config["metrics"] = [elt.value for elt in elts]
        elif isinstance(node.value, ast.Constant):
            self.handle_outer_constant_assignment(node)
        elif isinstance(node.value, ast.Tuple):
            self.handle_outer_tuple_assignment(node)
        else:
            self.unprocessed_nodes.append(node)


    def get_images_attr(self, node: ast.Assign):
        """
        It extracts information related to images.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for data in data_config dict.
        """

        self.data_config["train_data"]["input_format"] = "images"
        if node.targets[0].elts[1].id != "_":
            self.data_config["train_data"]["normalize_images"] = True
        else:
            self.data_config["train_data"]["normalize_images"] = False
        if node.value.args and isinstance(node.value.args[0], ast.Constant):
            self.data_config["train_data"]["path_data"] = node.value.args[0].value


    def add_permute_dim(self):
        """
        It permutes input and output of cnn layers if needed
        to make pytorch and tensorflow equivalent.
        """
        cnns = ["Conv1D", "Conv2D", "Conv3D", "PoolingLayer"]
        bml_modules = self.buml_model.modules

        def iterate_and_permute(modules):
            prev_module = None
            lyr_out_permuted = []
            for i, module in enumerate(modules):
                if i == len(modules)-1:
                    next_module = None
                else:
                    next_module = modules[i+1]

                prev_module = self.permute(
                    module, bml_modules, prev_module, next_module,
                    lyr_out_permuted, cnns
                )

        if self.buml_model.sub_nns:
            for subnn in self.buml_model.sub_nns:
                iterate_and_permute(subnn.modules)

        iterate_and_permute(self.buml_model.modules)




    def permute(self, module, modules: list,
                prev_module, next_module,
                lyr_out_permuted: list, cnns: list):
        """
        It permutes input and output of `name` layer if needed
        
        Parameters:
            module: A buml module (either a layer, a subnn or a tensorop).
            modules (list): A list of modules.
            prev_module: The module defined before 'module' in the nn model.
            next_module: The module defined after 'module' in the nn model.
            lyr_out_permuted (list): A list containing the modules that their
                output has been already permuted.
            cnns (list): A list containing the names of layers to which 
                permutation applies.

        Returns:
            The module defined before 'module' in the nn model.
        
        """
        current_cnn, prev_cnn, next_cnn = False, False, False
        if isinstance(next_module, Layer):
            if next_module.__class__.__name__ in cnns:
                next_cnn = True

        if isinstance(module, Layer):
            if module.__class__.__name__ in cnns:
                current_cnn = True
                if module.name_module_input:
                    prev_module_name = module.name_module_input
                    prev_module = next((obj for obj in modules if
                                        obj.name == prev_module_name), None)

                if isinstance(next_module, Layer):
                    if next_module.name_module_input:
                        next_in = next_module.name_module_input
                        if next_in != module.name:
                            next_cnn = False

        if isinstance(prev_module, Layer):
            if prev_module.__class__.__name__ in cnns:
                prev_cnn = True
        else:
            prev_cnn = False
        if current_cnn:
            if not prev_cnn and (prev_module is None or 
                                 prev_module.name not in lyr_out_permuted):
                module.permute_in = True
                if prev_module is not None:
                    lyr_out_permuted.append(prev_module.name)
            elif prev_cnn and prev_module.name in lyr_out_permuted:
                module.permute_in = True
            if not next_cnn:
                module.permute_out = True
                lyr_out_permuted.append(module.name)
        prev_module = module
        return prev_module



def transform_layer(lyr_type: str, lyr_params: dict,
                    padding_amount: int | None, layer_name=None):
    """
    It transforms layers and their params from TensorFlow to BUML.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.
        padding_amount (int | None): It  keeps track of padding in
            ZeroPadding layer. In TF, padding is added to conv layers
            using a separate layer, but in PyTorch and BUML it is defined
            as an attribute of the conv layer.
        lyr_name (str): The name of the layer.

    Returns:
        The type of the layer and its parameters in BUML.
    """
    process_positional_params(lyr_type, lyr_params, pos_params)

    param_to_list(lyr_type, lyr_params, int2list_params,
                  lyrs_of_int2list_params)

    lyr_params, padding_amount = set_conv_padding(lyr_type, lyr_params,
                                                  padding_amount)

    lyr_params = process_params(lyr_type, lyr_params)
    lyr_params["name"] = layer_name
    if not lyr_type.startswith("ZeroPadding"):
        lyr_type = layers_mapping[lyr_type]

    return lyr_type, lyr_params, padding_amount



def process_params(lyr_type: str, lyr_params: dict):
    """
    It processes and transforms the layers' parameters.

    Parameters:
        lyr_type (str): The type of the layer (PyTorch).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.

    Returns:
        The parameters transformed into BUML.
    """

    updated_lyr_params = {}

    lyrs_units = rnn_layers + ["Dense"]
    if lyr_type in lyrs_units and "units" in lyr_params:
        param = "out_features" if lyr_type == "Dense" else "hidden_size"
        updated_lyr_params[param] = lyr_params["units"]

    for param in lyr_params:
        if param == "activation":
            updated_lyr_params["actv_func"] = lyr_params[param]
        elif param == "return_sequences" and lyr_params[param] is True:
            updated_lyr_params["return_type"] = "full"
        elif param == "return_state" and lyr_params[param] is True:
            updated_lyr_params["return_type"] = "hidden"
        elif param in params_mapping:
            param_name = params_mapping[param]
            updated_lyr_params[param_name] = lyr_params[param]
        elif param == "units":
            pass
        else:
            print(f"parameter {param} of layer {lyr_type} is not found!")

    if "return_type" not in updated_lyr_params and lyr_type in rnn_layers:
        updated_lyr_params["return_type"] = "last"

    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params


def set_conv_padding(layer_type, lyr_params, padding_amount):
    """
    If padding is used before conv layers, its amount is stored in
    the 'padding_amount' attribute. This function adds the padding 
    to the conv layer.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.
        padding_amount (int | None): It  keeps track of padding in 
            ZeroPadding layer. In TF, padding is added to conv layers 
            using a separate layer, but in PyTorch and BUML it is defined
            as an attribute of the conv layer.

    Returns:
        The type of the layer and its parameters in BUML. 
    """


    if layer_type.startswith("ZeroPadding"):
        padding_amount = lyr_params["padding"]
    elif layer_type.startswith("Conv"):
        if padding_amount is not None:
            lyr_params["padding_amount"] = padding_amount
            padding_amount = None

    return lyr_params, padding_amount

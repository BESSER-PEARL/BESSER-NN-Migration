"""
Module to extract information from the AST of a neural network
written in PyTorch and transforms it to a BUML model.
It also extracts data and model configuration attributes.
"""
import ast
import sys
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
from besser.BUML.metamodel.nn import NN, Layer
import besser.BUML.metamodel.nn as mm_classes

from torch2tf.definitions import (
    layers_mapping, params_mapping, static_params,
    pos_params, int2list_params, lyrs_of_int2list_params,
    actv_fun_mapping, cnn_layers, loss_func_mapping,
    functional_to_module_mapping
)
from ast_parser_nn import ASTParser
from transform_code import (
    process_positional_params, set_static_params, param_to_list
)


class ASTParserTorch(ASTParser):
    """
    Class visiting and parsing PyTorch code AST.

    Attributes:
        input_nn_type (str): The type of the nn input architecture.
        only_nn (str): Whether to process only the model definition or also
            its configuration and dataset.
        activation_functions (dict): it keeps track of the activation
            function of layers. Keys are layer names and values are
            activation function names.        
    """

    def __init__(self, input_nn_type: str, only_nn: bool):
        super().__init__(input_nn_type, only_nn)

        self.activation_functions = {}
        # Track RNN output vs hidden variables separately
        self.rnn_output_vars = {}  # {module_name: output_var}
        self.rnn_hidden_vars = {}  # {module_name: hidden_var}
        # Track variable aliases (e.g., h_last -> h)
        self.variable_aliases = {}  # {alias_var: source_var}
        # Track multi-layer RNNs: {module_name: [layer_name_0, layer_name_1, ...]}
        self.multi_layer_rnns = {}
        # Track layer reuse count for creating unique reuse names
        self.layer_reuse_count = {}  # {layer_name: reuse_count}

    def extract_subscript_pattern(self, subscript_node: ast.Subscript) -> str:
        """
        Extract the subscript/slice pattern from an AST Subscript node as a string.

        Parameters:
            subscript_node (ast.Subscript): The AST subscript node

        Returns:
            str: String representation of the subscript pattern (e.g., "[-1]", "[:, -1, :]")
        """
        def slice_to_string(slice_node):
            if isinstance(slice_node, ast.Slice):
                lower = ast.unparse(slice_node.lower) if slice_node.lower else ""
                upper = ast.unparse(slice_node.upper) if slice_node.upper else ""
                step = ast.unparse(slice_node.step) if slice_node.step else ""
                if step:
                    return f"{lower}:{upper}:{step}"
                else:
                    return f"{lower}:{upper}" if lower or upper else ":"
            elif isinstance(slice_node, ast.Tuple):
                # Multi-dimensional slicing like [:, -1, :]
                elements = [slice_to_string(elt) for elt in slice_node.elts]
                return ", ".join(elements)
            else:
                # Single index like -1 or 0
                return ast.unparse(slice_node)

        pattern = slice_to_string(subscript_node.slice)
        return f"[{pattern}]"

    def create_subscript_tensorop(self, source_var: str, subscript_pattern: str, output_var: str):
        """
        Create a subscript TensorOp for general slicing operations.

        Parameters:
            source_var (str): The variable being subscripted
            subscript_pattern (str): The subscript pattern as a string (e.g., "[-1]")
            output_var (str): The output variable name

        Returns:
            None, but adds TensorOp to BUML model
        """
        op_name = f"_subscript_op_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        # Resolve variable aliases to find the actual source
        resolved_var = source_var
        while resolved_var in self.variable_aliases:
            resolved_var = self.variable_aliases[resolved_var]

        # Get the source module that produced the variable
        # If it's the input variable 'x', use 'x' as a special marker
        source_module = self.module_of_output.get(resolved_var, resolved_var)


        subscript_op = mm_classes.TensorOp(
            name=op_name,
            tns_type='subscript',
            layers_of_tensors=[source_module],  # The layer/op that produced the tensor
            subscript_indices=subscript_pattern
        )

        self.buml_model.modules.append(subscript_op)
        self.inputs_outputs[op_name] = [resolved_var, output_var]
        self.module_of_output[output_var] = op_name


    def handle_subscript_operation(self, subscript_node: ast.Subscript, temp_name: str, node: ast.Assign):
        """
        Unified handler for subscript operations - detects RNN vs non-RNN and handles accordingly.

        Parameters:
            subscript_node (ast.Subscript): The subscript AST node
            temp_name (str): The temporary variable name for the result
            node (ast.Assign): The parent assignment node

        Returns:
            None, but creates TensorOp or sets RNN return_type
        """
        subscripted_var = subscript_node.value.id if isinstance(subscript_node.value, ast.Name) else None

        # Check if this is an RNN subscript
        is_rnn_subscript = False
        if subscripted_var and subscripted_var in self.module_of_output:
            src_module = self.module_of_output[subscripted_var]
            src_layer = next((obj for obj in self.buml_model.layers if obj.name == src_module), None)
            if src_layer and hasattr(src_layer, 'return_type'):
                is_rnn_subscript = True

        # Create temp assignment node
        temp_target = ast.Name(id=temp_name, ctx=ast.Store())
        subscript_assign = ast.Assign(targets=[temp_target], value=subscript_node)
        subscript_assign.lineno = node.lineno
        subscript_assign.col_offset = node.col_offset

        if is_rnn_subscript:
            # RNN slicing: process with handle_forward_slicing to set return_type
            self.handle_forward_slicing(subscript_assign)
        else:
            # Non-RNN slicing: create subscript TensorOp for general slicing
            subscript_pattern = self.extract_subscript_pattern(subscript_node)
            self.create_subscript_tensorop(subscripted_var, subscript_pattern, temp_name)

        self.previous_assign = subscript_assign

    def handle_init(self, node: ast.Assign):
        """
        It retrieves the sub_nn layers, adds their activation functions 
        as parameters and stores them in the 'sub_nn' dict. It also 
        retreives the layers and their parameters and stores them in
        the 'layers' dict.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        module_name = node.targets[0].attr
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute)):
            # Checks if it is a Sequential or any NN layer
            module_type = node.value.func.attr
            if module_type == "Sequential":
                self.handle_sequential_layers(node, module_name)
            else:
                lyr_type, lyr_params = self.extract_layer(node.value)
                if lyr_type not in actv_fun_mapping:
                    # Check if this is a multi-layer RNN (num_layers > 1)
                    num_layers = lyr_params.get('num_layers', 1)
                    is_rnn = lyr_type in ['RNN', 'LSTM', 'GRU']

                    if is_rnn and num_layers > 1:
                        # Process positional params first so we can access named params
                        process_positional_params(lyr_type, lyr_params, pos_params)

                        # Create multiple BUML layers for stacked RNN
                        layer_names = []
                        original_return_type = lyr_params.get('return_type', 'full')

                        for i in range(num_layers):
                            layer_params = lyr_params.copy()
                            # Remove num_layers from params as BUML doesn't support it
                            layer_params.pop('num_layers', None)

                            # Update layer name
                            layer_params['name'] = f"{module_name}_layer_{i}"
                            layer_names.append(layer_params['name'])

                            # First layer keeps input_size, others use hidden_size as input
                            if i > 0:
                                layer_params['input_size'] = lyr_params['hidden_size']

                            # All layers except last must return full sequence
                            if i < num_layers - 1:
                                layer_params['return_type'] = 'full'
                            else:
                                layer_params['return_type'] = original_return_type

                            # Add empty positional_params for transform_layer
                            layer_params['positional_params'] = []

                            # Transform and create BUML layer
                            buml_lyr_type, buml_params = transform_layer(
                                lyr_type, layer_params, layer_params['name']
                            )
                            buml_layer = getattr(mm_classes, buml_lyr_type)(**buml_params)
                            self.buml_model.add_layer(buml_layer)

                        # Track this multi-layer RNN
                        self.multi_layer_rnns[module_name] = layer_names
                    else:
                        # Single layer - remove num_layers if present
                        lyr_params.pop('num_layers', None)
                        lyr_type, lyr_params = transform_layer(
                            lyr_type, lyr_params, module_name
                        )
                        buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
                        self.buml_model.add_layer(buml_layer)
                else:
                    self.activation_functions[module_name] = lyr_type

        # Used to get the proper order from forward the method
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
        permute = False
        # Extracts layers within Sequential
        for elt in node.value.args:
            if isinstance(elt, ast.Call):
                lyr_type, lyr_params = self.extract_layer(elt)

                if lyr_type in actv_fun_mapping:
                    # Check if previous layer supports activation functions
                    # Using same logic as generator's add_separate_activation_if_needed
                    prev_layer_supports_activ = False
                    if subnn.layers:
                        prev_layer = subnn.layers[-1]
                        prev_layer_class = prev_layer.__class__.__name__
                        prev_layer_parent = prev_layer.__class__.mro()[1].__name__

                        # Layers that DON'T support activation (same as generator logic)
                        unsupported = prev_layer_parent in ["NormalizationLayer", "LayerModifier"] or \
                                     prev_layer_class in ["EmbeddingLayer", "PoolingLayer", "FlattenLayer"]

                        prev_layer_supports_activ = not unsupported

                    if prev_layer_supports_activ:
                        # Merge with previous layer as activation attribute
                        subnn.layers[-1].actv_func = actv_fun_mapping[lyr_type]
                    else:
                        # Create standalone activation layer
                        actv_params = {
                            "name": f"layer_{layer_id}",
                            "actv_func": actv_fun_mapping[lyr_type]
                        }
                        subnn_layer = getattr(mm_classes, "GeneralLayer")(**actv_params)
                        subnn.add_layer(subnn_layer)
                        layer_id += 1
                elif lyr_type == "Permute":
                    last_lyr = subnn.layers[-1] if subnn.layers else None

                    if last_lyr is not None:
                        last_lyr_type = subnn.layers[-1].__class__.__name__
                        if last_lyr_type in cnn_layers:
                            subnn.layers[-1].permute_out = True
                        else:
                            permute = True
                    else:
                        permute = True
                else:
                    lyr_type, lyr_params = transform_layer(
                        lyr_type, lyr_params
                    )

                    lyr_params["name"] = f"layer_{layer_id}"
                    if permute:
                        lyr_params["permute_in"] = True
                        permute = False

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
        Processes forward method assignments including chained and nested calls.
        """
        # Handle chained calls (x.method1().method2())
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

                        # Link next call to this temp variable
                        if i + 1 < len(chain):
                            next_call = chain[i + 1]
                            next_func_value = next_call.func.value

                            # Check if next call is functional (F.xxx) or method (x.xxx)
                            if isinstance(next_func_value, ast.Name):
                                # Functional call like F.relu(x) where x is in args[0]
                                if next_call.args:
                                    next_call.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
                            elif isinstance(next_func_value, ast.Call):
                                # Method call like result.permute(...) where result is func.value
                                next_call.func.value = ast.Name(id=temp_name, ctx=ast.Load())

                    # Check if this call in the chain is nested
                    if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Call):
                        # Nested call within chain: decompose it first
                        inner_call = call.args[0]
                        inner_temp = f"_nested_in_chain_{self.tensor_op_counter}"
                        self.tensor_op_counter += 1
                        inner_target = ast.Name(id=inner_temp, ctx=ast.Store())
                        inner_node = ast.Assign(targets=[inner_target], value=inner_call)
                        inner_node.lineno = node.lineno
                        inner_node.col_offset = node.col_offset

                        # Process inner call without nested flag (it's just a layer)
                        self.process_single_call(inner_node)
                        self.previous_assign = inner_node

                        # Replace nested call with temp variable
                        call.args[0] = ast.Name(id=inner_temp, ctx=ast.Load())

                    synthetic_node.lineno = node.lineno
                    synthetic_node.col_offset = node.col_offset

                    # Mark as nested outer part if this is activation on the inner temp
                    is_nested_activation = False
                    if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Name):
                        if call.args[0].id.startswith('_nested_in_chain_'):
                            self.is_processing_nested_outer = True
                            is_nested_activation = True

                    self.process_single_call(synthetic_node)
                    self.previous_assign = synthetic_node

                    if is_nested_activation:
                        self.is_processing_nested_outer = False
                return

        # Handle subscript arguments (e.g., self.fc(h[-1]))
        # Process subscript before processing the call
        if isinstance(node.value, ast.Call) and node.value.args:
            if isinstance(node.value.args[0], ast.Subscript):
                subscript_node = node.value.args[0]
                temp_name = f"_subscript_temp_{self.tensor_op_counter}"
                self.tensor_op_counter += 1

                self.handle_subscript_operation(subscript_node, temp_name, node)

                # Replace subscript with temp variable in call
                node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())

        # Handle nested calls (F.relu(self.conv(x)))
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
                # Recursively call visit_Assign to handle multi-level nesting
                self.visit_Assign(inner_node)
                self.previous_assign = inner_node

                # Replace inner call with temp variable in outer call
                node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
                is_nested_call = True

        # Mark if this is a nested call's outer part (activation wrapping layer)
        if is_nested_call:
            self.is_processing_nested_outer = True

        self.process_single_call(node)
        self.previous_assign = node

        if is_nested_call:
            self.is_processing_nested_outer = False

    def process_single_call(self, node: ast.Assign):
        """
        Process a single (non-chained) call operation.
        Handles both module API (self.layer) and functional API (F.layer).
        """
        if not (isinstance(node.value, ast.Call) and
                hasattr(node.value.func, 'value') and
                isinstance(node.value.func.value, ast.Name)):
            self.extract_tensorop(node)
            return

        caller_id = node.value.func.value.id

        # Handle functional API (F.layer, torch.nn.functional.layer, or torch.layer)
        if caller_id == "F" or caller_id == "functional" or caller_id == "torch":
            func_name = node.value.func.attr

            # Check if it maps to a known module
            if func_name in functional_to_module_mapping:
                module_name = functional_to_module_mapping[func_name]

                # Generate synthetic layer name
                synthetic_name = f"f_{func_name}_{self.tensor_op_counter}"
                self.tensor_op_counter += 1

                # Check if it's an activation function
                if module_name in actv_fun_mapping:
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
                        # Create standalone activation layer using GeneralLayer
                        actv_lyr_name = f"activ_{module_name}_{self.tensor_op_counter}"
                        self.tensor_op_counter += 1
                        actv_lyr = mm_classes.GeneralLayer(
                            name=actv_lyr_name,
                            actv_func=actv_fun_mapping[module_name]
                        )
                        self.buml_model.add_layer(actv_lyr)
                        self.buml_model.modules.append(actv_lyr)

                        # Track inputs/outputs
                        output_var = node.targets[0].id
                        self.inputs_outputs[actv_lyr_name] = [input_var, output_var]
                        self.module_of_output[output_var] = actv_lyr_name
                    else:
                        # Attach activation to previous layer (original behavior)
                        if prev_lyr_obj:
                            prev_lyr_obj.actv_func = actv_fun_mapping[module_name]
                            # Update the layer's output to reflect final output (only for nested calls)
                            if hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer:
                                if prev_lyr_name in self.inputs_outputs:
                                    self.inputs_outputs[prev_lyr_name][1] = node.targets[0].id
                        # Update layer_of_output to point to the previous layer (activation is inline)
                        if input_var in self.module_of_output:
                            self.module_of_output[node.targets[0].id] = self.module_of_output[input_var]
                else:
                    # Handle as regular layer
                    lyr_params = {'positional_params': []}
                    for arg in node.value.args[1:]:  # Skip first arg (input tensor)
                        lyr_params['positional_params'].append(self.param_value(arg))

                    for kw in node.value.keywords:
                        lyr_params[kw.arg] = self.param_value(kw.value)

                    # Transform using existing logic
                    lyr_type, lyr_params = transform_layer(module_name, lyr_params, synthetic_name)

                    # Create and add layer
                    lyr_obj = getattr(mm_classes, lyr_type)(**lyr_params)
                    self.buml_model.add_layer(lyr_obj)

                    # Track inputs/outputs
                    self.inputs_outputs[synthetic_name] = [node.value.args[0].id, node.targets[0].id]
                    self.module_of_output[node.targets[0].id] = synthetic_name
                    self.buml_model.modules.append(lyr_obj)
            else:
                # Unknown functional - treat as tensor op
                self.extract_tensorop(node)
            return

        # Handle module API (self.layer)
        if caller_id == "self":
            # Populates inputs_outputs and layer_of_output from forward method
            module_name = node.value.func.attr

            # Check if this is a multi-layer RNN that needs to be expanded
            if module_name in self.multi_layer_rnns:
                self.expand_multi_layer_rnn_call(node, module_name, is_tuple=False)
                return

            # Extract input variable, handling inline operations like unsqueeze
            input_arg = node.value.args[0]
            if isinstance(input_arg, ast.Name):
                # Simple variable: x
                input_var = input_arg.id
            elif isinstance(input_arg, ast.Call):
                # Method call: x.unsqueeze(1) or x.squeeze(1)
                # Create a synthetic intermediate operation
                input_var = self.extract_inline_tensorop(input_arg, node)
            else:
                input_var = "x"  # Fallback

            self.inputs_outputs[module_name] = [input_var, node.targets[0].id]
            self.module_of_output[node.targets[0].id] = module_name

            # Check if input variable was saved for residual connection
            # If so, mark the layer to use a new output variable instead of reusing input
            if hasattr(self, '_variables_saved_for_residual') and input_var in self._variables_saved_for_residual:
                # Find the layer object and mark it
                module_obj = next((obj for obj in self.buml_model.layers if obj.name == module_name), None)
                if module_obj:
                    module_obj.input_reused = True
                # Remove from set since we've handled this case
                self._variables_saved_for_residual.discard(input_var)

            is_subnn_obj = next((obj for obj in self.buml_model.sub_nns if
                                 obj.name == module_name), None)

            if module_name in self.activation_functions:
                # Only merge activation into previous layer if:
                # 1. We're not in a nested call scenario (activation should be standalone)
                # 2. The previous assign was a simple layer call
                should_merge = not (hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer)

                if should_merge and self.previous_assign and isinstance(self.previous_assign.value, ast.Call):
                    if (hasattr(self.previous_assign.value.func, 'attr')):
                        prev_lyr_name = self.previous_assign.value.func.attr
                        prev_lyr_obj = next((obj for obj in self.buml_model.modules if
                                             obj.name == prev_lyr_name), None)
                        if prev_lyr_obj:
                            actv = self.activation_functions[module_name]
                            prev_lyr_obj.actv_func = actv_fun_mapping[actv]
                            # Update module_of_output to point to the layer, not the activation
                            output_var = node.targets[0].id
                            self.module_of_output[output_var] = prev_lyr_name
                            # Skip adding activation as separate layer
                            should_merge = True
                        else:
                            should_merge = False
                    else:
                        should_merge = False
                else:
                    should_merge = False

                # If not merging, create activation as a standalone GeneralLayer
                if not should_merge:
                    # Create unique name for standalone activation to avoid collisions
                    unique_name = f"{module_name}_{self.tensor_op_counter}"
                    self.tensor_op_counter += 1

                    actv = self.activation_functions[module_name]
                    actv_lyr = mm_classes.GeneralLayer(
                        name=unique_name,
                        actv_func=actv_fun_mapping[actv]
                    )
                    self.buml_model.modules.append(actv_lyr)

                    # Track inputs/outputs for standalone activation
                    input_var = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else "x"
                    output_var = node.targets[0].id
                    self.inputs_outputs[unique_name] = [input_var, output_var]
                    self.module_of_output[output_var] = unique_name

            elif not is_subnn_obj:
                self.is_permute_before_cnn(module_name)

            if module_name not in self.activation_functions:
                module_obj = next((obj for obj in self.buml_model.layers if
                                   obj.name == module_name), None)
                if not module_obj:
                    subnns = self.buml_model.sub_nns
                    module_obj = next((obj for obj in subnns if
                                       obj.name == module_name), None)

                # Check if this layer was already used - if so, create a synthetic instance
                # to track this specific use
                if module_obj and module_obj in self.buml_model.modules:
                    # Layer reuse detected - create a synthetic copy
                    import copy
                    synthetic_name = f"{module_name}_use_{self.tensor_op_counter}"
                    self.tensor_op_counter += 1

                    # Create a shallow copy with a new name
                    synthetic_module = copy.copy(module_obj)
                    synthetic_module.name = synthetic_name

                    # Update inputs_outputs to use the synthetic name
                    if module_name in self.inputs_outputs:
                        self.inputs_outputs[synthetic_name] = self.inputs_outputs[module_name]

                    # Update module_of_output for the output variable
                    output_var = node.targets[0].id
                    self.module_of_output[output_var] = synthetic_name

                    # Append synthetic module
                    self.buml_model.layers.append(synthetic_module)
                    self.buml_model.modules.append(synthetic_module)
                elif module_obj:
                    self.buml_model.modules.append(module_obj)

        else:
            #tensorops
            self.extract_tensorop(node)

    def expand_multi_layer_rnn_call(self, node: ast.Assign, module_name: str, is_tuple: bool):
        """
        Expand a multi-layer RNN call into sequential calls to individual layers.

        For example, if self.gru has num_layers=3:
        - out, h = self.gru(x) becomes:
          - temp_0 = self.gru_layer_0(x)
          - temp_1 = self.gru_layer_1(temp_0)
          - out, h = self.gru_layer_2(temp_1)
        """
        layer_names = self.multi_layer_rnns[module_name]

        # Get the input variable from the original call
        input_arg = node.value.args[0]
        if isinstance(input_arg, ast.Name):
            current_input = input_arg.id
        elif isinstance(input_arg, ast.Call):
            current_input = self.extract_inline_tensorop(input_arg, node)
        else:
            current_input = "x"

        # Process each layer in the stack
        for i, layer_name in enumerate(layer_names):
            is_last_layer = (i == len(layer_names) - 1)

            if is_last_layer:
                # Last layer - modify the original node and let the normal processing handle it
                node.value.func.attr = layer_name
                if i > 0:
                    # Update input to use temp var from previous layer
                    node.value.args[0] = ast.Name(id=current_input, ctx=ast.Load())

                # Now process this node with normal logic (no recursion since we're using layer_name not module_name)
                if is_tuple:
                    # Manually inline the tuple assignment logic to avoid recursion
                    var1 = node.targets[0].elts[0].id if not isinstance(node.targets[0].elts[0], ast.Tuple) else None
                    var2 = node.targets[0].elts[1].id if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0].id

                    if var1 and var1 != "_":
                        self.rnn_output_vars[layer_name] = var1
                        self.module_of_output[var1] = layer_name
                    if var2 and var2 != "_":
                        self.rnn_hidden_vars[layer_name] = var2
                        self.module_of_output[var2] = layer_name

                    first_elem = node.targets[0].elts[0]
                    second_elem = node.targets[0].elts[1] if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0]
                    first_is_underscore = isinstance(first_elem, ast.Name) and first_elem.id == "_"
                    second_is_underscore = isinstance(second_elem, ast.Name) and second_elem.id == "_"

                    lyr_obj = next((obj for obj in self.buml_model.layers if obj.name == layer_name), None)

                    if first_is_underscore and not second_is_underscore:
                        rnn_out = node.targets[0].elts[1].id if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0].id
                        if lyr_obj:
                            lyr_obj.return_type = "hidden"
                    elif not first_is_underscore and second_is_underscore:
                        rnn_out = node.targets[0].elts[0].id
                        if lyr_obj:
                            lyr_obj.return_type = "full"
                    else:
                        rnn_out = node.targets[0].elts[0].id
                        if lyr_obj:
                            lyr_obj.return_type = "both"

                    self.inputs_outputs[layer_name] = [current_input, rnn_out]

                    if lyr_obj:
                        self.buml_model.modules.append(lyr_obj)
                else:
                    # Simple call
                    output_var = node.targets[0].id
                    self.inputs_outputs[layer_name] = [current_input, output_var]
                    self.module_of_output[output_var] = layer_name

                    lyr_obj = next((obj for obj in self.buml_model.layers if obj.name == layer_name), None)
                    if lyr_obj:
                        self.buml_model.modules.append(lyr_obj)

            else:
                # Intermediate layer - simple call storing to temp variable
                temp_var = f"_mlrnn_{module_name}_{i}"
                self.inputs_outputs[layer_name] = [current_input, temp_var]
                self.module_of_output[temp_var] = layer_name

                # Add layer to modules
                module_obj = next((obj for obj in self.buml_model.layers if obj.name == layer_name), None)
                if module_obj:
                    self.buml_model.modules.append(module_obj)

                current_input = temp_var

    def handle_forward_tuple_assignment(self, node: ast.Assign):
        """
        It handles rnn tuple assignments such as
        'x, _ = self.l4(x)' or 'out, hidden = self.rnn(x)'

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        # Check if this is a method call (e.g., x.max(dim=1)) vs module call (e.g., self.rnn(x))
        if hasattr(node.value.func, 'value') and isinstance(node.value.func.value, ast.Name):
            caller_id = node.value.func.value.id
            # If caller is not "self", it's a method call on a tensor - handle as TensorOp
            if caller_id != "self":
                # This is a method call like x.max(dim=1) that returns a tuple
                # Extract the first value and handle as TensorOp
                self.extract_tensorop(node)
                return

        module_name = node.value.func.attr

        # Check if this is a multi-layer RNN that needs to be expanded
        if module_name in self.multi_layer_rnns:
            self.expand_multi_layer_rnn_call(node, module_name, is_tuple=True)
            return

        # Track both variables (output and hidden) for later slicing operations
        var1 = node.targets[0].elts[0].id if not isinstance(node.targets[0].elts[0], ast.Tuple) else None
        var2 = node.targets[0].elts[1].id if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0].id

        # For RNNs: var1 is typically output sequence, var2 is hidden state
        # Track which variable is which for this RNN module
        if var1 and var1 != "_":
            self.rnn_output_vars[module_name] = var1
            self.module_of_output[var1] = module_name
        if var2 and var2 != "_":
            self.rnn_hidden_vars[module_name] = var2
            self.module_of_output[var2] = module_name

        # Determine which is the main output for inputs_outputs tracking
        # Determine return_type based on which elements are used
        first_elem = node.targets[0].elts[0]
        second_elem = node.targets[0].elts[1] if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0]

        first_is_underscore = isinstance(first_elem, ast.Name) and first_elem.id == "_"
        second_is_underscore = isinstance(second_elem, ast.Name) and second_elem.id == "_"

        lyr_obj = next((obj for obj in self.buml_model.layers if
                        obj.name == module_name), None)

        if first_is_underscore and not second_is_underscore:
            # Only hidden state is used: _, h = rnn(x)
            if isinstance(node.targets[0].elts[1], ast.Tuple):
                rnn_out = node.targets[0].elts[1].elts[0].id
            else:
                rnn_out = node.targets[0].elts[1].id
            if lyr_obj:
                lyr_obj.return_type = "hidden"
        elif not first_is_underscore and second_is_underscore:
            # Only output sequence is used: out, _ = rnn(x)
            rnn_out = node.targets[0].elts[0].id
            if lyr_obj:
                lyr_obj.return_type = "full"
        else:
            # Both are used: out, h = rnn(x)
            rnn_out = node.targets[0].elts[0].id
            if lyr_obj:
                lyr_obj.return_type = "both"

        # Extract input variable - handle both simple variables and method calls
        input_arg = node.value.args[0]
        if isinstance(input_arg, ast.Name):
            # Simple variable: x
            rnn_in = input_arg.id
        elif isinstance(input_arg, ast.Call):
            # Method call: x.unsqueeze(1) or x.squeeze(1)
            # Create a synthetic intermediate operation
            rnn_in = self.extract_inline_tensorop(input_arg, node)
        else:
            # Fallback
            rnn_in = "x"

        self.inputs_outputs[module_name] = [rnn_in, rnn_out]

        module_obj = next((obj for obj in self.buml_model.layers if
                           obj.name == module_name), None)
        if not module_obj:
            module_obj = next((obj for obj in self.buml_model.sub_nns if
                               obj.name == module_name), None)
        self.buml_model.modules.append(module_obj)
        self.previous_assign = node

    def handle_forward_variable_assignment(self, node: ast.Assign):
        """
        Handle simple variable assignments like inp = x or r = x (residual).
        Tracks variable aliasing and marks source as reused for later operations.

        Parameters:
            node (ast.Assign): The AST node representing the assignment

        Returns:
            None, but updates tracking dictionaries
        """
        target_var = node.targets[0].id
        source_var = node.value.id

        # If source is tracked in module_of_output, propagate it to target
        if source_var in self.module_of_output:
            source_module = self.module_of_output[source_var]
            self.module_of_output[target_var] = source_module

            # Track that this variable has been saved for later use (residual connection)
            # The NEXT operation that modifies source_var should use a new output variable
            # We'll set a flag that the next layer/op processing can check
            if not hasattr(self, '_variables_saved_for_residual'):
                self._variables_saved_for_residual = set()
            self._variables_saved_for_residual.add(source_var)
        else:
            # Source is not tracked, mark as network input with special marker
            self.module_of_output[target_var] = 'INPUT'

        self.previous_assign = node

    def handle_forward_binop(self, node: ast.Assign):
        """
        It handles binary operations such as 'x = a + b' or 'x = a.squeeze(1) + b'
        Extracts any inline tensor operations (like squeeze) and creates a TensorOp for the binop.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        binop = node.value

        # Extract left operand variable
        if isinstance(binop.left, ast.Name):
            left_var = binop.left.id
        elif isinstance(binop.left, ast.Call):
            # Extract inline operation like x.squeeze(1)
            left_var = self.extract_inline_tensorop(binop.left, node)
        elif isinstance(binop.left, ast.Subscript):
            # Handle subscript like out[:, -1, :]
            temp_name = f"_subscript_temp_{self.tensor_op_counter}"
            self.tensor_op_counter += 1
            self.handle_subscript_operation(binop.left, temp_name, node)
            left_var = temp_name
        elif isinstance(binop.left, (ast.Constant, ast.Num)):
            # Handle constant operand
            left_var = binop.left.value if isinstance(binop.left, ast.Constant) else binop.left.n
        elif isinstance(binop.left, ast.BinOp):
            # Handle nested binop like (x1 + x2) + x3
            temp_name = f"_binop_temp_{self.tensor_op_counter}"
            self.tensor_op_counter += 1
            temp_target = ast.Name(id=temp_name, ctx=ast.Store())
            binop_assign = ast.Assign(targets=[temp_target], value=binop.left)
            binop_assign.lineno = node.lineno
            binop_assign.col_offset = node.col_offset
            self.handle_forward_binop(binop_assign)
            left_var = temp_name
        else:
            print(f"Warning: Unsupported left operand type in BinOp: {type(binop.left).__name__}")
            return

        # Extract right operand variable
        if isinstance(binop.right, ast.Name):
            right_var = binop.right.id
        elif isinstance(binop.right, ast.Call):
            # Extract inline operation
            right_var = self.extract_inline_tensorop(binop.right, node)
        elif isinstance(binop.right, ast.Subscript):
            # Handle subscript like out[:, -1, :]
            temp_name = f"_subscript_temp_{self.tensor_op_counter}"
            self.tensor_op_counter += 1
            self.handle_subscript_operation(binop.right, temp_name, node)
            right_var = temp_name
        elif isinstance(binop.right, (ast.Constant, ast.Num)):
            # Handle constant operand
            right_var = binop.right.value if isinstance(binop.right, ast.Constant) else binop.right.n
        elif isinstance(binop.right, ast.BinOp):
            # Handle nested binop like x1 + (x2 + x3)
            temp_name = f"_binop_temp_{self.tensor_op_counter}"
            self.tensor_op_counter += 1
            temp_target = ast.Name(id=temp_name, ctx=ast.Store())
            binop_assign = ast.Assign(targets=[temp_target], value=binop.right)
            binop_assign.lineno = node.lineno
            binop_assign.col_offset = node.col_offset
            self.handle_forward_binop(binop_assign)
            right_var = temp_name
        else:
            print(f"Warning: Unsupported right operand type in BinOp: {type(binop.right).__name__}")
            return

        # Determine the operation type
        op_map = {
            'Add': 'binop_add',
            'Sub': 'binop_subtract',
            'Mult': 'binop_multiply',
            'Div': 'binop_divide'
        }
        op_type_name = binop.op.__class__.__name__
        tns_type = op_map.get(op_type_name)

        if tns_type is None:
            print(f"Warning: Unsupported binary operation: {op_type_name}")
            return

        # Get the layer/module names that produced these variables
        # For constants (float/int), we use the value directly instead of a layer name
        left_layer = left_var if isinstance(left_var, (int, float)) else self.module_of_output.get(left_var)
        right_layer = right_var if isinstance(right_var, (int, float)) else self.module_of_output.get(right_var)

        if left_layer is None or right_layer is None:
            print(f"Warning: Cannot determine source layers for BinOp operands")
            return

        # Determine if variables are output or hidden for RNNs with return_type="both"
        # Resolve aliases first
        actual_left_var = self.variable_aliases.get(left_var, left_var) if isinstance(left_var, str) else left_var
        actual_right_var = self.variable_aliases.get(right_var, right_var) if isinstance(right_var, str) else right_var

        var_types = []
        for lyr_name, actual_var in zip([left_layer, right_layer], [actual_left_var, actual_right_var]):
            # Skip type checking for constant values
            if isinstance(lyr_name, (int, float)):
                var_types.append("output")  # Constants don't have type, use default
            elif lyr_name in self.rnn_hidden_vars and actual_var == self.rnn_hidden_vars[lyr_name]:
                var_types.append("hidden")
            elif lyr_name in self.rnn_output_vars and actual_var == self.rnn_output_vars[lyr_name]:
                var_types.append("output")
            else:
                var_types.append("output")  # Default

        # Create TensorOp for the binary operation
        tensorop_param = {
            "tns_type": tns_type,
            "layers_of_tensors": [left_layer, right_layer],
            "actual_vars": var_types,  # Track which component (output/hidden) each refers to
            "name": f"op_{self.tensor_op_counter}"
        }

        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1

        # Track the output variable
        output_var = node.targets[0].id
        self.module_of_output[output_var] = tensorop_param["name"]

    def handle_forward_shape_unpacking(self, node: ast.Assign):
        """
        Handle tuple unpacking of shape attributes (e.g., b, t, _ = x.shape).
        Creates TensorOp assignments for each shape dimension.

        Parameters:
            node (ast.Assign): The AST node with tuple target and attribute value.

        Returns:
            None, but creates TensorOps for shape dimension extraction.
        """
        # Extract the tuple elements (variable names)
        tuple_target = node.targets[0]
        var_names = []
        for elt in tuple_target.elts:
            if isinstance(elt, ast.Name):
                var_names.append(elt.id)
            else:
                var_names.append('_')  # Placeholder for unused variables

        # Extract the source variable (e.g., 'x' from 'x.shape')
        if isinstance(node.value.value, ast.Name):
            source_var = node.value.value.id
            attr_name = node.value.attr

            # Only handle .shape attribute
            if attr_name == 'shape':
                # Create a TensorOp for each non-underscore variable
                for idx, var_name in enumerate(var_names):
                    if var_name != '_':  # Skip underscore placeholders
                        # Use the variable name directly as the TensorOp name
                        # This ensures the generated code uses the correct variable names
                        tensorop_param = {
                            "name": var_name,
                            "tns_type": "shape_dim",
                            "layers_of_tensors": [source_var],  # Source variable
                            "reduce_dim": idx  # Dimension index
                        }
                        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
                        self.buml_model.add_tensor_op(tns_obj)

                        # Track the output variable
                        self.module_of_output[var_name] = var_name

    def handle_forward_slicing(self, node: ast.Assign):
        """
        It handles rnn slicing calls such as 'x = x[:, -1, :]' or 'h = h[-1]'

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        # Get the variable being subscripted (e.g., 'h1' from 'h1[-1]')
        subscripted_var = node.value.value.id
        result_var = node.targets[0].id

        # Look up which module produced this variable
        if subscripted_var in self.module_of_output:
            prev_module_name = self.module_of_output[subscripted_var]

            # Try to find the layer object
            lyr_obj = next((obj for obj in self.buml_model.layers if
                            obj.name == prev_module_name), None)

            # If layer found and is RNN, handle RNN-specific slicing
            if lyr_obj and hasattr(lyr_obj, 'return_type'):
                # RNN layer - continue with RNN-specific processing below
                pass
            else:
                # Not an RNN layer - create subscript TensorOp
                result_var = node.targets[0].id
                subscript_pattern = self.extract_subscript_pattern(node.value)
                self.create_subscript_tensorop(subscripted_var, subscript_pattern, result_var)
                self.previous_assign = node
                return
        else:
            # Variable not in module_of_output - likely an input parameter or undefined
            # Create subscript TensorOp (create_subscript_tensorop handles unknown vars)
            result_var = node.targets[0].id
            subscript_pattern = self.extract_subscript_pattern(node.value)
            self.create_subscript_tensorop(subscripted_var, subscript_pattern, result_var)
            self.previous_assign = node
            return

        # Determine the return type based on slicing pattern
        if isinstance(node.value.slice, ast.UnaryOp):
            # Pattern: h[-1] means extracting last layer's hidden state
            # Check if BOTH output and hidden were captured (not underscore)
            has_output = prev_module_name in self.rnn_output_vars
            has_hidden = prev_module_name in self.rnn_hidden_vars

            if has_output and has_hidden:
                # Both output and hidden were captured - need "both"
                lyr_obj.return_type = "both"
            else:
                # Only hidden was captured
                lyr_obj.return_type = "hidden"
        elif isinstance(node.value.slice, ast.Tuple) and len(node.value.slice.elts) == 3:
            # Pattern: out[:, -1, :] means extracting last timestep from sequence
            # Check if this is extracting from an already-returned output
            # If the RNN already has return_sequences=True, create a subscript TensorOp instead
            if lyr_obj.return_type == "full":
                # RNN returns full sequence, create subscript TensorOp to extract last timestep
                subscript_pattern = self.extract_subscript_pattern(node.value)
                self.create_subscript_tensorop(subscripted_var, subscript_pattern, result_var)
                self.previous_assign = node
                return
            # Don't overwrite "both" if it was already set from a previous slicing operation
            if lyr_obj.return_type != "both":
                lyr_obj.return_type = "last"
        else:
            print(f"Warning: Unrecognized subscript pattern on '{subscripted_var}'")

        # Track the result variable so it can be used in subsequent operations (e.g., concat)
        result_var = node.targets[0].id
        self.module_of_output[result_var] = prev_module_name

        # IMPORTANT: Track which actual variable this sliced result came from
        # This is crucial for correct concat operations
        # e.g., h1_last = h1[-1] means h1_last is an alias for h1
        self.variable_aliases[result_var] = subscripted_var

        self.previous_assign = node


    def get_path_data(self, node: ast.Assign):
        """
        It extracts the path for training and test data
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the data_config dict with the data path. 
        """
        keywords = node.value.keywords
        path = next(
            (k.value.value for k in keywords if k.arg == 'root'), None
        )
        if "train" in node.targets[0].id or "train" in path:
            self.data_config["train_data"]["path_data"] = path

        elif "test" in node.targets[0].id or "test" in path:
            self.data_config["test_data"]["path_data"] = path

        else:
            print("Path is not recognised!")


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
        transform_args = node.value.args[0].elts
        normalize_arg = next((
            arg for arg in transform_args if arg.func.attr == "Normalize"),
            None
        )

        if normalize_arg:
            self.data_config["train_data"]["normalize_images"] = True


    def is_permute_before_cnn(self, lyr_name: str):
        """
        It adds the permute op as parameter to its following layer if
        it is a cnn layer.

        Parameters:
            lyr_name (str): The name of the layer.

        Returns:
            None, but populates the buml model.
        """
        lyr_obj = next((obj for obj in self.buml_model.layers if
                        obj.name == lyr_name), None)
        lyr_type = lyr_obj.__class__.__name__
        if (lyr_type in cnn_layers and len(self.buml_model.tensor_ops)!=0):

            if (isinstance(self.previous_assign.targets[0], ast.Name) and
                isinstance(self.previous_assign.value, ast.Call)):
                # Safe check: ensure func.value exists and is a Name before accessing .id
                if (hasattr(self.previous_assign.value.func, 'value') and
                    isinstance(self.previous_assign.value.func.value, ast.Name) and
                    self.previous_assign.value.func.value.id != "self"):
                    ops_name = self.previous_assign.value.func.attr
                    if ops_name == "permute":
                        lyrs = self.buml_model.layers
                        lyr_obj = next((obj for obj in lyrs if
                                        obj.name == lyr_name), None)
                        lyr_obj.permute_in = True
                        self.buml_model.tensor_ops.pop()
                        self.buml_model.modules.pop()


    def extract_inline_tensorop(self, call_node: ast.Call, parent_node: ast.Assign):
        """
        Extracts inline tensor operations like x.unsqueeze(1) or x.squeeze(1)
        and creates a synthetic intermediate operation.

        Parameters:
            call_node (ast.Call): The inline method call node
            parent_node (ast.Assign): The parent assignment node

        Returns:
            str: The name of the intermediate variable created
        """
        # Check if it's a method call
        if not isinstance(call_node.func, ast.Attribute):
            return "x"

        op_type = call_node.func.attr

        # Check if it's a method call on a variable or on another call (chained)
        if isinstance(call_node.func.value, ast.Name):
            base_var = call_node.func.value.id
        elif isinstance(call_node.func.value, ast.Call):
            # Chained call like h.unsqueeze(1).squeeze(1)
            # Recursively process the inner call first
            base_var = self.extract_inline_tensorop(call_node.func.value, parent_node)
        else:
            # Not a simple method call, return default
            if hasattr(call_node.func, 'value') and hasattr(call_node.func.value, 'id'):
                return call_node.func.value.id
            return "x"

        # Only handle squeeze/unsqueeze/repeat for now
        # Layer calls (self.layer_name(...)) should be handled by the caller
        if op_type not in ['squeeze', 'unsqueeze', 'repeat']:
            # If this is a layer call (base_var == 'self'), signal to caller
            if base_var == 'self':
                return None  # Signal that this needs special handling
            return base_var

        # Create intermediate variable name
        intermediate_var = f"_inline_{op_type}_{self.tensor_op_counter}"

        # Get the layer/module that produced the base variable
        base_layer = self.module_of_output.get(base_var)
        if base_layer is None:
            # If not tracked, just return the base variable
            return base_var

        # Create tensor operation with layers_of_tensors to track the source
        tensorop_param = {
            "tns_type": op_type,
            "layers_of_tensors": [base_layer],  # Track which layer this operates on
            "name": f"op_{self.tensor_op_counter}"
        }

        # Extract parameters based on operation type
        if op_type == 'repeat':
            # repeat takes multiple arguments (one per dimension)
            # e.g., .repeat(1, t, 1) means repeat 1x along dim0, t times along dim1, 1x along dim2
            repeat_counts = []
            for arg in call_node.args:
                repeat_counts.append(self.param_value(arg))
            tensorop_param["repeat_dim"] = repeat_counts
        else:
            # squeeze/unsqueeze take a single dimension parameter
            dim_value = None
            if len(call_node.args) > 0:
                dim_value = self.param_value(call_node.args[0])
            else:
                for kw in call_node.keywords:
                    if kw.arg == "dim":
                        dim_value = self.param_value(kw.value)
            tensorop_param["reduce_dim"] = dim_value

        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.buml_model.modules.append(tns_obj)
        self.tensor_op_counter += 1

        # Track the intermediate variable
        self.module_of_output[intermediate_var] = tensorop_param["name"]

        return intermediate_var

    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """
        # Handle both simple assignments (x = ...) and tuple assignments (x, _ = ...)
        if isinstance(node.targets[0], ast.Name):
            target_var = node.targets[0].id
        elif isinstance(node.targets[0], ast.Tuple):
            # Tuple assignment - use the first element (e.g., max_pool, _ = x.max())
            first_elem = node.targets[0].elts[0]
            target_var = first_elem.id if isinstance(first_elem, ast.Name) else "?"
        else:
            target_var = "?"

        # Handle .values attribute accessor (e.g., x.max(dim=1).values)
        call_node = node.value
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'values':
            # Unwrap to get the underlying Call node
            call_node = node.value.value

        if not isinstance(call_node, ast.Call):
            return

        op_type = call_node.func.attr
        op_args = call_node.args
        tensorop_param = None
        if op_type == "permute":
            tensorop_param = self.extract_tensorop_permute(op_args)
        elif op_type == "cat":
            tensorop_param = self.extract_tensorop_concatenate(node)
        elif op_type == "mul" or op_type == "matmul":
            layers_of_tensors = [self.module_of_output[op_args[0].id],
                                 self.module_of_output[op_args[1].id]]
            tensorop_param = {"tns_type": op_type+"tiply",
                              "layers_of_tensors": layers_of_tensors}
        elif op_type == "transpose":
            transpose_dim = [op_args[i].value for i in range(len(op_args))]
            # Track which variable this operation is called on (e.g., x.transpose())
            source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                source_layers = ['INPUT']
            else:
                source_layers = None
            tensorop_param = {"tns_type": op_type,
                              "transpose_dim": transpose_dim,
                              "layers_of_tensors": source_layers}
        elif op_type == "reshape" or op_type == "view":
            # Handle variable number of arguments (e.g., b*t, 32 or b, t, 16)
            # view() is the same as reshape() in PyTorch
            # For nested calls like x.view(x.size(0), -1), the nested call may have been
            # processed already by the general nested call handling, creating a temp variable
            reshape_dim = []
            for arg in op_args:
                if isinstance(arg, ast.Call) and hasattr(arg.func, 'attr') and arg.func.attr == 'size':
                    # This is a nested x.size(dim) call - extract it as a separate operation
                    if len(arg.args) > 0:
                        dim_idx = self.param_value(arg.args[0])
                        source_var = arg.func.value.id if isinstance(arg.func.value, ast.Name) else None
                        if source_var and source_var in self.module_of_output:
                            source_layers = [self.module_of_output[source_var]]
                        elif source_var:
                            source_layers = ['INPUT']
                        else:
                            source_layers = None

                        # Create the shape_dim tensor operation
                        op_name = f"op_{self.tensor_op_counter}"
                        shape_tensorop_param = {"tns_type": "shape_dim",
                                                "reduce_dim": dim_idx,
                                                "layers_of_tensors": source_layers,
                                                "name": op_name}
                        tns_obj = getattr(mm_classes, "TensorOp")(**shape_tensorop_param)
                        self.buml_model.add_tensor_op(tns_obj)
                        self.tensor_op_counter += 1
                        # Use the operation name in reshape_dim - it will be resolved to variable by generator
                        reshape_dim.append(op_name)
                    else:
                        reshape_dim.append(self.param_value(arg))
                elif isinstance(arg, ast.Name) and arg.id in self.module_of_output:
                    # This might be a temp variable from nested call handling
                    # Check if it refers to a tensor operation
                    layer_name = self.module_of_output[arg.id]
                    # Check if this is a tensor operation (would have _op suffix in modules_details)
                    if layer_name.startswith('op_'):
                        # This is a tensor operation - use its name directly
                        reshape_dim.append(layer_name)
                    else:
                        reshape_dim.append(self.param_value(arg))
                else:
                    reshape_dim.append(self.param_value(arg))

            # Track which variable this operation is called on (e.g., x.reshape())
            source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                # Variable not in module_of_output means it's the original network input
                source_layers = ['INPUT']
            else:
                source_layers = None
            tensorop_param = {"tns_type": "reshape",  # Normalize to "reshape"
                              "reshape_dim": reshape_dim}
            if source_layers:
                tensorop_param["layers_of_tensors"] = source_layers
        elif op_type == "size":
            # x.size(0) -> tf.shape(x)[0]
            # Get the dimension argument
            if len(op_args) > 0:
                dim_idx = self.param_value(op_args[0])
            else:
                # size() without args returns full shape - not commonly used in assignments
                print(f"Warning: size() without dimension argument - not fully supported")
                return

            # Track which variable this operation is called on
            source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                source_layers = ['INPUT']
            else:
                source_layers = None

            tensorop_param = {"tns_type": "shape_dim",
                              "reduce_dim": dim_idx,  # Use reduce_dim for consistency with existing code
                              "layers_of_tensors": source_layers}
        elif op_type == "mean":
            # Extract the dim parameter from keywords
            reduce_dim = None
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    reduce_dim = self.param_value(kw.value)
            if reduce_dim is None:
                print(f"Warning: mean operation without dim parameter - not supported")
                return
            # Track which variable this operation is called on
            # For method calls: x.mean() -> source_var = x
            # For module calls: torch.mean(x, ...) -> source_var = first argument
            if isinstance(call_node.func.value, ast.Name):
                # Check if it's a module call (torch.mean) or method call (x.mean)
                if call_node.func.value.id in ['torch', 'F', 'nn']:
                    # Module call: get first argument as source
                    source_var = op_args[0].id if len(op_args) > 0 and isinstance(op_args[0], ast.Name) else None
                else:
                    # Method call: func.value is the source
                    source_var = call_node.func.value.id
            else:
                source_var = None

            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                # Variable not in module_of_output means it's the original network input
                source_layers = ['INPUT']
            else:
                source_layers = None
            tensorop_param = {"tns_type": "mean",
                              "reduce_dim": reduce_dim,
                              "layers_of_tensors": source_layers}
        elif op_type == "max":
            # Extract the dim parameter from keywords
            reduce_dim = None
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    reduce_dim = self.param_value(kw.value)
            if reduce_dim is None:
                print(f"Warning: max operation without dim parameter - not supported")
                return
            # Track which variable this operation is called on
            # For method calls: x.max() -> source_var = x
            # For module calls: torch.max(x, ...) -> source_var = first argument
            if isinstance(call_node.func.value, ast.Name):
                if call_node.func.value.id in ['torch', 'F', 'nn']:
                    # Module call: get first argument as source
                    source_var = op_args[0].id if len(op_args) > 0 and isinstance(op_args[0], ast.Name) else None
                else:
                    # Method call: func.value is the source
                    source_var = call_node.func.value.id
            else:
                source_var = None

            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                # Variable not in module_of_output means it's the original network input
                source_layers = ['INPUT']
            else:
                source_layers = None
            tensorop_param = {"tns_type": "max",
                              "reduce_dim": reduce_dim,
                              "layers_of_tensors": source_layers}
        elif op_type == "amax":
            # torch.amax is same as torch.max with dim parameter
            reduce_dim = None
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    reduce_dim = self.param_value(kw.value)
            if reduce_dim is None:
                print(f"Warning: amax operation without dim parameter - not supported")
                return
            # Track which variable this operation is called on
            # For method calls: x.amax() -> source_var = x
            # For module calls: torch.amax(x, ...) -> source_var = first argument
            if isinstance(call_node.func.value, ast.Name):
                if call_node.func.value.id in ['torch', 'F', 'nn']:
                    # Module call: get first argument as source
                    source_var = op_args[0].id if len(op_args) > 0 and isinstance(op_args[0], ast.Name) else None
                else:
                    # Method call: func.value is the source
                    source_var = call_node.func.value.id
            else:
                source_var = None

            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                # Variable not in module_of_output means it's the original network input
                source_layers = ['INPUT']
            else:
                source_layers = None
            tensorop_param = {"tns_type": "max",
                              "reduce_dim": reduce_dim,
                              "layers_of_tensors": source_layers}
        elif op_type == "squeeze":
            # Extract the dim parameter - can be positional arg or keyword
            squeeze_dim = None
            if len(op_args) > 0:
                squeeze_dim = self.param_value(op_args[0])
            else:
                for kw in call_node.keywords:
                    if kw.arg == "dim":
                        squeeze_dim = self.param_value(kw.value)
            tensorop_param = {"tns_type": "squeeze",
                              "reduce_dim": squeeze_dim}
        elif op_type == "unsqueeze":
            # Extract the dim parameter - can be positional arg or keyword
            unsqueeze_dim = None
            if len(op_args) > 0:
                unsqueeze_dim = self.param_value(op_args[0])
            else:
                for kw in call_node.keywords:
                    if kw.arg == "dim":
                        unsqueeze_dim = self.param_value(kw.value)
            tensorop_param = {"tns_type": "unsqueeze",
                              "reduce_dim": unsqueeze_dim}
        elif op_type == "normalize":
            # F.normalize(input, p=2, dim=1) -> L2 normalization
            # First arg is the input tensor, then parameters
            norm_p = 2  # Default to L2
            norm_dim = None

            # Extract input tensor (first argument)
            if len(op_args) > 0 and isinstance(op_args[0], ast.Name):
                source_var = op_args[0].id
                # Check positional args: normalize(input, p, dim)
                if len(op_args) > 1:
                    norm_p = self.param_value(op_args[1])
                if len(op_args) > 2:
                    norm_dim = self.param_value(op_args[2])
            else:
                source_var = None

            # Check keyword args
            for kw in call_node.keywords:
                if kw.arg == "p":
                    norm_p = self.param_value(kw.value)
                elif kw.arg == "dim":
                    norm_dim = self.param_value(kw.value)

            # Track which variable this operation is called on
            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                source_layers = ['INPUT']
            else:
                source_layers = None

            tensorop_param = {"tns_type": "normalize",
                              "reduce_dim": norm_dim,
                              "layers_of_tensors": source_layers}
            # Note: norm_p is typically 2 for L2 normalization
            # TF's l2_normalize only supports L2, so we'll use that
        elif op_type == "flatten":
            # x.flatten(start_dim=1) -> create FlattenLayer
            # Extract start_dim and end_dim parameters
            start_dim = 1  # Default
            end_dim = -1   # Default

            if len(op_args) > 0:
                start_dim = self.param_value(op_args[0])
            if len(op_args) > 1:
                end_dim = self.param_value(op_args[1])

            for kw in call_node.keywords:
                if kw.arg == "start_dim":
                    start_dim = self.param_value(kw.value)
                elif kw.arg == "end_dim":
                    end_dim = self.param_value(kw.value)

            # Track source variable
            source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
            name_module_input = self.module_of_output.get(source_var, source_var)

            # Create FlattenLayer
            layer_name = f"flatten_{self.tensor_op_counter}"
            self.tensor_op_counter += 1

            flatten_params = {
                "name": layer_name,
                "start_dim": start_dim,
                "end_dim": end_dim,
                "name_module_input": name_module_input
            }

            flatten_layer = getattr(mm_classes, "FlattenLayer")(**flatten_params)
            self.buml_model.add_layer(flatten_layer)

            # Track output variable
            if isinstance(node.targets[0], ast.Name):
                output_var = node.targets[0].id
                self.module_of_output[output_var] = layer_name

            return  # Don't create a TensorOp
        elif op_type == "repeat":
            # x.repeat(1, t, 1) -> repeats tensor along each dimension
            # Extract repeat counts for each dimension, resolving variable references
            repeat_counts = []
            for arg in op_args:
                if isinstance(arg, ast.Name):
                    # Variable reference - resolve to operation name
                    var_name = arg.id
                    if var_name in self.module_of_output:
                        repeat_counts.append(self.module_of_output[var_name])
                    else:
                        repeat_counts.append(var_name)
                else:
                    repeat_counts.append(self.param_value(arg))

            # Track source variable
            source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
            if source_var and source_var in self.module_of_output:
                source_layers = [self.module_of_output[source_var]]
            elif source_var:
                source_layers = ['INPUT']
            else:
                source_layers = None

            tensorop_param = {"tns_type": "repeat",
                              "repeat_dim": repeat_counts,
                              "layers_of_tensors": source_layers}
        else:
            print(f"{op_type} is not recognized!")
            return

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1

            # Check if this tensorop's source variable was saved for residual connection
            # For method-style calls (x.transpose(), x.permute(), etc.), extract source variable
            if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
                source_var = call_node.func.value.id
                if hasattr(self, '_variables_saved_for_residual') and source_var in self._variables_saved_for_residual:
                    tns_obj.input_reused = True
                    self._variables_saved_for_residual.discard(source_var)

            # Track tensorop output so activations can detect it
            # Handle both simple and tuple assignments
            if isinstance(node.targets[0], ast.Name):
                output_var = node.targets[0].id
            elif isinstance(node.targets[0], ast.Tuple):
                first_elem = node.targets[0].elts[0]
                output_var = first_elem.id if isinstance(first_elem, ast.Name) else None
            else:
                output_var = None

            if output_var:
                self.module_of_output[output_var] = op_name
            # Note: inputs_outputs for tensorops handled differently - they use layers_of_tensors


    def extract_tensorop_concatenate(self, node):
        """
        It extracts the concatenate tensorop information.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            The tensorop parameters.
        """
        ops_args = node.value.args[0].elts
        tensorop_param = None
        if isinstance(ops_args[0], ast.Subscript):
            # Check if previous_assign exists and has expected structure before accessing
            if (self.previous_assign and
                hasattr(self.previous_assign, 'value') and
                isinstance(self.previous_assign.value, ast.Call) and
                hasattr(self.previous_assign.value, 'func') and
                hasattr(self.previous_assign.value.func, 'attr')):
                prev_lyr_name = self.previous_assign.value.func.attr
                lyr_obj = next((obj for obj in self.buml_model.layers if
                                obj.name == prev_lyr_name), None)
                if lyr_obj:
                    lyr_obj.return_type =  "hidden"

        # Extract variable names - handle both simple names and method calls
        def extract_var_name_and_create_op(arg):
            if isinstance(arg, ast.Name):
                return arg.id
            elif isinstance(arg, ast.Call):
                # Check if it's a layer call (self.layer_name(...))
                if (isinstance(arg.func, ast.Attribute) and
                    isinstance(arg.func.value, ast.Name) and
                    arg.func.value.id == 'self'):
                    # Layer call - need to handle layer reuse properly
                    layer_name = arg.func.attr

                    # Find the layer in the BUML model's layers list
                    layer_obj = next((lyr for lyr in self.buml_model.layers if lyr.name == layer_name), None)

                    if layer_obj:
                        # Check if this is a reuse (layer already in modules)
                        if layer_obj in self.buml_model.modules:
                            # Layer reuse - increment reuse count
                            self.layer_reuse_count[layer_name] = self.layer_reuse_count.get(layer_name, 0) + 1
                            use_count = self.layer_reuse_count[layer_name]

                            # Create a synthetic layer reference for this specific use
                            import copy
                            reuse_layer = copy.copy(layer_obj)
                            reuse_layer.name = f"{layer_name}_use_{use_count}"
                            reuse_layer.input_reused = True  # Mark that input is shared with other layers

                            # Track the input for this use
                            if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
                                input_var = arg.args[0].id
                                if input_var in self.module_of_output:
                                    reuse_layer.name_module_input = self.module_of_output[input_var]

                            self.buml_model.modules.append(reuse_layer)

                            # Create temp variable for this use
                            temp_name = f"_nested_temp_{self.tensor_op_counter}"
                            self.tensor_op_counter += 1
                            self.module_of_output[temp_name] = reuse_layer.name
                            return temp_name
                        else:
                            # First use of this layer - also mark as input_reused if it will be reused
                            # Track the input for this layer
                            if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
                                input_var = arg.args[0].id
                                if input_var in self.module_of_output:
                                    layer_obj.name_module_input = self.module_of_output[input_var]
                                    layer_obj.input_reused = True  # Mark that input is shared

                            self.buml_model.modules.append(layer_obj)

                            # Create temp variable for the layer's output
                            temp_name = f"_nested_temp_{self.tensor_op_counter}"
                            self.tensor_op_counter += 1
                            self.module_of_output[temp_name] = layer_name
                            return temp_name
                    else:
                        print(f"Warning: Layer '{layer_name}' not found in BUML model")
                        return None
                else:
                    # Method call like x.squeeze(1) - create the operation and return result var
                    result = self.extract_inline_tensorop(arg, node)
                    if result is None:
                        print(f"Warning: extract_inline_tensorop returned None for {ast.unparse(arg)}")
                        return None
                    return result
            elif isinstance(arg, ast.Subscript):
                # Subscript like h_n[-2] - create subscript TensorOp
                temp_name = f"_subscript_temp_{self.tensor_op_counter}"
                self.tensor_op_counter += 1
                self.handle_subscript_operation(arg, temp_name, node)
                return temp_name
            else:
                return None

        # Extract all variables (not just 2)
        variables = []
        for arg in ops_args:
            var = extract_var_name_and_create_op(arg)
            if var is None:
                print(f"Warning: Cannot extract variable from concat argument")
                return None
            variables.append(var)


        # Resolve aliases for all variables
        actual_vars = []
        for var in variables:
            actual_var = var if var in self.module_of_output else self.variable_aliases.get(var, var)
            actual_vars.append(actual_var)

        layers_of_tensors = [self.module_of_output[actual_var] for actual_var in actual_vars]
        cat_dim = self.param_value(node.value.keywords[0].value)

        # Check if this is torch.cat([h[-2], h[-1]], dim=1) pattern for bidirectional RNN
        # Handle both inline subscripts and variables from subscripts
        if (len(ops_args) == 2 and len(set(layers_of_tensors)) == 1):  # Concatenating 2 things from same source
            source_layer_name = layers_of_tensors[0]
            source_layer = next((obj for obj in self.buml_model.layers if obj.name == source_layer_name), None)

            # Check if source is a bidirectional RNN returning hidden states
            if (source_layer and
                hasattr(source_layer, 'bidirectional') and source_layer.bidirectional and
                hasattr(source_layer, 'return_type') and source_layer.return_type == 'hidden'):

                # Case 1: Inline subscripts like torch.cat([h[-2], h[-1]])
                if all(isinstance(arg, ast.Subscript) for arg in ops_args):
                    # Extract subscript indices
                    indices = []
                    for arg in ops_args:
                        if isinstance(arg.slice, ast.UnaryOp) and isinstance(arg.slice.op, ast.USub):
                            indices.append(-arg.slice.operand.value)
                        elif isinstance(arg.slice, ast.Constant):
                            indices.append(arg.slice.value)
                        else:
                            indices = None
                            break

                    # Check if indices are -2 and -1 (in any order)
                    if indices and set(indices) == {-2, -1}:
                        output_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                        if output_var:
                            self.module_of_output[output_var] = source_layer_name
                        return None

                # Case 2: Variables from subscripts like torch.cat([h_forward, h_backward])
                # If both variables point to the same bidirectional RNN, this is the forward/backward concat
                # which is already handled by bidirectional unpacking
                elif all(isinstance(arg, ast.Name) for arg in ops_args):
                    # This is concatenating variables that came from the bidirectional RNN
                    # Skip creating the TensorOp since bidirectional unpacking handles it
                    output_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                    if output_var:
                        self.module_of_output[output_var] = source_layer_name
                    return None

        # Determine if each variable is output or hidden for RNNs with return_type="both"
        var_types = []
        for lyr_name, actual_var in zip(layers_of_tensors, actual_vars):
            if lyr_name in self.rnn_hidden_vars and actual_var == self.rnn_hidden_vars[lyr_name]:
                var_types.append("hidden")
            elif lyr_name in self.rnn_output_vars and actual_var == self.rnn_output_vars[lyr_name]:
                var_types.append("output")
            else:
                var_types.append("output")  # Default

        tensorop_param = {"tns_type": "concatenate",
                          "layers_of_tensors": layers_of_tensors,
                          "concatenate_dim": cat_dim,
                          # Store which component (output/hidden) each refers to
                          "actual_vars": var_types}

        return tensorop_param


    def extract_tensorop_permute(self, ops_args):
        """
        It extracts the permute tensorop information.

        Returns:
            The tensorop parameters.
        """
        tensorop_param = None
        modules = self.buml_model.modules
        prev_module = modules[-1] if modules else None
        if isinstance(prev_module, Layer):
            lyr_type = prev_module.__class__.__name__
            if lyr_type in cnn_layers:
                prev_module.permute_out = True
            else:
                # Not a CNN layer, create tensorop
                permute_dim = []
                for arg in ops_args:
                    if isinstance(arg, ast.Constant):
                        permute_dim.append(arg.value)
                    elif isinstance(arg, ast.Num):  # Python < 3.8
                        permute_dim.append(arg.n)
                tensorop_param = {"tns_type": "permute",
                                  "permute_dim": permute_dim}
        else:
            permute_dim = []
            for arg in ops_args:
                if isinstance(arg, ast.Constant):
                    permute_dim.append(arg.value)
                elif isinstance(arg, ast.Num):  # Python < 3.8
                    permute_dim.append(arg.n)
            tensorop_param = {"tns_type": "permute",
                              "permute_dim": permute_dim}
        return tensorop_param


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
        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "datasets":
                self.get_path_data(node)
            elif node.value.func.value.id == "transforms":
                self.get_images_attr(node)
            elif "Loss" in node.value.func.attr:
                loss = node.value.func.attr
                cnf = self.data_config["config"]
                cnf["loss_function"] = loss_func_mapping[loss]

        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.attr == "optim":
                self.get_params_from_optimizer(node)
            elif node.value.func.attr == "DataLoader":
                batch_size = next(
                    (k.value.value for k in node.value.keywords
                     if k.arg == "batch_size"),
                    None
                )
                if batch_size is not None:
                    self.data_config["config"]["batch_size"] = batch_size

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
                metrics = [elt.value for elt in node.value.elts]
                self.data_config["config"]["metrics"] = metrics
        elif isinstance(node.value, ast.Constant):
            self.handle_outer_constant_assignment(node)
        elif isinstance(node.value, ast.Tuple):
            self.handle_outer_tuple_assignment(node)
        else:
            self.unprocessed_nodes.append(node)




def transform_layer(lyr_type: str, lyr_params: dict,
                    lyr_name: str | None = None):
    """
    It transforms layers and their params from PyTorch to BUML.

    Parameters:
        lyr_type (str): The type of the layer (PyTorch).
        lyr_params (dict): A dictionnary storing the layer parameters and
            their values.
        lyr_name (str | None): The name of the layer.

    Returns:
        The type of the layer and its parameters in BUML.
    """

    process_positional_params(lyr_type, lyr_params, pos_params)

    param_to_list(lyr_type, lyr_params, int2list_params,
                               lyrs_of_int2list_params)

    lyr_params = process_params(lyr_type, lyr_params)
    lyr_params["name"] = lyr_name

    # Extract dimension from spatial dropout layers (Dropout1d, Dropout2d, Dropout3d)
    if lyr_type in ["Dropout1d", "Dropout2d", "Dropout3d"]:
        lyr_params["dimension"] = lyr_type[-2]  # Extract '1', '2', or '3'

    lyr_type = layers_mapping[lyr_type]
    return lyr_type, lyr_params



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

    for param in lyr_params:
        if param in params_mapping:
            updated_lyr_params[params_mapping[param]] = lyr_params[param]
        elif param in ["actv_func", "name", "inplace"]:
            # Ignored parameters: actv_func/name handled separately, inplace not needed in TF
            if param in ["actv_func", "name"]:
                updated_lyr_params[param] = lyr_params[param]
        else:
            print(f"parameter {param} of layer {lyr_type} is not found!")


    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params

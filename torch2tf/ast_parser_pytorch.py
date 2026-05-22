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
                    lyr_type, lyr_params = transform_layer(
                        lyr_type, lyr_params, module_name
                    )
                    print("nadia lyr_params", lyr_params)
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
                    subnn.layers[-1].actv_func = actv_fun_mapping[lyr_type]
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
                # Subscript detected: process it first
                subscript_node = node.value.args[0]
                temp_name = f"_subscript_temp_{self.tensor_op_counter}"
                self.tensor_op_counter += 1
                temp_target = ast.Name(id=temp_name, ctx=ast.Store())

                # Create synthetic assignment for subscript
                subscript_assign = ast.Assign(targets=[temp_target], value=subscript_node)
                subscript_assign.lineno = node.lineno
                subscript_assign.col_offset = node.col_offset
                self.handle_forward_slicing(subscript_assign)
                self.previous_assign = subscript_assign

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

        # Handle functional API (F.layer or torch.nn.functional.layer)
        if caller_id == "F" or caller_id == "functional":
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
                    actv = self.activation_functions[module_name]
                    actv_lyr = mm_classes.GeneralLayer(
                        name=module_name,
                        actv_func=actv_fun_mapping[actv]
                    )
                    self.buml_model.modules.append(actv_lyr)

            elif not is_subnn_obj:
                self.is_permute_before_cnn(module_name)

            if module_name not in self.activation_functions:
                module_obj = next((obj for obj in self.buml_model.layers if
                                   obj.name == module_name), None)
                if not module_obj:
                    subnns = self.buml_model.sub_nns
                    module_obj = next((obj for obj in subnns if
                                       obj.name == module_name), None)

                self.buml_model.modules.append(module_obj)

        else:
            #tensorops
            self.extract_tensorop(node)

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
        module_name = node.value.func.attr

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
            temp_target = ast.Name(id=temp_name, ctx=ast.Store())
            subscript_assign = ast.Assign(targets=[temp_target], value=binop.left)
            subscript_assign.lineno = node.lineno
            subscript_assign.col_offset = node.col_offset
            self.handle_forward_slicing(subscript_assign)
            left_var = temp_name
        elif isinstance(binop.left, (ast.Constant, ast.Num)):
            # Handle constant operand
            left_var = binop.left.value if isinstance(binop.left, ast.Constant) else binop.left.n
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
            temp_target = ast.Name(id=temp_name, ctx=ast.Store())
            subscript_assign = ast.Assign(targets=[temp_target], value=binop.right)
            subscript_assign.lineno = node.lineno
            subscript_assign.col_offset = node.col_offset
            self.handle_forward_slicing(subscript_assign)
            right_var = temp_name
        elif isinstance(binop.right, (ast.Constant, ast.Num)):
            # Handle constant operand
            right_var = binop.right.value if isinstance(binop.right, ast.Constant) else binop.right.n
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
        self.buml_model.modules.append(tns_obj)
        self.tensor_op_counter += 1

        # Track the output variable
        output_var = node.targets[0].id
        self.module_of_output[output_var] = tensorop_param["name"]

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

        # Look up which RNN module produced this variable
        if subscripted_var in self.module_of_output:
            prev_module_name = self.module_of_output[subscripted_var]
        else:
            # Fallback: assume previous_assign was the RNN call
            if hasattr(self.previous_assign.value, 'func') and hasattr(self.previous_assign.value.func, 'attr'):
                prev_module_name = self.previous_assign.value.func.attr
            else:
                print(f"Warning: Cannot determine RNN module for subscript on '{subscripted_var}'")
                self.previous_assign = node
                return

        lyr_obj = next((obj for obj in self.buml_model.layers if
                        obj.name == prev_module_name), None)

        if not lyr_obj:
            print(f"Warning: Layer '{prev_module_name}' not found for slicing operation")
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
            # Pattern: out[:, -1, :] means extracting last timestep
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

        # Only handle squeeze/unsqueeze for now
        if op_type not in ['squeeze', 'unsqueeze']:
            return base_var

        # Create intermediate variable name
        intermediate_var = f"_inline_{op_type}_{self.tensor_op_counter}"

        # Extract dimension parameter
        dim_value = None
        if len(call_node.args) > 0:
            dim_value = self.param_value(call_node.args[0])
        else:
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    dim_value = self.param_value(kw.value)

        # Get the layer/module that produced the base variable
        base_layer = self.module_of_output.get(base_var)
        if base_layer is None:
            # If not tracked, just return the base variable
            return base_var

        # Create tensor operation with layers_of_tensors to track the source
        tensorop_param = {
            "tns_type": op_type,
            "reduce_dim": dim_value,
            "layers_of_tensors": [base_layer],  # Track which layer this operates on
            "name": f"op_{self.tensor_op_counter}"
        }

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
        op_type = node.value.func.attr
        op_args = node.value.args
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
            tensorop_param = {"tns_type": op_type,
                              "transpose_dim": transpose_dim}
        elif op_type == "reshape":
            reshape_dim = [op_args[0].value, op_args[1].value]
            tensorop_param = {"tns_type": op_type,
                              "reshape_dim": reshape_dim}
        elif op_type == "mean":
            # Extract the dim parameter from keywords
            reduce_dim = None
            for kw in node.value.keywords:
                if kw.arg == "dim":
                    reduce_dim = self.param_value(kw.value)
            if reduce_dim is None:
                print(f"Warning: mean operation without dim parameter - not supported")
                return
            tensorop_param = {"tns_type": "mean",
                              "reduce_dim": reduce_dim}
        elif op_type == "squeeze":
            # Extract the dim parameter - can be positional arg or keyword
            squeeze_dim = None
            if len(op_args) > 0:
                squeeze_dim = self.param_value(op_args[0])
            else:
                for kw in node.value.keywords:
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
                for kw in node.value.keywords:
                    if kw.arg == "dim":
                        unsqueeze_dim = self.param_value(kw.value)
            tensorop_param = {"tns_type": "unsqueeze",
                              "reduce_dim": unsqueeze_dim}
        else:
            print(f"{op_type} is not recognized!")
            return

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1

            # Track tensorop output so activations can detect it
            output_var = node.targets[0].id
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
            prev_lyr_name = self.previous_assign.value.func.attr
            lyr_obj = next((obj for obj in self.buml_model.layers if
                            obj.name == prev_lyr_name), None)
            lyr_obj.return_type =  "hidden"
        else:
            # Extract variable names - handle both simple names and method calls
            def extract_var_name_and_create_op(arg):
                if isinstance(arg, ast.Name):
                    return arg.id
                elif isinstance(arg, ast.Call):
                    # Method call like x.squeeze(1) - create the operation and return result var
                    return self.extract_inline_tensorop(arg, node)
                else:
                    return None

            var1 = extract_var_name_and_create_op(ops_args[0])
            var2 = extract_var_name_and_create_op(ops_args[1])

            if var1 is None or var2 is None:
                # Can't extract variable names - skip this concat
                print(f"Warning: Cannot extract variables from concat operation")
                return None

            # Resolve aliases
            actual_var1 = self.variable_aliases.get(var1, var1)
            actual_var2 = self.variable_aliases.get(var2, var2)

            layers_of_tensors = [self.module_of_output[actual_var1],
                                 self.module_of_output[actual_var2]]
            cat_dim = self.param_value(node.value.keywords[0].value)

            # Determine if each variable is output or hidden for RNNs with return_type="both"
            var_types = []
            for lyr_name, actual_var in zip(layers_of_tensors, [actual_var1, actual_var2]):
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
        elif param == "actv_func":
            updated_lyr_params[param] = lyr_params[param]
        else:
            print(f"parameter {param} of layer {lyr_type} is not found!")


    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params

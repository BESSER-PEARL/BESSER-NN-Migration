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

    # Temporary variable name patterns
    TEMP_SUBSCRIPT_OP = "_subscript_op_{}"
    TEMP_CHAIN = "_chain_temp_{}_{}"
    TEMP_NESTED_IN_CHAIN = "_nested_in_chain_{}"
    TEMP_SUBSCRIPT = "_subscript_temp_{}"
    TEMP_NESTED = "_nested_temp_{}"
    TEMP_MLRNN = "_mlrnn_{}_{}"
    TEMP_BINOP = "_binop_temp_{}"
    TEMP_INLINE = "_inline_{}_{}"

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
        # Track the output variable of the immediately previous layer
        self.prev_layer_output = None  # str: variable name
        # Note: migration_warnings inherited from base ASTParser class
        # Performance optimization: O(1) lookup dicts instead of O(n) searches
        self.layer_by_name = {}  # {layer_name: layer_obj}
        self.module_by_name = {}  # {module_name: module_obj}

        # Operation handler mapping for dispatch pattern
        self._op_handlers = {
            "permute": lambda cn, n, args: self.extract_tensorop_permute(args),
            "cat": lambda cn, n, args: self.extract_tensorop_concatenate(n),
            "mul": lambda cn, n, args: self._extract_op_multiply(cn, n, "mul"),
            "matmul": lambda cn, n, args: self._extract_op_multiply(cn, n, "matmul"),
            "transpose": lambda cn, n, args: self._extract_op_transpose(cn, args),
            "reshape": lambda cn, n, args: self._extract_op_reshape(cn, n, args),
            "view": lambda cn, n, args: self._extract_op_reshape(cn, n, args),
            "size": lambda cn, n, args: self._extract_op_size(cn, n, args),
            "mean": lambda cn, n, args: self._extract_op_mean(cn, n, args),
            "max": lambda cn, n, args: self._extract_op_max(cn, n, args),
            "amax": lambda cn, n, args: self._extract_op_amax(cn, n, args),
            "squeeze": lambda cn, n, args: self._extract_op_squeeze(cn, args),
            "unsqueeze": lambda cn, n, args: self._extract_op_unsqueeze(cn, args),
            "normalize": lambda cn, n, args: self._extract_op_normalize(cn, args),
            "flatten": lambda cn, n, args: self._extract_op_flatten(cn, n, args),
            "repeat": lambda cn, n, args: self._extract_op_repeat(cn, args),
        }

    def _add_layer_with_tracking(self, layer_obj):
        """Add layer to model and update lookup dict for O(1) access."""
        self.buml_model.add_layer(layer_obj)
        if hasattr(layer_obj, 'name'):
            self.layer_by_name[layer_obj.name] = layer_obj
            self.module_by_name[layer_obj.name] = layer_obj

    def _add_module_with_tracking(self, module_obj):
        """Add module to model and update lookup dict for O(1) access."""
        self.buml_model.modules.append(module_obj)
        if hasattr(module_obj, 'name'):
            self.module_by_name[module_obj.name] = module_obj

    def _get_layer_by_name(self, layer_name):
        """Get layer by name using O(1) lookup, fallback to linear search if not in dict."""
        if layer_name in self.layer_by_name:
            return self.layer_by_name[layer_name]
        # Fallback to linear search and update dict (for layers added before tracking)
        layer_obj = next((obj for obj in self.buml_model.layers if obj.name == layer_name), None)
        if layer_obj:
            self.layer_by_name[layer_name] = layer_obj
        return layer_obj

    def _get_module_by_name(self, module_name):
        """Get module by name using O(1) lookup, fallback to linear search if not in dict."""
        if module_name in self.module_by_name:
            return self.module_by_name[module_name]
        # Fallback to linear search and update dict (for modules added before tracking)
        module_obj = next((obj for obj in self.buml_model.modules if obj.name == module_name), None)
        if module_obj:
            self.module_by_name[module_name] = module_obj
        return module_obj

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
        op_name = self.TEMP_SUBSCRIPT_OP.format(self.tensor_op_counter)
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
            src_layer = self._get_layer_by_name(src_module)
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

    def _create_rnn_layer_params(self, lyr_params, i, num_layers, module_name, original_return_type):
        """Create parameters for individual RNN layer in multi-layer stack."""
        layer_params = lyr_params.copy()
        layer_params.pop('num_layers', None)
        layer_params['name'] = f"{module_name}_layer_{i}"

        if i > 0:
            layer_params['input_size'] = lyr_params['hidden_size']

        if i < num_layers - 1:
            layer_params['return_type'] = 'full'
        else:
            layer_params['return_type'] = original_return_type

        layer_params['positional_params'] = []
        return layer_params

    def _add_rnn_layer(self, lyr_type, layer_params):
        """Transform and add a single RNN layer to BUML model."""
        try:
            buml_lyr_type, buml_params = transform_layer(lyr_type, layer_params, layer_params['name'])
            buml_layer = getattr(mm_classes, buml_lyr_type)(**buml_params)
            self.buml_model.add_layer(buml_layer)
        except ValueError as e:
            self.migration_warnings.append(f"Layer '{layer_params['name']}': {str(e)}")

    def _create_multi_layer_rnn(self, lyr_type, lyr_params, module_name):
        """Create multiple BUML layers for stacked RNN."""
        process_positional_params(lyr_type, lyr_params, pos_params)
        num_layers = lyr_params.get('num_layers', 1)
        original_return_type = lyr_params.get('return_type', 'full')
        layer_names = []

        for i in range(num_layers):
            layer_params = self._create_rnn_layer_params(lyr_params, i, num_layers, module_name, original_return_type)
            layer_names.append(layer_params['name'])
            self._add_rnn_layer(lyr_type, layer_params)

        self.multi_layer_rnns[module_name] = layer_names

    def _create_single_layer(self, lyr_type, lyr_params, module_name):
        """Create single BUML layer."""
        lyr_params.pop('num_layers', None)
        try:
            lyr_type, lyr_params = transform_layer(lyr_type, lyr_params, module_name)
            buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
            self.buml_model.add_layer(buml_layer)
        except ValueError as e:
            self.migration_warnings.append(f"Layer '{module_name}': {str(e)}")

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
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            module_type = node.value.func.attr
            if module_type == "Sequential":
                self.handle_sequential_layers(node, module_name)
            else:
                lyr_type, lyr_params = self.extract_layer(node.value)
                if lyr_type not in actv_fun_mapping:
                    num_layers = lyr_params.get('num_layers', 1)
                    is_rnn = lyr_type in ['RNN', 'LSTM', 'GRU']

                    if is_rnn and num_layers > 1:
                        self._create_multi_layer_rnn(lyr_type, lyr_params, module_name)
                    else:
                        self._create_single_layer(lyr_type, lyr_params, module_name)
                else:
                    self.activation_functions[module_name] = lyr_type

        self.buml_model.modules.clear()


    def _check_prev_layer_supports_activation(self, subnn):
        """Check if previous layer in sequential supports activation functions."""
        if not subnn.layers:
            return False

        prev_layer = subnn.layers[-1]
        prev_layer_class = prev_layer.__class__.__name__
        prev_layer_parent = prev_layer.__class__.mro()[1].__name__

        unsupported = prev_layer_parent in ["NormalizationLayer", "LayerModifier"] or \
                     prev_layer_class in ["EmbeddingLayer", "PoolingLayer", "FlattenLayer"]

        return not unsupported

    def _handle_activation_layer(self, subnn, lyr_type, layer_id):
        """Handle activation function in sequential model."""
        actv_func = actv_fun_mapping.get(lyr_type)
        if actv_func is None:
            self.migration_warnings.append(
                f"Unsupported activation function '{lyr_type}'. This activation will be skipped in the migration."
            )
            return layer_id

        if self._check_prev_layer_supports_activation(subnn):
            subnn.layers[-1].actv_func = actv_func
            return layer_id
        else:
            actv_params = {"name": f"layer_{layer_id}", "actv_func": actv_func}
            subnn_layer = getattr(mm_classes, "GeneralLayer")(**actv_params)
            subnn.add_layer(subnn_layer)
            return layer_id + 1

    def _handle_permute_layer(self, subnn):
        """Handle Permute operation in sequential model."""
        last_lyr = subnn.layers[-1] if subnn.layers else None

        if last_lyr is not None:
            last_lyr_type = subnn.layers[-1].__class__.__name__
            if last_lyr_type in cnn_layers:
                subnn.layers[-1].permute_out = True
                return False
        return True

    def _handle_regular_sequential_layer(self, subnn, lyr_type, lyr_params, layer_id, permute):
        """Handle regular layer in sequential model."""
        try:
            lyr_type, lyr_params = transform_layer(lyr_type, lyr_params)
            lyr_params["name"] = f"layer_{layer_id}"
            if permute:
                lyr_params["permute_in"] = True

            subnn_layer = getattr(mm_classes, lyr_type)(**lyr_params)
            subnn.add_layer(subnn_layer)
            return layer_id + 1, False
        except ValueError as e:
            self.migration_warnings.append(f"Sequential layer {layer_id}: {str(e)}")
            return layer_id, permute

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

        for elt in node.value.args:
            if isinstance(elt, ast.Call):
                lyr_type, lyr_params = self.extract_layer(elt)

                if lyr_type in actv_fun_mapping:
                    layer_id = self._handle_activation_layer(subnn, lyr_type, layer_id)
                elif lyr_type == "Permute":
                    permute = self._handle_permute_layer(subnn)
                else:
                    layer_id, permute = self._handle_regular_sequential_layer(subnn, lyr_type, lyr_params, layer_id, permute)

            elif isinstance(elt, ast.Name):
                subnn_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == elt.id), None)
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

    def _process_nested_call_in_chain(self, call, node):
        """Extract and process nested call within a chain element."""
        if not (isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Call)):
            return

        inner_call = call.args[0]
        inner_temp = self.TEMP_NESTED_IN_CHAIN.format(self.tensor_op_counter)
        self.tensor_op_counter += 1
        inner_target = ast.Name(id=inner_temp, ctx=ast.Store())
        inner_node = ast.Assign(targets=[inner_target], value=inner_call)
        inner_node.lineno = node.lineno
        inner_node.col_offset = node.col_offset

        self.process_single_call(inner_node)
        self.previous_assign = inner_node
        call.args[0] = ast.Name(id=inner_temp, ctx=ast.Load())

    def _link_chain_calls(self, chain, i, temp_name):
        """Link current chain call to the next via temporary variable."""
        if i + 1 >= len(chain):
            return

        next_call = chain[i + 1]
        next_func_value = next_call.func.value

        if isinstance(next_func_value, ast.Name):
            # Functional call like F.relu(x) where x is in args[0]
            if next_call.args:
                next_call.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
        elif isinstance(next_func_value, ast.Call):
            # Method call like result.permute(...) where result is func.value
            next_call.func.value = ast.Name(id=temp_name, ctx=ast.Load())

    def _is_nested_activation(self, call):
        """Check if this call is an activation on a nested-in-chain temp variable."""
        if not (isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Name)):
            return False
        return call.args[0].id.startswith(self.TEMP_NESTED_IN_CHAIN.split('{')[0])

    def _process_chained_calls(self, node):
        """Process chained method calls by decomposing and handling each element."""
        chain = self.decompose_chained_call(node)

        for i, call in enumerate(chain):
            is_last = (i == len(chain) - 1)

            # Create synthetic node for this chain element
            if is_last:
                synthetic_node = ast.Assign(targets=node.targets, value=call)
            else:
                temp_name = self.TEMP_CHAIN.format(self.tensor_op_counter, i)
                temp_target = ast.Name(id=temp_name, ctx=ast.Store())
                synthetic_node = ast.Assign(targets=[temp_target], value=call)
                self._link_chain_calls(chain, i, temp_name)

            # Handle nested calls within this chain element
            self._process_nested_call_in_chain(call, node)

            synthetic_node.lineno = node.lineno
            synthetic_node.col_offset = node.col_offset

            # Process with nested activation flag if needed
            is_nested_activ = self._is_nested_activation(call)
            if is_nested_activ:
                self.is_processing_nested_outer = True

            self.process_single_call(synthetic_node)
            self.previous_assign = synthetic_node

            if is_nested_activ:
                self.is_processing_nested_outer = False

    def handle_forward_simple_call(self, node: ast.Assign):
        """
        Processes forward method assignments including chained and nested calls.
        """
        # Handle chained calls (x.method1().method2())
        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'value'):
            if isinstance(node.value.func.value, ast.Call):
                self._process_chained_calls(node)
                return

        # Handle subscript arguments and nested calls, then process
        self._handle_subscript_argument(node)
        is_nested = self._handle_nested_call(node)

        if is_nested:
            self.is_processing_nested_outer = True

        self.process_single_call(node)
        self.previous_assign = node

        if is_nested:
            self.is_processing_nested_outer = False

    def _handle_subscript_argument(self, node):
        """Process subscript arguments in calls (e.g., self.fc(h[-1]))."""
        if not (isinstance(node.value, ast.Call) and node.value.args):
            return
        if not isinstance(node.value.args[0], ast.Subscript):
            return

        subscript_node = node.value.args[0]
        temp_name = self.TEMP_SUBSCRIPT.format(self.tensor_op_counter)
        self.tensor_op_counter += 1

        self.handle_subscript_operation(subscript_node, temp_name, node)
        node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())

    def _handle_nested_call(self, node):
        """Process nested calls (e.g., F.relu(self.conv(x))). Returns True if nested."""
        if not (isinstance(node.value, ast.Call) and node.value.args):
            return False
        if not isinstance(node.value.args[0], ast.Call):
            return False

        inner_call = node.value.args[0]
        temp_name = self.TEMP_NESTED.format(self.tensor_op_counter)
        self.tensor_op_counter += 1
        temp_target = ast.Name(id=temp_name, ctx=ast.Store())

        inner_node = ast.Assign(targets=[temp_target], value=inner_call)
        inner_node.lineno = node.lineno
        inner_node.col_offset = node.col_offset

        self.visit_Assign(inner_node)
        self.previous_assign = inner_node

        node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
        return True

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

        # Dispatch based on caller type
        if caller_id in ("F", "functional", "torch"):
            self._process_functional_api(node)
        elif caller_id == "self":
            module_name = node.value.func.attr
            self._process_module_api(node, module_name)
        else:
            self.extract_tensorop(node)

    def _process_last_layer_tuple(self, node, layer_name, current_input):
        """Process last layer with tuple assignment in multi-layer RNN."""
        self._extract_tuple_target_vars(node, layer_name)
        rnn_out = self._determine_rnn_return_type(node, layer_name)
        self.inputs_outputs[layer_name] = [current_input, rnn_out]

        lyr_obj = self._get_layer_by_name(layer_name)
        if lyr_obj:
            self.buml_model.modules.append(lyr_obj)

    def _process_last_layer_simple(self, node, layer_name, current_input):
        """Process last layer with simple assignment in multi-layer RNN."""
        output_var = node.targets[0].id
        self.inputs_outputs[layer_name] = [current_input, output_var]
        self.module_of_output[output_var] = layer_name

        lyr_obj = self._get_layer_by_name(layer_name)
        if lyr_obj:
            self.buml_model.modules.append(lyr_obj)

    def _process_intermediate_layer(self, module_name, layer_name, current_input, i):
        """Process intermediate layer in multi-layer RNN."""
        temp_var = self.TEMP_MLRNN.format(module_name, i)
        self.inputs_outputs[layer_name] = [current_input, temp_var]
        self.module_of_output[temp_var] = layer_name

        module_obj = self._get_layer_by_name(layer_name)
        if module_obj:
            self.buml_model.modules.append(module_obj)

        return temp_var

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
        current_input = self._extract_rnn_input_arg(node)

        for i, layer_name in enumerate(layer_names):
            is_last_layer = (i == len(layer_names) - 1)

            if is_last_layer:
                node.value.func.attr = layer_name
                if i > 0:
                    node.value.args[0] = ast.Name(id=current_input, ctx=ast.Load())

                if is_tuple:
                    self._process_last_layer_tuple(node, layer_name, current_input)
                else:
                    self._process_last_layer_simple(node, layer_name, current_input)
            else:
                current_input = self._process_intermediate_layer(module_name, layer_name, current_input, i)

    def _extract_tuple_target_vars(self, node, module_name):
        """Extract and track output/hidden variables from tuple assignment targets."""
        var1 = node.targets[0].elts[0].id if not isinstance(node.targets[0].elts[0], ast.Tuple) else None
        var2 = node.targets[0].elts[1].id if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0].id

        if var1 and var1 != "_":
            self.rnn_output_vars[module_name] = var1
            self.module_of_output[var1] = module_name
        if var2 and var2 != "_":
            self.rnn_hidden_vars[module_name] = var2
            self.module_of_output[var2] = module_name

    def _determine_rnn_return_type(self, node, module_name):
        """Determine RNN return type and main output variable based on underscore pattern."""
        first_elem = node.targets[0].elts[0]
        second_elem = node.targets[0].elts[1] if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0]

        first_is_underscore = isinstance(first_elem, ast.Name) and first_elem.id == "_"
        second_is_underscore = isinstance(second_elem, ast.Name) and second_elem.id == "_"

        lyr_obj = next((obj for obj in self.buml_model.layers if obj.name == module_name), None)

        if first_is_underscore and not second_is_underscore:
            rnn_out = node.targets[0].elts[1].elts[0].id if isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].id
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

        return rnn_out

    def _extract_rnn_input_arg(self, node):
        """Extract input argument from RNN call, handling both variables and inline operations."""
        input_arg = node.value.args[0]
        if isinstance(input_arg, ast.Name):
            return input_arg.id
        elif isinstance(input_arg, ast.Call):
            return self.extract_inline_tensorop(input_arg, node)
        else:
            return "x"

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
            if caller_id != "self":
                self.extract_tensorop(node)
                return

        module_name = node.value.func.attr

        # Check if this is a multi-layer RNN that needs to be expanded
        if module_name in self.multi_layer_rnns:
            self.expand_multi_layer_rnn_call(node, module_name, is_tuple=True)
            return

        # Extract and track tuple target variables
        self._extract_tuple_target_vars(node, module_name)

        # Determine return type and main output variable
        rnn_out = self._determine_rnn_return_type(node, module_name)

        # Extract input argument
        rnn_in = self._extract_rnn_input_arg(node)

        # Update tracking structures
        self.inputs_outputs[module_name] = [rnn_in, rnn_out]
        module_obj = next((obj for obj in self.buml_model.layers if obj.name == module_name), None)
        if not module_obj:
            module_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)
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

    def _extract_binop_operand(self, operand, node, side):
        """Extract variable from binary operation operand (left or right)."""
        if isinstance(operand, ast.Name):
            return operand.id
        elif isinstance(operand, ast.Call):
            return self.extract_inline_tensorop(operand, node)
        elif isinstance(operand, ast.Subscript):
            temp_name = self.TEMP_SUBSCRIPT.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            self.handle_subscript_operation(operand, temp_name, node)
            return temp_name
        elif isinstance(operand, (ast.Constant, ast.Num)):
            return operand.value if isinstance(operand, ast.Constant) else operand.n
        elif isinstance(operand, ast.BinOp):
            temp_name = self.TEMP_BINOP.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            temp_target = ast.Name(id=temp_name, ctx=ast.Store())
            binop_assign = ast.Assign(targets=[temp_target], value=operand)
            binop_assign.lineno = node.lineno
            binop_assign.col_offset = node.col_offset
            self.handle_forward_binop(binop_assign)
            return temp_name
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: Binary operation has unsupported {side} operand type '{type(operand).__name__}'. This operation will be skipped."
            )
            return None

    def _determine_binop_var_types(self, left_layer, right_layer, left_var, right_var):
        """Determine var types (output/hidden) for RNN operands."""
        actual_left_var = self.variable_aliases.get(left_var, left_var) if isinstance(left_var, str) else left_var
        actual_right_var = self.variable_aliases.get(right_var, right_var) if isinstance(right_var, str) else right_var

        var_types = []
        for lyr_name, actual_var in zip([left_layer, right_layer], [actual_left_var, actual_right_var]):
            if isinstance(lyr_name, (int, float)):
                var_types.append("output")
            elif lyr_name in self.rnn_hidden_vars and actual_var == self.rnn_hidden_vars[lyr_name]:
                var_types.append("hidden")
            elif lyr_name in self.rnn_output_vars and actual_var == self.rnn_output_vars[lyr_name]:
                var_types.append("output")
            else:
                var_types.append("output")
        return var_types

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

        # Extract operands
        left_var = self._extract_binop_operand(binop.left, node, "left")
        if left_var is None:
            return

        right_var = self._extract_binop_operand(binop.right, node, "right")
        if right_var is None:
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
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported binary operation '{op_type_name}'. Only add, subtract, multiply, and divide are supported."
            )
            return

        # Get the layer/module names that produced these variables
        # For constants (float/int), we use the value directly instead of a layer name
        left_layer = left_var if isinstance(left_var, (int, float)) else self.module_of_output.get(left_var)
        right_layer = right_var if isinstance(right_var, (int, float)) else self.module_of_output.get(right_var)

        if left_layer is None or right_layer is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: Cannot determine source layers for binary operation. Make sure both operands are defined earlier."
            )
            return

        # Determine if variables are output or hidden for RNNs with return_type="both"
        var_types = self._determine_binop_var_types(left_layer, right_layer, left_var, right_var)

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

        # Update prev_layer_output for TensorOps
        self.prev_layer_output = output_var

        # Mark source layers for residual connections
        # When we have x = a + b where a and b are from different layers,
        # mark both source layers so their inputs are preserved
        if tns_type == "binop_add" and isinstance(left_layer, str) and isinstance(right_layer, str):
            # Find the layer objects for both operands
            for layer_name in [left_layer, right_layer]:
                if layer_name and not isinstance(layer_name, (int, float)):
                    # Look up the layer that produced this output
                    layer_obj = self._get_layer_by_name(layer_name)
                    if layer_obj:
                        # Mark that this layer's input should be preserved (residual connection)
                        layer_obj.input_reused = True

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
                # Resolve source variable to its producing module
                # If source_var is in module_of_output, use that module as the source
                # Otherwise, use source_var directly (will be treated as INPUT by generator)
                source_module = self.module_of_output.get(source_var, source_var)

                # Create a TensorOp for each non-underscore variable
                for idx, var_name in enumerate(var_names):
                    if var_name != '_':  # Skip underscore placeholders
                        # Use the variable name directly as the TensorOp name
                        # This ensures the generated code uses the correct variable names
                        tensorop_param = {
                            "name": var_name,
                            "tns_type": "shape_dim",
                            "layers_of_tensors": [source_module],  # Resolved source module
                            "reduce_dim": idx  # Dimension index
                        }
                        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
                        self.buml_model.add_tensor_op(tns_obj)

                        # Track the output variable
                        self.module_of_output[var_name] = var_name

    def _handle_non_rnn_slicing(self, node, subscripted_var, result_var):
        """Create subscript TensorOp for non-RNN slicing operations."""
        subscript_pattern = self.extract_subscript_pattern(node.value)
        self.create_subscript_tensorop(subscripted_var, subscript_pattern, result_var)
        self.previous_assign = node

    def _update_hidden_as_output(self, lyr_obj, prev_module_name):
        """Update layer to use hidden state as primary output."""
        lyr_obj.use_hidden_as_output = True
        if prev_module_name in self.rnn_hidden_vars:
            hidden_var = self.rnn_hidden_vars[prev_module_name]
            if prev_module_name in self.inputs_outputs:
                self.inputs_outputs[prev_module_name][1] = hidden_var

    def _determine_rnn_slice_return_type(self, node, lyr_obj, prev_module_name, subscripted_var, result_var):
        """Determine RNN return type based on slicing pattern."""
        if isinstance(node.value.slice, ast.UnaryOp):
            has_output = prev_module_name in self.rnn_output_vars
            has_hidden = prev_module_name in self.rnn_hidden_vars

            if has_output and has_hidden:
                lyr_obj.return_type = "both"
                self._update_hidden_as_output(lyr_obj, prev_module_name)
            else:
                lyr_obj.return_type = "hidden"
        elif isinstance(node.value.slice, ast.Tuple) and len(node.value.slice.elts) == 3:
            if lyr_obj.return_type == "full":
                self._handle_non_rnn_slicing(node, subscripted_var, result_var)
                return True
            if lyr_obj.return_type != "both":
                lyr_obj.return_type = "last"
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unrecognized subscript pattern on variable '{subscripted_var}'. This may not migrate correctly."
            )
        return False

    def handle_forward_slicing(self, node: ast.Assign):
        """
        It handles rnn slicing calls such as 'x = x[:, -1, :]' or 'h = h[-1]'

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        subscripted_var = node.value.value.id
        result_var = node.targets[0].id

        # Look up which module produced this variable
        if subscripted_var not in self.module_of_output:
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        prev_module_name = self.module_of_output[subscripted_var]
        lyr_obj = next((obj for obj in self.buml_model.layers if obj.name == prev_module_name), None)

        # If not an RNN layer, handle as regular subscript
        if not lyr_obj or not hasattr(lyr_obj, 'return_type'):
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        # Determine RNN return type based on slice pattern
        if self._determine_rnn_slice_return_type(node, lyr_obj, prev_module_name, subscripted_var, result_var):
            return

        # Track the result variable and alias
        self.module_of_output[result_var] = prev_module_name
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
            self.migration_warnings.append(
                f"Line {node.lineno}: Could not determine if path is for training or test data. Path will be ignored."
            )


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
        intermediate_var = self.TEMP_INLINE.format(op_type, self.tensor_op_counter)

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
        self.buml_model.add_tensor_op(tns_obj)  # Already adds to modules
        self.tensor_op_counter += 1

        # Track the intermediate variable
        self.module_of_output[intermediate_var] = tensorop_param["name"]

        return intermediate_var

    def _create_and_track_tensorop(self, tensorop_param, call_node, node):
        """
        Create TensorOp object and track its output.

        Parameters:
            tensorop_param (dict): Parameters for TensorOp creation
            call_node (ast.Call): The call node for source variable extraction
            node (ast.Assign): The assignment node for output tracking

        Returns:
            None, but adds TensorOp to model and tracks output
        """
        op_name = f"op_{self.tensor_op_counter}"
        tensorop_param["name"] = op_name
        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1

        # Check if this tensorop's source variable was saved for residual connection
        if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
            source_var = call_node.func.value.id
            if hasattr(self, '_variables_saved_for_residual') and source_var in self._variables_saved_for_residual:
                tns_obj.input_reused = True
                self._variables_saved_for_residual.discard(source_var)

        # Track tensorop output - handle both simple and tuple assignments
        if isinstance(node.targets[0], ast.Name):
            output_var = node.targets[0].id
        elif isinstance(node.targets[0], ast.Tuple):
            first_elem = node.targets[0].elts[0]
            output_var = first_elem.id if isinstance(first_elem, ast.Name) else None
        else:
            output_var = None

        if output_var:
            self.module_of_output[output_var] = op_name

    def _handle_functional_activation(self, node, module_name):
        """Handle functional API activation functions."""
        # Extract input variable
        if not node.value.args or not isinstance(node.value.args[0], ast.Name):
            self.migration_warnings.append(
                f"Line {node.lineno}: Could not extract input for activation function. Using default input variable 'x'."
            )
            input_var = "x"
        else:
            input_var = node.value.args[0].id

        prev_is_tensorop = False
        prev_lyr_obj = None

        # Check if previous module is a TensorOp
        if input_var in self.module_of_output:
            prev_lyr_name = self.module_of_output[input_var]
            prev_lyr_obj = self._get_module_by_name(prev_lyr_name)
            prev_cls = prev_lyr_obj.__class__.__name__ if prev_lyr_obj else "None"
            if prev_lyr_obj and prev_cls == "TensorOp":
                prev_is_tensorop = True

        if prev_is_tensorop:
            # Create standalone activation layer
            actv_lyr_name = f"activ_{module_name}_{self.tensor_op_counter}"
            self.tensor_op_counter += 1
            actv_lyr = mm_classes.GeneralLayer(
                name=actv_lyr_name,
                actv_func=actv_fun_mapping[module_name]
            )
            self.buml_model.add_layer(actv_lyr)
            self.buml_model.modules.append(actv_lyr)

            output_var = node.targets[0].id
            self.inputs_outputs[actv_lyr_name] = [input_var, output_var]
            self.module_of_output[output_var] = actv_lyr_name
        else:
            # Attach activation to previous layer
            if prev_lyr_obj:
                prev_lyr_obj.actv_func = actv_fun_mapping[module_name]
                if hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer:
                    if prev_lyr_name in self.inputs_outputs:
                        self.inputs_outputs[prev_lyr_name][1] = node.targets[0].id
            if input_var in self.module_of_output:
                self.module_of_output[node.targets[0].id] = self.module_of_output[input_var]

    def _handle_functional_regular_layer(self, node, module_name, synthetic_name):
        """Handle functional API regular (non-activation) layers."""
        lyr_params = {'positional_params': []}
        for arg in node.value.args[1:]:  # Skip first arg (input tensor)
            lyr_params['positional_params'].append(self.param_value(arg))

        for kw in node.value.keywords:
            lyr_params[kw.arg] = self.param_value(kw.value)

        try:
            lyr_type, lyr_params = transform_layer(module_name, lyr_params, synthetic_name)
            lyr_obj = getattr(mm_classes, lyr_type)(**lyr_params)
            self.buml_model.add_layer(lyr_obj)

            self.inputs_outputs[synthetic_name] = [node.value.args[0].id, node.targets[0].id]
            self.module_of_output[node.targets[0].id] = synthetic_name
            self.buml_model.modules.append(lyr_obj)
        except ValueError as e:
            self.migration_warnings.append(f"Functional layer '{synthetic_name}': {str(e)}")

    def _handle_module_activation(self, node, module_name):
        """Handle module API activation functions with merge logic."""
        should_merge = not (hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer)

        if should_merge and self.previous_assign and isinstance(self.previous_assign.value, ast.Call):
            if hasattr(self.previous_assign.value.func, 'attr'):
                prev_lyr_name = self.previous_assign.value.func.attr
                prev_lyr_obj = self._get_module_by_name(prev_lyr_name)
                if prev_lyr_obj:
                    actv = self.activation_functions[module_name]
                    actv_func = actv_fun_mapping.get(actv)
                    if actv_func is None:
                        self.migration_warnings.append(
                            f"Unsupported activation function '{actv}'. This activation will be skipped in the migration."
                        )
                        should_merge = False
                    else:
                        prev_lyr_obj.actv_func = actv_func
                        output_var = node.targets[0].id
                        self.module_of_output[output_var] = prev_lyr_name
                        should_merge = True
                else:
                    should_merge = False
            else:
                should_merge = False
        else:
            should_merge = False

        # Create standalone activation if not merged
        if not should_merge:
            unique_name = f"{module_name}_{self.tensor_op_counter}"
            self.tensor_op_counter += 1

            actv = self.activation_functions[module_name]
            actv_func = actv_fun_mapping.get(actv)
            if actv_func is None:
                self.migration_warnings.append(
                    f"Unsupported activation function '{actv}'. This activation will be skipped in the migration."
                )
            else:
                actv_lyr = mm_classes.GeneralLayer(name=unique_name, actv_func=actv_func)
                self.buml_model.modules.append(actv_lyr)

                input_var = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else "x"
                output_var = node.targets[0].id
                self.inputs_outputs[unique_name] = [input_var, output_var]
                self.module_of_output[output_var] = unique_name

    def _handle_module_layer_reuse(self, node, module_name, module_obj):
        """Handle layer reuse by creating synthetic copy."""
        import copy
        synthetic_name = f"{module_name}_use_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        synthetic_module = copy.copy(module_obj)
        synthetic_module.name = synthetic_name

        if module_name in self.inputs_outputs:
            self.inputs_outputs[synthetic_name] = self.inputs_outputs[module_name]

        output_var = node.targets[0].id
        self.module_of_output[output_var] = synthetic_name

        self.buml_model.layers.append(synthetic_module)
        self.buml_model.modules.append(synthetic_module)

    def _process_module_api(self, node, module_name):
        """Process module API calls (self.layer)."""
        # Check multi-layer RNN
        if module_name in self.multi_layer_rnns:
            self.expand_multi_layer_rnn_call(node, module_name, is_tuple=False)
            return

        # Extract input variable
        input_arg = node.value.args[0]
        if isinstance(input_arg, ast.Name):
            input_var = input_arg.id
        elif isinstance(input_arg, ast.Call):
            input_var = self.extract_inline_tensorop(input_arg, node)
        else:
            input_var = "x"

        self.inputs_outputs[module_name] = [input_var, node.targets[0].id]
        self.module_of_output[node.targets[0].id] = module_name

        # Detect parallel operations and TensorOp usage
        module_obj = self._get_layer_by_name(module_name)
        if module_obj and input_var in self.module_of_output:
            source_module = self.module_of_output[input_var]
            if source_module.startswith("op_"):
                module_obj.input_reused = True
                if self.prev_layer_output and input_var != self.prev_layer_output:
                    module_obj.name_module_input = source_module
            elif self.prev_layer_output and input_var != self.prev_layer_output:
                module_obj.input_reused = True

        self.prev_layer_output = node.targets[0].id

        # Check residual connection
        if hasattr(self, '_variables_saved_for_residual') and input_var in self._variables_saved_for_residual:
            if module_obj:
                module_obj.input_reused = True
            self._variables_saved_for_residual.discard(input_var)

        is_subnn_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)

        # Handle activation functions
        if module_name in self.activation_functions:
            self._handle_module_activation(node, module_name)
        elif not is_subnn_obj:
            self.is_permute_before_cnn(module_name)

        # Handle non-activation layers
        if module_name not in self.activation_functions:
            module_obj = next((obj for obj in self.buml_model.layers if obj.name == module_name), None)
            if not module_obj:
                module_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)

            # Handle layer reuse
            if module_obj and module_obj in self.buml_model.modules:
                self._handle_module_layer_reuse(node, module_name, module_obj)
            elif module_obj:
                self.buml_model.modules.append(module_obj)

    def _process_functional_api(self, node):
        """Process functional API calls (F.layer, torch.layer)."""
        func_name = node.value.func.attr

        if func_name not in functional_to_module_mapping:
            self.extract_tensorop(node)
            return

        module_name = functional_to_module_mapping[func_name]
        synthetic_name = f"f_{func_name}_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        if module_name in actv_fun_mapping:
            self._handle_functional_activation(node, module_name)
        else:
            self._handle_functional_regular_layer(node, module_name, synthetic_name)

    def _extract_op_multiply(self, call_node, node, op_type):
        """Extract mul/matmul operation parameters."""
        op_args = call_node.args
        if (not op_args or len(op_args) < 2 or
            not isinstance(op_args[0], ast.Name) or not isinstance(op_args[1], ast.Name)):
            self.migration_warnings.append(
                f"Line {node.lineno}: The {op_type} operation has invalid arguments and will be skipped."
            )
            return None

        left_var = op_args[0].id
        right_var = op_args[1].id

        if left_var not in self.module_of_output or right_var not in self.module_of_output:
            self.migration_warnings.append(
                f"Line {node.lineno}: Cannot find source variables for {op_type} operation. Make sure both operands are defined earlier in the code."
            )
            return None

        layers_of_tensors = [self.module_of_output[left_var], self.module_of_output[right_var]]
        return {"tns_type": op_type+"tiply", "layers_of_tensors": layers_of_tensors}

    def _extract_op_transpose(self, call_node, op_args):
        """Extract transpose operation parameters."""
        transpose_dim = [op_args[i].value for i in range(len(op_args))]
        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None
        return {"tns_type": "transpose", "transpose_dim": transpose_dim, "layers_of_tensors": source_layers}

    def _extract_op_reshape(self, call_node, node, op_args):
        """Extract reshape/view operation parameters."""
        reshape_dim = []
        for arg in op_args:
            if isinstance(arg, ast.Call) and hasattr(arg.func, 'attr') and arg.func.attr == 'size':
                if len(arg.args) > 0:
                    dim_idx = self.param_value(arg.args[0])
                    source_var = arg.func.value.id if isinstance(arg.func.value, ast.Name) else None
                    if source_var and source_var in self.module_of_output:
                        source_layers = [self.module_of_output[source_var]]
                    elif source_var:
                        source_layers = ['INPUT']
                    else:
                        source_layers = None

                    op_name = f"op_{self.tensor_op_counter}"
                    shape_tensorop_param = {"tns_type": "shape_dim", "reduce_dim": dim_idx,
                                           "layers_of_tensors": source_layers, "name": op_name}
                    tns_obj = getattr(mm_classes, "TensorOp")(**shape_tensorop_param)
                    self.buml_model.add_tensor_op(tns_obj)
                    self.tensor_op_counter += 1
                    reshape_dim.append(op_name)
                else:
                    reshape_dim.append(self.param_value(arg))
            elif isinstance(arg, ast.Name) and arg.id in self.module_of_output:
                layer_name = self.module_of_output[arg.id]
                if layer_name.startswith('op_'):
                    reshape_dim.append(layer_name)
                else:
                    reshape_dim.append(self.param_value(arg))
            else:
                reshape_dim.append(self.param_value(arg))

        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        tensorop_param = {"tns_type": "reshape", "reshape_dim": reshape_dim}
        if source_layers:
            tensorop_param["layers_of_tensors"] = source_layers
        return tensorop_param

    def _extract_op_size(self, call_node, node, op_args):
        """Extract size operation parameters."""
        if len(op_args) > 0:
            dim_idx = self.param_value(op_args[0])
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: The size() operation without dimension argument is not fully supported. Specify a dimension for better results."
            )
            return None

        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        return {"tns_type": "shape_dim", "reduce_dim": dim_idx, "layers_of_tensors": source_layers}

    def _extract_op_mean(self, call_node, node, op_args):
        """Extract mean operation parameters."""
        reduce_dim = None
        for kw in call_node.keywords:
            if kw.arg == "dim":
                reduce_dim = self.param_value(kw.value)
        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: The mean operation requires a dim parameter. Please specify which dimension to reduce."
            )
            return None

        if isinstance(call_node.func.value, ast.Name):
            if call_node.func.value.id in ['torch', 'F', 'nn']:
                source_var = op_args[0].id if len(op_args) > 0 and isinstance(op_args[0], ast.Name) else None
            else:
                source_var = call_node.func.value.id
        else:
            source_var = None

        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None
        return {"tns_type": "mean", "reduce_dim": reduce_dim, "layers_of_tensors": source_layers}

    def _extract_op_max(self, call_node, node, op_args):
        """Extract max operation parameters."""
        reduce_dim = None
        for kw in call_node.keywords:
            if kw.arg == "dim":
                reduce_dim = self.param_value(kw.value)
        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: The max operation requires a dim parameter. Please specify which dimension to reduce."
            )
            return None

        if isinstance(call_node.func.value, ast.Name):
            if call_node.func.value.id in ['torch', 'F', 'nn']:
                source_var = op_args[0].id if len(op_args) > 0 and isinstance(op_args[0], ast.Name) else None
            else:
                source_var = call_node.func.value.id
        else:
            source_var = None

        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None
        return {"tns_type": "max", "reduce_dim": reduce_dim, "layers_of_tensors": source_layers}

    def _extract_op_amax(self, call_node, node, op_args):
        """Extract amax operation parameters (same as max with dim)."""
        reduce_dim = None
        for kw in call_node.keywords:
            if kw.arg == "dim":
                reduce_dim = self.param_value(kw.value)
        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: The amax operation requires a dim parameter. Please specify which dimension to reduce."
            )
            return None

        if isinstance(call_node.func.value, ast.Name):
            if call_node.func.value.id in ['torch', 'F', 'nn']:
                source_var = op_args[0].id if len(op_args) > 0 and isinstance(op_args[0], ast.Name) else None
            else:
                source_var = call_node.func.value.id
        else:
            source_var = None

        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None
        return {"tns_type": "max", "reduce_dim": reduce_dim, "layers_of_tensors": source_layers}

    def _extract_op_squeeze(self, call_node, op_args):
        """Extract squeeze operation parameters."""
        squeeze_dim = None
        if len(op_args) > 0:
            squeeze_dim = self.param_value(op_args[0])
        else:
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    squeeze_dim = self.param_value(kw.value)
        return {"tns_type": "squeeze", "reduce_dim": squeeze_dim}

    def _extract_op_unsqueeze(self, call_node, op_args):
        """Extract unsqueeze operation parameters."""
        unsqueeze_dim = None
        if len(op_args) > 0:
            unsqueeze_dim = self.param_value(op_args[0])
        else:
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    unsqueeze_dim = self.param_value(kw.value)
        return {"tns_type": "unsqueeze", "reduce_dim": unsqueeze_dim}

    def _extract_op_normalize(self, call_node, op_args):
        """Extract normalize operation parameters."""
        norm_dim = None
        source_var = None

        if len(op_args) > 0 and isinstance(op_args[0], ast.Name):
            source_var = op_args[0].id
            if len(op_args) > 2:
                norm_dim = self.param_value(op_args[2])

        for kw in call_node.keywords:
            if kw.arg == "dim":
                norm_dim = self.param_value(kw.value)

        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        return {"tns_type": "normalize", "reduce_dim": norm_dim, "layers_of_tensors": source_layers}

    def _extract_op_flatten(self, call_node, node, op_args):
        """Extract flatten operation and create FlattenLayer (not a TensorOp)."""
        start_dim = 1
        end_dim = -1

        if len(op_args) > 0:
            start_dim = self.param_value(op_args[0])
        if len(op_args) > 1:
            end_dim = self.param_value(op_args[1])

        for kw in call_node.keywords:
            if kw.arg == "start_dim":
                start_dim = self.param_value(kw.value)
            elif kw.arg == "end_dim":
                end_dim = self.param_value(kw.value)

        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
        name_module_input = self.module_of_output.get(source_var, source_var)

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

        if isinstance(node.targets[0], ast.Name):
            output_var = node.targets[0].id
            self.module_of_output[output_var] = layer_name

        return "flatten_created"  # Special marker to skip TensorOp creation

    def _extract_op_repeat(self, call_node, op_args):
        """Extract repeat operation parameters."""
        repeat_counts = []
        for arg in op_args:
            if isinstance(arg, ast.Name):
                var_name = arg.id
                if var_name in self.module_of_output:
                    repeat_counts.append(self.module_of_output[var_name])
                else:
                    repeat_counts.append(var_name)
            else:
                repeat_counts.append(self.param_value(arg))

        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        return {"tns_type": "repeat", "repeat_dim": repeat_counts, "layers_of_tensors": source_layers}

    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """
        # Handle .values attribute accessor (e.g., x.max(dim=1).values)
        call_node = node.value
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'values':
            call_node = node.value.value

        if not isinstance(call_node, ast.Call):
            return

        # Safely extract operation type
        if not hasattr(call_node.func, 'attr'):
            self.migration_warnings.append(
                f"Line {node.lineno}: Unknown tensor operation encountered. This operation will be skipped in the migration."
            )
            return

        op_type = call_node.func.attr
        op_args = call_node.args

        # Dispatch to appropriate handler using mapping
        handler = self._op_handlers.get(op_type)
        if not handler:
            self.migration_warnings.append(
                f"Unrecognized tensor operation '{op_type}'. This operation will be skipped in the migration."
            )
            return

        tensorop_param = handler(call_node, node, op_args)

        # Flatten creates a layer, not a tensorop (special case)
        if tensorop_param == "flatten_created":
            return

        # Create and track TensorOp if handler returned params
        if tensorop_param:
            self._create_and_track_tensorop(tensorop_param, call_node, node)


    def _extract_concat_arg_variable(self, arg, node):
        """Extract variable name from concatenation argument, handling layer calls and inline ops."""
        if isinstance(arg, ast.Name):
            return arg.id
        elif isinstance(arg, ast.Call):
            # Check if it's a layer call (self.layer_name(...))
            if (isinstance(arg.func, ast.Attribute) and
                isinstance(arg.func.value, ast.Name) and
                arg.func.value.id == 'self'):
                layer_name = arg.func.attr
                layer_obj = next((lyr for lyr in self.buml_model.layers if lyr.name == layer_name), None)

                if layer_obj:
                    # Handle layer reuse
                    if layer_obj in self.buml_model.modules:
                        self.layer_reuse_count[layer_name] = self.layer_reuse_count.get(layer_name, 0) + 1
                        use_count = self.layer_reuse_count[layer_name]

                        import copy
                        reuse_layer = copy.copy(layer_obj)
                        reuse_layer.name = f"{layer_name}_use_{use_count}"
                        reuse_layer.input_reused = True

                        if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
                            input_var = arg.args[0].id
                            if input_var in self.module_of_output:
                                reuse_layer.name_module_input = self.module_of_output[input_var]

                        self.buml_model.modules.append(reuse_layer)
                        temp_name = self.TEMP_NESTED.format(self.tensor_op_counter)
                        self.tensor_op_counter += 1
                        self.module_of_output[temp_name] = reuse_layer.name
                        return temp_name
                    else:
                        # First use
                        if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
                            input_var = arg.args[0].id
                            if input_var in self.module_of_output:
                                layer_obj.name_module_input = self.module_of_output[input_var]
                                layer_obj.input_reused = True

                        self.buml_model.modules.append(layer_obj)
                        temp_name = self.TEMP_NESTED.format(self.tensor_op_counter)
                        self.tensor_op_counter += 1
                        self.module_of_output[temp_name] = layer_name
                        return temp_name
                else:
                    self.migration_warnings.append(
                        f"Concatenation operation: Layer '{layer_name}' referenced but not found in the model. Check if it was defined earlier."
                    )
                    return None
            else:
                # Method call like x.squeeze(1)
                result = self.extract_inline_tensorop(arg, node)
                if result is None:
                    self.migration_warnings.append(
                        f"Concatenation operation: Could not process inline operation '{ast.unparse(arg)}'. This argument will be skipped."
                    )
                    return None
                return result
        elif isinstance(arg, ast.Subscript):
            temp_name = self.TEMP_SUBSCRIPT.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            self.handle_subscript_operation(arg, temp_name, node)
            return temp_name
        else:
            return None

    def _check_bidirectional_rnn_concat(self, ops_args, layers_of_tensors, node):
        """Check if this is a bidirectional RNN concatenation pattern and handle it."""
        if not (len(ops_args) == 2 and len(set(layers_of_tensors)) == 1):
            return False

        source_layer_name = layers_of_tensors[0]
        source_layer = self._get_layer_by_name(source_layer_name)

        if not (source_layer and
                hasattr(source_layer, 'bidirectional') and source_layer.bidirectional and
                hasattr(source_layer, 'return_type') and source_layer.return_type == 'hidden'):
            return False

        # Case 1: Inline subscripts like torch.cat([h[-2], h[-1]])
        if all(isinstance(arg, ast.Subscript) for arg in ops_args):
            indices = []
            for arg in ops_args:
                if isinstance(arg.slice, ast.UnaryOp) and isinstance(arg.slice.op, ast.USub):
                    indices.append(-arg.slice.operand.value)
                elif isinstance(arg.slice, ast.Constant):
                    indices.append(arg.slice.value)
                else:
                    indices = None
                    break

            if indices and set(indices) == {-2, -1}:
                output_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                if output_var:
                    self.module_of_output[output_var] = source_layer_name
                return True

        # Case 2: Variables from subscripts
        elif all(isinstance(arg, ast.Name) for arg in ops_args):
            output_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
            if output_var:
                self.module_of_output[output_var] = source_layer_name
            return True

        return False

    def _determine_rnn_var_types(self, layers_of_tensors, actual_vars):
        """Determine if each variable is output or hidden for RNNs with return_type='both'."""
        var_types = []
        for lyr_name, actual_var in zip(layers_of_tensors, actual_vars):
            if lyr_name in self.rnn_hidden_vars and actual_var == self.rnn_hidden_vars[lyr_name]:
                var_types.append("hidden")
            elif lyr_name in self.rnn_output_vars and actual_var == self.rnn_output_vars[lyr_name]:
                var_types.append("output")
            else:
                var_types.append("output")
        return var_types

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

        # Update previous layer return_type if first arg is subscript
        if isinstance(ops_args[0], ast.Subscript):
            if (self.previous_assign and
                hasattr(self.previous_assign, 'value') and
                isinstance(self.previous_assign.value, ast.Call) and
                hasattr(self.previous_assign.value, 'func') and
                hasattr(self.previous_assign.value.func, 'attr')):
                prev_lyr_name = self.previous_assign.value.func.attr
                lyr_obj = next((obj for obj in self.buml_model.layers if obj.name == prev_lyr_name), None)
                if lyr_obj:
                    lyr_obj.return_type = "hidden"

        # Extract all variables from concatenation arguments
        variables = []
        for arg in ops_args:
            var = self._extract_concat_arg_variable(arg, node)
            if var is None:
                self.migration_warnings.append(
                    f"Line {node.lineno}: Cannot extract variable from concatenation argument. Make sure all concat arguments are valid variables or layer outputs."
                )
                return None
            variables.append(var)

        # Resolve aliases
        actual_vars = []
        for var in variables:
            actual_var = var if var in self.module_of_output else self.variable_aliases.get(var, var)
            actual_vars.append(actual_var)

        layers_of_tensors = [self.module_of_output[actual_var] for actual_var in actual_vars]
        cat_dim = self.param_value(node.value.keywords[0].value)

        # Check for bidirectional RNN pattern
        if self._check_bidirectional_rnn_concat(ops_args, layers_of_tensors, node):
            return None

        # Determine var types for RNNs
        var_types = self._determine_rnn_var_types(layers_of_tensors, actual_vars)

        return {
            "tns_type": "concatenate",
            "layers_of_tensors": layers_of_tensors,
            "concatenate_dim": cat_dim,
            "actual_vars": var_types
        }


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

    # Check if layer type is supported
    if lyr_type not in layers_mapping:
        raise ValueError(
            f"Unsupported layer type '{lyr_type}'. "
            f"This PyTorch layer is not yet supported in the migration tool. "
            f"Supported layers: {', '.join(sorted(layers_mapping.keys())[:10])}... "
            f"(and {len(layers_mapping) - 10} more)"
        )

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
        # Note: Unknown parameters are silently skipped (not critical for migration)


    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params

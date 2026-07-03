"""
Module to extract information from the AST of a neural network
written in PyTorch and transforms it to a BUML model.
It also extracts data and model configuration attributes.
"""
import ast
import copy
import sys
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
from besser.BUML.metamodel.nn import NN, Layer
import besser.BUML.metamodel.nn as mm_classes

from torch2tf.definitions import (
    layers_mapping, params_mapping, static_params,
    pos_params, int2list_params, lyrs_of_int2list_params,
    actv_fun_mapping, channels_first_layers, loss_func_mapping,
    functional_to_module_mapping
)
from ast_parser_nn import ASTParser
from transform_code import (
    process_positional_params, set_static_params, param_to_list
)


class ASTParserTorch(ASTParser):
    def _cleanup_temp_variables(self):
        """
        Replace temporary variables (_chain_temp, _nested_temp, etc.) with cleaner names.
        This runs after all forward method processing to preserve original variable names.
        """
        # Collect all temp variable patterns
        temp_patterns = [
            self.TEMP_CHAIN.split('{')[0],      # _chain_temp_
            self.TEMP_NESTED.split('{')[0],     # _nested_temp_
            self.TEMP_NESTED_IN_CHAIN.split('{')[0],  # _nested_in_chain_
        ]

        temp_vars = set()

        # For each temp variable, find what it should be renamed to
        # Strategy: if a temp is only used as input to one operation that outputs to a real var,
        # replace the temp with that real var
        for temp_var in temp_vars:
            # CRITICAL FIX: Don't collapse _nested_temp_* if produced by a reused layer
            # In nested calls like r = proj(dropout(m)) where dropout is reused,
            # we need _nested_temp_2 = dropout_use_1(m) as an intermediate assignment
            if temp_var.startswith(self.TEMP_NESTED.split('{')[0]):
                # Check if this temp is produced by a reused layer (contains "_use_")
                producer = self.module_of_output.get(temp_var)
                if producer and '_use_' in producer:
                    # Skip this temp variable - must be preserved for layer reuse
                    continue

            # Find operations that use this temp as input
            consumers = []

            # If temp has exactly 1 consumer and that consumer outputs to a non-temp var,
            # replace the temp with the consumer's output var
            if len(consumers) == 1:
                consumer_key = consumers[0]
                consumer_out = None
                # Check if consumer output is not a temp
                is_consumer_out_temp = any(consumer_out.startswith(p) for p in temp_patterns) if consumer_out else False

                if not is_consumer_out_temp and consumer_out:
                    # Update module_of_output
                    if temp_var in self.module_of_output:
                        if consumer_out not in self.module_of_output:
                            self.module_of_output[consumer_out] = self.module_of_output[temp_var]
                        del self.module_of_output[temp_var]

                    # CRITICAL: Also update tensorop/layer object attributes
                    for module in self.buml_model.modules:
                        if module.output_var is not None and module.output_var == temp_var:
                            module.output_var = consumer_out
                        if module.input_var is not None and module.input_var == temp_var:
                            module.input_var = consumer_out


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

    def _collect_used_variables(self, node):
        """
        Collect all variable names that are referenced (not just assigned) in the AST.
        This helps identify unused variables like LSTM cell states.
        """
        used_vars = set()

        class VarCollector(ast.NodeVisitor):
            def visit_Name(self, n):
                # Only collect names that are being loaded (used), not stored (assigned)
                if isinstance(n.ctx, ast.Load):
                    used_vars.add(n.id)
                self.generic_visit(n)

        VarCollector().visit(node)
        return used_vars

    def _mark_unused_cell_states(self, forward_node):
        """
        Check if LSTM cell state variables are actually used in the forward method.
        If not, mark the layer so the template can output `_` instead of the variable name.
        """
        used_vars = self._collect_used_variables(forward_node)

        for module_name, cell_var in self.lstm_cell_vars.items():
            if cell_var not in used_vars:
                # Cell state is not used, mark the layer
                layer = self._get_layer_by_name(module_name)
                if layer:
                    layer.cell_state_unused = True

    def __init__(self, input_nn_type: str, only_nn: bool):
        super().__init__(input_nn_type, only_nn)

        self.activation_functions = {}
        self.lstm_cell_vars = {}  # Track LSTM cell state variables to check if unused
        self.forward_node = None  # Store forward method node for later analysis
        # Track RNN output vs hidden variables separately
        self.rnn_output_vars = {}  # {module_name: output_var}
        self.rnn_hidden_vars = {}  # {module_name: hidden_var}
        # Track variable aliases (e.g., h_last -> h)
        self.variable_aliases = {}  # {alias_var: source_var}
        # Track multi-layer RNNs: {module_name: [layer_name_0, layer_name_1, ...]}
        self.multi_layer_rnns = {}
        # Track original num_layers for multi-layer RNNs: {module_name: num_layers}
        self.rnn_num_layers = {}
        # Track layer reuse count for creating unique reuse names
        self.layer_reuse_count = {}  # {layer_name: reuse_count}
        # Track the output variable of the immediately previous layer
        self.prev_layer_output = None  # str: variable name
        # Counter for adding unique suffix to ALL layer names
        self.layer_counter = 0
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
            "interpolate": lambda cn, n, args: self._extract_op_interpolate(cn, n, args),
            "pad": lambda cn, n, args: self._extract_op_pad(cn, n, args),
            "dropout": lambda cn, n, args: self._extract_op_dropout(cn, n, args),
            "zeros_like": lambda cn, n, args: self._extract_op_zeros_like(cn, args),
            "split": lambda cn, n, args: self._extract_op_split(cn, n, args),
            "chunk": lambda cn, n, args: self._extract_op_chunk(cn, n, args),
        }

    def _add_layer_with_tracking(self, layer_obj):
        """
        Add layer to model with counter suffix and update lookup dict for O(1) access.
        All layers get a suffix like _1, _2, etc. to ensure unique keys in modules_details.
        The generator strips this suffix when creating layer syntax and uses is_layer_call
        to skip duplicate definitions in __init__.
        """
        if hasattr(layer_obj, 'name'):
            base_name = layer_obj.name
            # Store base name for lookups (before adding suffix)
            if base_name not in self.layer_by_name:
                self.layer_by_name[base_name] = layer_obj
                self.module_by_name[base_name] = layer_obj

            # Add suffix to ALL layers using _c prefix to distinguish from other _N suffixes
            self.layer_counter += 1
            layer_obj.name = f"{base_name}_c{self.layer_counter}"

            # Store in lookup dicts using the SUFFIXED name
            self.layer_by_name[layer_obj.name] = layer_obj
            self.module_by_name[layer_obj.name] = layer_obj

        self.buml_model.add_layer(layer_obj)

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

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Override to store forward method and check for unused cell states after parsing.
        """
        # Find and store the forward method node
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == 'forward':
                self.forward_node = child
                break

        # Call parent's visit_ClassDef to do the actual parsing
        super().visit_ClassDef(node)

        # Clean up temporary variables to preserve original names
        self._cleanup_temp_variables()

        # After parsing, check for unused LSTM cell states
        if self.forward_node and self.lstm_cell_vars:
            self._mark_unused_cell_states(self.forward_node)

        # Store bidirectional concat variable names in the model for code generation
        if hasattr(self, '_bidirectional_concat_var_names'):
            self.buml_model.bidirectional_concat_var_names = self._bidirectional_concat_var_names

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

    def _resolve_variable_alias(self, var):
        """Resolve variable aliases to find the actual source."""
        resolved_var = var
        while resolved_var in self.variable_aliases:
            resolved_var = self.variable_aliases[resolved_var]
        return resolved_var

    def _create_subscript_op(self, op_name, source_module, subscript_pattern):
        """Create subscript TensorOp object."""
        return mm_classes.TensorOp(
            name=op_name,
            tns_type='subscript',
            layers_of_tensors=[source_module],
            subscript_indices=subscript_pattern
        )

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
        resolved_var = self._resolve_variable_alias(source_var)
        source_module = self.module_of_output.get(resolved_var, resolved_var)


        # Strip __hidden or __cell suffix from source_module for tensorop
        if source_module and source_module.endswith("__hidden"):
            source_module = source_module[:-8]
        elif source_module and source_module.endswith("__cell"):
            source_module = source_module[:-6]

        subscript_op = self._create_subscript_op(op_name, source_module, subscript_pattern)
        self.buml_model.modules.append(subscript_op)
        # NEW: Also set on tensorop object
        subscript_op.input_var = resolved_var
        subscript_op.output_var = output_var
        self.module_of_output[output_var] = op_name
        # Update prev_layer_output so next operation can detect branching
        self.prev_layer_output = output_var


    def _is_rnn_subscript(self, subscripted_var):
        """Check if subscript is on an RNN layer."""
        if not (subscripted_var and subscripted_var in self.module_of_output):
            return False

        src_module = self.module_of_output[subscripted_var]
        # Strip __hidden or __cell suffix to get actual layer name
        if src_module.endswith("__hidden"):
            src_module = src_module[:-8]
        elif src_module.endswith("__cell"):
            src_module = src_module[:-6]
        src_layer = self._get_layer_by_name(src_module)
        return src_layer and hasattr(src_layer, 'return_type')

    def _create_temp_assignment(self, temp_name, subscript_node, node):
        """Create temporary assignment node for subscript operation."""
        temp_target = ast.Name(id=temp_name, ctx=ast.Store())
        subscript_assign = ast.Assign(targets=[temp_target], value=subscript_node)
        subscript_assign.lineno = node.lineno
        subscript_assign.col_offset = node.col_offset
        return subscript_assign

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
        # If subscript value is a call (e.g., self.conv(x)[:, :, 0]), process it first
        if isinstance(subscript_node.value, ast.Call):
            inner_temp = self.TEMP_NESTED.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            inner_target = ast.Name(id=inner_temp, ctx=ast.Store())
            inner_node = ast.Assign(targets=[inner_target], value=subscript_node.value)
            inner_node.lineno = node.lineno
            inner_node.col_offset = node.col_offset
            self.process_single_call(inner_node)
            self.previous_assign = inner_node

            # CRITICAL FIX: Update the nested layer's output_var to match the final subscript output
            # For nested subscript like self.conv(x)[:, 0, :], we want:
            #   _subscript_temp_1 = self.conv(x)
            #   _subscript_temp_1 = _subscript_temp_1[:, 0, :]
            # NOT:
            #   _nested_temp_2 = self.conv(x)
            #   _subscript_temp_1 = _nested_temp_2[:, 0, :]
            if inner_temp in self.module_of_output:
                module_name = self.module_of_output[inner_temp]
                module_obj = self._get_layer_by_name(module_name)
                if module_obj:
                    module_obj.output_var = temp_name
                    self.module_of_output[temp_name] = module_name
                    # Use the updated variable name for the subscript input
                    subscripted_var = temp_name
            else:
                subscripted_var = inner_temp
            # Update the subscript node to reference the temp variable
            subscript_node.value = ast.Name(id=inner_temp, ctx=ast.Load())
        else:
            subscripted_var = subscript_node.value.id if isinstance(subscript_node.value, ast.Name) else None

        subscript_assign = self._create_temp_assignment(temp_name, subscript_node, node)


        if self._is_rnn_subscript(subscripted_var):
            self.handle_forward_slicing(subscript_assign)
        else:
            subscript_pattern = self.extract_subscript_pattern(subscript_node)
            self.create_subscript_tensorop(subscripted_var, subscript_pattern, temp_name)

        self.previous_assign = subscript_assign

    def _create_rnn_layer_params(self, lyr_params, i, num_layers, module_name, original_return_type):
        """Create parameters for individual RNN layer in multi-layer stack."""
        layer_params = lyr_params.copy()
        layer_params.pop('num_layers', None)
        layer_params['name'] = f"{module_name}_layer_{i+1}"

        if i > 0:
            # For layers after the first, input_size = hidden_size of previous layer
            # If previous layer is bidirectional, multiply by 2
            prev_hidden_size = lyr_params['hidden_size']
            if lyr_params.get('bidirectional', False):
                prev_hidden_size *= 2
            layer_params['input_size'] = prev_hidden_size

        if i < num_layers - 1:
            layer_params['return_type'] = 'full'
        else:
            layer_params['return_type'] = original_return_type

        layer_params['positional_params'] = []
        return layer_params

    def _add_rnn_layer(self, lyr_type, layer_params):
        """Transform and create a single RNN layer, store in lookup dicts."""
        try:
            buml_lyr_type, buml_params = transform_layer(lyr_type, layer_params, layer_params['name'])
            buml_layer = getattr(mm_classes, buml_lyr_type)(**buml_params)
            # Store in lookup dicts but DON'T add to model yet
            # Layer will be added via _add_layer_with_tracking when processing forward()
            layer_name = layer_params['name']
            self.layer_by_name[layer_name] = buml_layer
            self.module_by_name[layer_name] = buml_layer
        except ValueError as e:
            self.migration_warnings.append(f"Layer '{layer_params['name']}': {str(e)}")

    def _create_multi_layer_rnn(self, lyr_type, lyr_params, module_name):
        """Create multiple BUML layers for stacked RNN."""
        process_positional_params(lyr_type, lyr_params, pos_params)
        num_layers = lyr_params.get('num_layers', 1)
        original_return_type = lyr_params.get('return_type', 'full')
        dropout_rate = lyr_params.get('dropout', 0.0)
        layer_names = []

        for i in range(num_layers):
            layer_params = self._create_rnn_layer_params(lyr_params, i, num_layers, module_name, original_return_type)
            # Remove dropout from individual layers - will be added as separate layers instead
            # because PyTorch dropout (between layers) != TensorFlow dropout (on inputs)
            layer_params['dropout'] = 0.0
            layer_names.append(layer_params['name'])
            self._add_rnn_layer(lyr_type, layer_params)

            # Create dropout layer between RNN layers (not after last layer)
            if i < num_layers - 1 and dropout_rate > 0:
                dropout_name = f"{module_name}_dropout_{i+1}"
                dropout_layer = getattr(mm_classes, "DropoutLayer")(
                    name=dropout_name,
                    rate=dropout_rate
                )
                # Store in lookup dicts - will be inserted in forward() via expand_multi_layer_rnn_call
                self.layer_by_name[dropout_name] = dropout_layer
                self.module_by_name[dropout_name] = dropout_layer

        self.multi_layer_rnns[module_name] = layer_names
        self.rnn_num_layers[module_name] = num_layers

    def _create_single_layer(self, lyr_type, lyr_params, module_name):
        """Create single BUML layer."""
        lyr_params.pop('num_layers', None)
        try:
            lyr_type, lyr_params = transform_layer(lyr_type, lyr_params, module_name)
            buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
            # Store in lookup dict but DON'T add to model yet
            # Layer will be added via _add_layer_with_tracking when processing forward()
            self.layer_by_name[module_name] = buml_layer
            self.module_by_name[module_name] = buml_layer
        except ValueError as e:
            self.migration_warnings.append(f"Layer '{module_name}': {str(e)}")

    def handle_init(self, node: ast.Assign):
        """
        It retrieves the sub_nn layers, adds their activation functions
        as parameters and stores them in the 'sub_nn' dict. It also
        retrieves the layers and their parameters and stores them in
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
            if last_lyr_type in channels_first_layers:
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
        """
        Extract and process nested call within a chain element.
        Recursively handles multiple levels of nesting like pool(act(conv(x))).
        """
        if not (isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Call)):
            return

        inner_call = call.args[0]

        # RECURSIVELY process nested calls within the inner call
        # For pool(act(conv(x))), this extracts conv(x) from within act()
        self._process_nested_call_in_chain(inner_call, node)

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
        all_temp_names = []  # Track all temp variables created in this chain

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
                all_temp_names.append(temp_name)

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

        # Clean up chain temp variables to preserve original variable names
        if len(chain) > 1 and all_temp_names:
            target_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
            if not target_var:
                return

            # Replace all temp variables in the chain with target_var
            # This handles chains of any length (2+)
            for temp_name in all_temp_names:
                # Update module_of_output: transfer temp's module to target_var
                if temp_name in self.module_of_output:
                    if target_var not in self.module_of_output:
                        self.module_of_output[target_var] = self.module_of_output[temp_name]
                    del self.module_of_output[temp_name]

                # CRITICAL: Also update module object attributes (layers AND tensorops)
                for module in self.buml_model.modules:
                    if module.output_var is not None and module.output_var == temp_name:
                        module.output_var = target_var
                    if module.input_var is not None and module.input_var == temp_name:
                        module.input_var = target_var

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

        # Check if this created an alias (no-op subscript) and use the base variable instead
        if temp_name in self.variable_aliases:
            base_var = self.variable_aliases[temp_name]
            node.value.args[0] = ast.Name(id=base_var, ctx=ast.Load())
        else:
            node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())

    def _is_noop_squeeze(self, call_node):
        """Check if a call is a no-op squeeze(0) on single-layer RNN hidden state."""
        if not isinstance(call_node, ast.Call):
            return False
        if not isinstance(call_node.func, ast.Attribute):
            return False
        if call_node.func.attr != 'squeeze':
            return False

        # Extract base variable
        if not isinstance(call_node.func.value, ast.Name):
            return False
        base_var = call_node.func.value.id

        # Check if it's squeeze(0)
        dim_value = None
        if len(call_node.args) > 0:
            dim_value = self.param_value(call_node.args[0])
        else:
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    dim_value = self.param_value(kw.value)

        if dim_value != 0:
            return False

        # Check if base_var is an RNN hidden/cell state
        if base_var not in self.module_of_output:
            return False

        base_layer = self.module_of_output[base_var]
        if not (base_layer and (base_layer.endswith('__hidden') or base_layer.endswith('__cell'))):
            return False

        # Check if it's a single-layer RNN
        rnn_layer_name = base_layer.replace('__hidden', '').replace('__cell', '')
        rnn_layer = self._get_layer_by_name(rnn_layer_name)
        if not rnn_layer:
            return False

        num_layers = getattr(rnn_layer, 'num_layers', None)
        if rnn_layer.__class__.__name__ not in ('SimpleRNNLayer', 'LSTMLayer', 'GRULayer'):
            return False
        if not (num_layers is None or num_layers == 1):
            return False

        return True

    def _handle_nested_call(self, node):
        """Process nested calls (e.g., F.relu(self.conv(x))). Returns True if nested."""
        if not (isinstance(node.value, ast.Call) and node.value.args):
            return False
        if not isinstance(node.value.args[0], ast.Call):
            return False

        inner_call = node.value.args[0]

        # Check if inner call is a no-op squeeze: if so, skip intermediate and use base var
        if self._is_noop_squeeze(inner_call):
            base_var = inner_call.func.value.id
            node.value.args[0] = ast.Name(id=base_var, ctx=ast.Load())
            return False  # Not treated as nested since we're bypassing it

        # Use the outer output variable instead of creating a temp variable
        # For x = F.relu(self.bn(self.conv(x))), use 'x' throughout
        # BUT for shape operations like .size(), use synthetic variable to avoid type conflicts
        outer_var = node.targets[0].id

        # Check if inner call is a shape operation
        is_shape_op = (isinstance(inner_call, ast.Call) and
                       hasattr(inner_call.func, 'attr') and
                       inner_call.func.attr in ('size', 'shape'))

        if is_shape_op:
            # Use synthetic variable for shape operations to preserve tensor variable
            temp_var_name = f"_shape_{self.tensor_op_counter}"
            temp_target = ast.Name(id=temp_var_name, ctx=ast.Store())
        else:
            temp_target = ast.Name(id=outer_var, ctx=ast.Store())

        inner_node = ast.Assign(targets=[temp_target], value=inner_call)
        inner_node.lineno = node.lineno
        inner_node.col_offset = node.col_offset

        # Handle subscript arguments in the inner call before processing
        self._handle_subscript_argument(inner_node)

        self.visit_Assign(inner_node)
        self.previous_assign = inner_node

        # Replace the inner call with reference to the temp variable
        replacement_var = temp_var_name if is_shape_op else outer_var
        node.value.args[0] = ast.Name(id=replacement_var, ctx=ast.Load())
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

        # Extract the actual sequence output variable (first element)
        first_elem = node.targets[0].elts[0]
        if isinstance(first_elem, ast.Name):
            sequence_output_var = first_elem.id
        else:
            sequence_output_var = None


        lyr_obj = self._get_layer_by_name(layer_name)
        if lyr_obj:
            # NEW: Set input/output on layer object
            lyr_obj.input_var = current_input
            # output_var should be the sequence output (first element), not rnn_out (main output)
            lyr_obj.output_var = sequence_output_var
            self.buml_model.modules.append(lyr_obj)

    def _process_last_layer_simple(self, node, layer_name, current_input):
        """Process last layer with simple assignment in multi-layer RNN."""
        output_var = node.targets[0].id
        self.module_of_output[output_var] = layer_name

        lyr_obj = self._get_layer_by_name(layer_name)
        if lyr_obj:
            # NEW: Set input/output on layer object
            lyr_obj.input_var = current_input
            lyr_obj.output_var = output_var
            self.buml_model.modules.append(lyr_obj)

    def _process_intermediate_layer(self, module_name, layer_name, current_input, i):
        """Process intermediate layer in multi-layer RNN."""
        temp_var = self.TEMP_MLRNN.format(module_name, i + 1)
        self.module_of_output[temp_var] = layer_name

        module_obj = self._get_layer_by_name(layer_name)
        if module_obj:
            # NEW: Set input/output on layer object
            module_obj.input_var = current_input
            module_obj.output_var = temp_var
        if module_obj:
            self.buml_model.modules.append(module_obj)

        # Check if there's a dropout layer after this intermediate RNN layer
        dropout_name = f"{module_name}_dropout_{i+1}"
        dropout_obj = self._get_layer_by_name(dropout_name)
        if dropout_obj:
            # Insert dropout layer between RNN layers
            dropout_temp_var = f"{temp_var}_dropout"
            # NEW: Also set on dropout layer object
            dropout_obj.input_var = temp_var
            dropout_obj.output_var = dropout_temp_var
            self.module_of_output[dropout_temp_var] = dropout_name
            self.buml_model.modules.append(dropout_obj)
            return dropout_temp_var

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

        # Handle LSTM case: out, (h, c) = self.lstm(x)
        if isinstance(node.targets[0].elts[1], ast.Tuple):
            # LSTM: second element is a tuple (h, c)
            var2 = node.targets[0].elts[1].elts[0].id if isinstance(node.targets[0].elts[1].elts[0], ast.Name) else None
            var3 = node.targets[0].elts[1].elts[1].id if len(node.targets[0].elts[1].elts) > 1 and isinstance(node.targets[0].elts[1].elts[1], ast.Name) else None
        elif len(node.targets[0].elts) == 3:
            # LSTM with 3-element flat tuple: out, h, c = self.lstm(x)
            var2 = node.targets[0].elts[1].id if isinstance(node.targets[0].elts[1], ast.Name) else None
            var3 = node.targets[0].elts[2].id if isinstance(node.targets[0].elts[2], ast.Name) else None
        else:
            # RNN/GRU: second element is just h
            var2 = node.targets[0].elts[1].id
            var3 = None


        if var1 and var1 != "_":
            self.rnn_output_vars[module_name] = var1
            self.module_of_output[var1] = module_name

        if var2 and var2 != "_":
            self.rnn_hidden_vars[module_name] = var2
            # Track hidden state with special suffix to distinguish from output sequence
            self.module_of_output[var2] = module_name + "__hidden"
            # For RNN tuple output (out, h), store hidden state with __hidden suffix
            # Use var2 as both input and output to preserve the original unpacked variable name
            # NEW: Set hidden state variable on layer object
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.hidden_state_var = var2
                lyr_obj.hidden_unused = False
        elif var2 == "_":
            # Store "_" to tell generator to use underscore instead of auto-generating a name
            # NEW: Mark hidden state as unused on layer object
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.hidden_state_var = "_"
                lyr_obj.hidden_unused = True

        if var3 and var3 != "_":
            # Track cell state for LSTM with special suffix
            self.module_of_output[var3] = module_name + "__cell"
            # Store cell state variable name to check if it's used later
            self.lstm_cell_vars[module_name] = var3
            # Use var3 as both input and output to preserve the original unpacked variable name
            # NEW: Set cell state variable on layer object
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.cell_state_var = var3
                lyr_obj.cell_unused = False
        elif var3 == "_":
            # Store "_" to tell generator to use underscore instead of auto-generating a name
            # NEW: Mark cell state as unused on layer object
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.cell_state_var = "_"
                lyr_obj.cell_unused = True

    def _determine_rnn_return_type(self, node, module_name):
        """Determine RNN return type and main output variable based on underscore pattern."""
        num_elts = len(node.targets[0].elts)
        first_elem = node.targets[0].elts[0]
        second_elem = node.targets[0].elts[1] if not isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].elts[0]

        first_is_underscore = isinstance(first_elem, ast.Name) and first_elem.id == "_"
        second_is_underscore = isinstance(second_elem, ast.Name) and second_elem.id == "_"

        lyr_obj = self._get_layer_by_name(module_name)


        if first_is_underscore and not second_is_underscore:
            rnn_out = node.targets[0].elts[1].elts[0].id if isinstance(node.targets[0].elts[1], ast.Tuple) else node.targets[0].elts[1].id
            if lyr_obj:
                lyr_obj.return_type = "hidden"
        elif not first_is_underscore and second_is_underscore:
            rnn_out = node.targets[0].elts[0].id
            if lyr_obj:
                # For 3-element tuples (LSTM), even if h and c are "_", we need return_state=True
                if num_elts == 3 or isinstance(node.targets[0].elts[1], ast.Tuple):
                    lyr_obj.return_type = "both"
                else:
                    lyr_obj.return_type = "full"
        else:
            rnn_out = node.targets[0].elts[0].id
            if lyr_obj:
                lyr_obj.return_type = "both"

        return rnn_out

    def _extract_rnn_input_arg(self, node):
        """Extract input argument from RNN call, handling both variables and inline operations.

        Also detects and handles initial hidden state parameter (h0) for seq2seq/decoder patterns.
        Returns: input_var (str)
        Side effect: Sets self._current_rnn_has_initial_state if h0 is provided
        """
        if not node.value.args:
            return "x"

        # Extract first argument (input sequence)
        input_arg = node.value.args[0]
        if isinstance(input_arg, ast.Name):
            input_var = input_arg.id
        elif isinstance(input_arg, ast.Call):
            input_var = self.extract_inline_tensorop(input_arg, node)
        else:
            input_var = "x"

        # Check for second argument (initial hidden state)
        if len(node.value.args) > 1:
            hidden_arg = node.value.args[1]
            if isinstance(hidden_arg, ast.Name):
                # Simple case: rnn(x, h0)
                self._current_rnn_initial_hidden = hidden_arg.id
            elif isinstance(hidden_arg, ast.Tuple):
                # LSTM case: rnn(x, (h0, c0))
                self._current_rnn_initial_hidden = "tuple_state"
            else:
                # Complex expression (not supported yet)
                self._current_rnn_initial_hidden = None
        else:
            self._current_rnn_initial_hidden = None

        return input_var

    def _handle_lstm_tuple_hx(self, node, module_obj):
        """Handle LSTM case with (h, c) tuple initial state."""
        if len(node.value.args) <= 1 or not isinstance(node.value.args[1], ast.Tuple):
            return
        if not node.value.args[1].elts or not isinstance(node.value.args[1].elts[0], ast.Name):
            return

        h_var = node.value.args[1].elts[0].id
        if h_var in self.module_of_output:
            source_module = self.module_of_output[h_var].replace("__hidden", "").replace("__cell", "")
            module_obj.hx_source = source_module
            source_layer = self._get_layer_by_name(source_module)
            if source_layer:
                source_layer.input_reused = True

    def _handle_simple_hx(self, module_obj):
        """Handle simple case with single hidden state variable."""
        if self._current_rnn_initial_hidden not in self.module_of_output:
            return

        source_module = self.module_of_output[self._current_rnn_initial_hidden].replace("__hidden", "").replace("__cell", "")
        module_obj.hx_source = source_module
        source_layer = self._get_layer_by_name(source_module)
        if source_layer:
            source_layer.input_reused = True

        if source_module.startswith('op_'):
            for mod in self.buml_model.modules:
                if hasattr(mod, 'name') and mod.name == source_module:
                    mod.is_rnn_initial_state = True
                    break

    def _process_rnn_initial_hidden(self, node, module_obj):
        """Process initial hidden state for RNN seq2seq patterns."""
        if not (module_obj and hasattr(self, '_current_rnn_initial_hidden') and self._current_rnn_initial_hidden):
            return

        if self._current_rnn_initial_hidden == "tuple_state":
            self._handle_lstm_tuple_hx(node, module_obj)
        else:
            self._handle_simple_hx(module_obj)

        self._current_rnn_initial_hidden = None

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
        if hasattr(node.value.func, 'value') and isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id != "self":
                self.extract_tensorop(node)
                return

        module_name = node.value.func.attr

        if module_name in self.multi_layer_rnns:
            self.expand_multi_layer_rnn_call(node, module_name, is_tuple=True)
            return

        # Extract input BEFORE _extract_tuple_target_vars which updates module_of_output
        rnn_in = self._extract_rnn_input_arg(node)

        # Check if rnn_in is the input of an identity tensorop (e.g., inp = x)
        # NEW: We need to update rnn_in before storing in layer.input_var
        for module in self.buml_model.modules:
            if (hasattr(module, 'tns_type') and module.tns_type == 'identity' and
                hasattr(module, 'input_var') and module.input_var == rnn_in):
                # Found identity with input_var matching rnn_in
                # Check if rnn_in was reassigned by checking module_of_output
                if rnn_in not in self.module_of_output:
                    # Not reassigned yet (network input), use identity's output
                    rnn_in = module.output_var
                    pass
                break

        self._extract_tuple_target_vars(node, module_name)
        rnn_out = self._determine_rnn_return_type(node, module_name)

        # Extract the actual sequence output variable (first element)
        first_elem = node.targets[0].elts[0]
        if isinstance(first_elem, ast.Name):
            sequence_output_var = first_elem.id
        else:
            sequence_output_var = None

        module_obj = self._get_layer_by_name(module_name)
        if not module_obj:
            module_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)

        # NEW: Set input/output on layer object
        if module_obj:
            module_obj.input_var = rnn_in
            # output_var should be the sequence output (first element), not rnn_out (main output)
            module_obj.output_var = sequence_output_var

        self._process_rnn_initial_hidden(node, module_obj)

        if module_obj and rnn_in in self.module_of_output:
            module_obj.name_module_input = self.module_of_output[rnn_in]

        self.buml_model.modules.append(module_obj)
        self.previous_assign = node

    def handle_forward_variable_assignment(self, node: ast.Assign):
        """
        Handle simple variable assignments like inp = x or r = x (residual).
        Creates an identity TensorOp to preserve the assignment in generated code.

        Parameters:
            node (ast.Assign): The AST node representing the assignment

        Returns:
            None, but creates identity TensorOp and updates tracking
        """
        target_var = node.targets[0].id
        source_var = node.value.id

        # Determine source module
        if source_var in self.module_of_output:
            source_module = self.module_of_output[source_var]
        else:
            # Source is network input
            source_module = 'INPUT'

        # Generate unique name for identity tensorop to avoid overwrites in modules_details
        # When same variable is assigned multiple times (e.g., r = x appears 4 times for residuals),
        if not hasattr(self, '_identity_counter'):
            self._identity_counter = {}
        self._identity_counter[target_var] = self._identity_counter.get(target_var, 0) + 1
        unique_name = f"{target_var}_identity_{self._identity_counter[target_var]}"

        # Create an identity TensorOp to generate the assignment in output code
        # This will output: target_var = source_var
        identity_op = mm_classes.TensorOp(
            name=unique_name,
            tns_type="identity",
            layers_of_tensors=[source_module]
        )

        # Store input/output vars for generator: use target_var as output so all map to same variable
        # NEW: Also set on tensorop object
        identity_op.input_var = source_var
        identity_op.output_var = target_var

        # Track the target variable: map to the unique tensorop name
        self.module_of_output[target_var] = unique_name

        # Mark source as reused (branching point for residual/wide-deep patterns)
        if not hasattr(self, '_variables_saved_for_residual'):
            self._variables_saved_for_residual = set()
        self._variables_saved_for_residual.add(source_var)

        self.buml_model.modules.append(identity_op)
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

    def visit_AugAssign(self, node: ast.AugAssign):
        """
        Handle augmented assignment (in-place operations like +=, -=, *=, /=, //=).
        Converts them to regular assignment: x += y becomes x = x + y

        Parameters:
            node (ast.AugAssign): The augmented assignment node

        Returns:
            None, processes the converted assignment
        """
        # Convert AugAssign to regular Assign
        # x += y  -->  x = x + y

        # Create a BinOp node: x + y
        binop = ast.BinOp(
            left=ast.Name(id=node.target.id, ctx=ast.Load()),
            op=node.op,  # Reuse the same operator (+, -, *, /, //, etc.)
            right=node.value
        )

        # Create an Assign node: x = (x + y)
        assign = ast.Assign(
            targets=[node.target],
            value=binop
        )

        # Copy line number info for error reporting
        assign.lineno = node.lineno
        assign.col_offset = node.col_offset

        # Process as regular assignment
        self.visit_Assign(assign)

    def _get_binop_operation_type(self, binop, node):
        """Map AST binop operation to BUML tensor operation type."""
        op_map = {
            'Add': 'binop_add',
            'Sub': 'binop_subtract',
            'Mult': 'binop_multiply',
            'Div': 'binop_divide',
            'FloorDiv': 'binop_floor_divide'
        }
        op_type_name = binop.op.__class__.__name__
        tns_type = op_map.get(op_type_name)

        if tns_type is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported binary operation '{op_type_name}'. "
                f"Only add, subtract, multiply, divide, and floor divide are supported."
            )
        return tns_type

    def _mark_binop_source_layers_as_reused(self, layer_names):
        """Mark source layers in binop as input_reused to preserve intermediate variables."""
        for layer_name in layer_names:
            if not isinstance(layer_name, str) or isinstance(layer_name, (int, float)):
                continue
            if layer_name.startswith("op_"):
                continue

            base_layer_name = (layer_name.replace('__hidden', '')
                                        .replace('__cell', '')
                                        .replace('__forward', '')
                                        .replace('__backward', ''))
            layer_obj = self._get_layer_by_name(base_layer_name)
            if layer_obj and not hasattr(layer_obj, '_input_reused_marked'):
                layer_obj.input_reused = True
                layer_obj._input_reused_marked = True

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

        left_var = self._extract_binop_operand(binop.left, node, "left")
        if left_var is None:
            return

        right_var = self._extract_binop_operand(binop.right, node, "right")
        if right_var is None:
            return

        tns_type = self._get_binop_operation_type(binop, node)
        if tns_type is None:
            return

        left_layer = left_var if isinstance(left_var, (int, float)) else self.module_of_output.get(left_var)
        right_layer = right_var if isinstance(right_var, (int, float)) else self.module_of_output.get(right_var)

        if left_layer is None or right_layer is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: Cannot determine source layers for binary operation. "
                f"Make sure both operands are defined earlier."
            )
            return

        var_types = self._determine_binop_var_types(left_layer, right_layer, left_var, right_var)

        tensorop_param = {
            "tns_type": tns_type,
            "layers_of_tensors": [left_layer, right_layer],
            "actual_vars": var_types,
            "name": f"op_{self.tensor_op_counter}"
        }

        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1

        output_var = node.targets[0].id
        self.module_of_output[output_var] = tensorop_param["name"]
        # NEW: Also set on tensorop object
        tns_obj.input_var = None  # Binary ops have two inputs tracked via layers_of_tensors
        tns_obj.output_var = output_var
        self.prev_layer_output = output_var

        self._mark_binop_source_layers_as_reused([left_layer, right_layer])

    def _extract_tuple_var_names(self, tuple_target):
        """Extract variable names from tuple unpacking target."""
        var_names = []
        for elt in tuple_target.elts:
            if isinstance(elt, ast.Name):
                var_names.append(elt.id)
            else:
                var_names.append('_')
        return var_names

    def _create_shape_dim_tensorop(self, var_name, idx, source_module):
        """Create TensorOp for shape dimension extraction."""
        tensorop_param = {
            "name": var_name,
            "tns_type": "shape_dim",
            "layers_of_tensors": [source_module],
            "reduce_dim": idx
        }
        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.module_of_output[var_name] = var_name
        # For shape_dim, we need to track what variable the shape is extracted from
        # so the TensorFlow generator can use the correct source variable
        source_var = None
        if source_module == 'INPUT':
            # For INPUT, we need to find the actual input parameter name
            # Look for the first layer's input variable or use 'x' as default
            # for module_name in self.module_of_output.values():
            #         break
            # if source_var is None:
            source_var = 'x'  # Default forward parameter name
        else:
            # CRITICAL FIX: Look up the module's output_var instead of using module name
            # For shape extraction from lstm_out.shape, source_module='lstm' but we need 'lstm_out'
            module_obj = next((obj for obj in self.buml_model.modules if obj.name == source_module), None)
            if module_obj and hasattr(module_obj, 'output_var') and module_obj.output_var:
                source_var = module_obj.output_var
            else:
                source_var = source_module

        # NEW: Set on tensorop object instead
        tns_obj.input_var = source_var
        tns_obj.output_var = var_name

    def handle_forward_shape_unpacking(self, node: ast.Assign):
        """
        Handle tuple unpacking of shape attributes (e.g., b, t, _ = x.shape).
        Creates TensorOp assignments for each shape dimension.

        Parameters:
            node (ast.Assign): The AST node with tuple target and attribute value.

        Returns:
            None, but creates TensorOps for shape dimension extraction.
        """
        tuple_target = node.targets[0]
        var_names = self._extract_tuple_var_names(tuple_target)

        if isinstance(node.value.value, ast.Name):
            source_var = node.value.value.id
            attr_name = node.value.attr

            if attr_name == 'shape':
                # Get source module: if not found and it's 'x', use 'INPUT'
                if source_var in self.module_of_output:
                    source_module = self.module_of_output[source_var]
                elif source_var == 'x':
                    source_module = 'INPUT'
                else:
                    source_module = source_var

                for idx, var_name in enumerate(var_names):
                    if var_name != '_':
                        self._create_shape_dim_tensorop(var_name, idx, source_module)

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
            # CRITICAL: Also update layer object attribute
            lyr_obj.output_var = hidden_var

    def _strip_rnn_suffix(self, layer_name):
        """Strip __hidden or __cell suffix from layer name."""
        if layer_name.endswith("__hidden"):
            return layer_name[:-8]
        elif layer_name.endswith("__cell"):
            return layer_name[:-6]
        return layer_name

    def _get_target_var_for_output(self, result_var, subscripted_var):
        """Determine target variable: preserve user vars, collapse auto-generated temps."""
        if result_var.startswith('_subscript_temp_'):
            return subscripted_var
        else:
            return result_var

    def _handle_rnn_both_outputs(self, node, lyr_obj, prev_module_name, subscripted_var, result_var):
        """Handle RNN with both sequence output and hidden state used."""
        lyr_obj.return_type = "both"

        if prev_module_name.endswith("__hidden"):
            # CRITICAL FIX: Map result_var to base_layer_name (not __hidden suffix)
            # After subscript assignment (x = h[-1]), subsequent layers should use x's current value,
            # not resolve __hidden reference which would use the tuple unpacked variable (h) instead
            base_layer_name = prev_module_name[:-8]  # Strip __hidden suffix
            self.module_of_output[result_var] = base_layer_name
            self.variable_aliases[result_var] = subscripted_var

            target_var = self._get_target_var_for_output(result_var, subscripted_var)
            # NEW: Store on layer object instead
            lyr_obj.hidden_subscript_source = subscripted_var  # The tuple unpacked variable (h1)
            lyr_obj.hidden_subscript_target = target_var        # The subscript result variable (h1_last)
            # DON'T update hidden_state_var here - it should remain the original unpacked variable name
            # hidden_state_var tracks the variable name from tuple unpacking (out, h = rnn(x)),
            # NOT the subscript result assignment (x = h[0])
            # The subscript operation is handled separately as a follow-up assignment

            self.previous_assign = node
            return True
        return False

    def _handle_rnn_hidden_only(self, node, lyr_obj, base_layer_name, prev_module_name, subscripted_var, result_var):
        """Handle RNN with only hidden state used."""
        lyr_obj.return_type = "hidden"
        self._update_hidden_as_output(lyr_obj, base_layer_name)

        if hasattr(lyr_obj, 'bidirectional') and lyr_obj.bidirectional:
            return False

        target_var = self._get_target_var_for_output(result_var, subscripted_var)
        # NEW: Store on layer object instead
        lyr_obj.hidden_subscript_source = subscripted_var  # The tuple unpacked variable (h)
        lyr_obj.hidden_subscript_target = target_var        # The subscript result variable (x)
        # DON'T update hidden_state_var here - it should remain the original unpacked variable name
        # hidden_state_var tracks the variable name from tuple unpacking (_ h = rnn(x)),
        # NOT the subscript result assignment (x = h[-1])
        # The subscript operation is handled separately as a follow-up assignment
        # CRITICAL FIX: Map result_var to base_layer_name (not __hidden suffix)
        # After subscript assignment (x = h[-1]), subsequent layers should use x's current value,
        # not resolve __hidden reference which would use the tuple unpacked variable (h) instead
        self.module_of_output[result_var] = base_layer_name
        self.variable_aliases[result_var] = subscripted_var
        self.previous_assign = node
        return True

    def _determine_rnn_slice_return_type(self, node, lyr_obj, prev_module_name, subscripted_var, result_var):
        """Determine RNN return type based on slicing pattern.

        Args:
            prev_module_name: Module name WITH suffix (e.g., 'rnn1__hidden'), used for module_of_output tracking
        """
        if isinstance(node.value.slice, ast.UnaryOp) or isinstance(node.value.slice, ast.Constant):
            # Handle both negative (UnaryOp) and positive (Constant) integer indices
            base_layer_name = self._strip_rnn_suffix(prev_module_name)

            has_output = base_layer_name in self.rnn_output_vars
            has_hidden = base_layer_name in self.rnn_hidden_vars

            if has_output and has_hidden:
                return self._handle_rnn_both_outputs(node, lyr_obj, prev_module_name, subscripted_var, result_var)
            else:
                return self._handle_rnn_hidden_only(node, lyr_obj, base_layer_name, prev_module_name, subscripted_var, result_var)

        elif isinstance(node.value.slice, ast.Tuple) and len(node.value.slice.elts) == 3:
            if lyr_obj.return_type in ("full", "both"):
                self._handle_non_rnn_slicing(node, subscripted_var, result_var)
                return True
            if lyr_obj.return_type != "both":
                lyr_obj.return_type = "last"
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unrecognized subscript pattern on variable '{subscripted_var}'. "
                f"This may not migrate correctly."
            )
        return False

    def _get_layer_lookup_name(self, module_name):
        """Strip __hidden or __cell suffix from module name."""
        if module_name.endswith("__hidden"):
            return module_name[:-8]
        elif module_name.endswith("__cell"):
            return module_name[:-6]
        return module_name

    def _handle_bidirectional_rnn_slice(self, node, lyr_obj, layer_lookup_name, subscripted_var, result_var):
        """Handle bidirectional RNN h[-2]/h[-1] or h[i] subscript patterns.

        For bidirectional RNNs with num_layers=N:
        - Negative indices: h[-2] = forward, h[-1] = backward (last layer)
        - Positive indices: h[2*N-2] = forward, h[2*N-1] = backward (last layer)
        """
        if not (lyr_obj and hasattr(lyr_obj, 'bidirectional') and lyr_obj.bidirectional):
            return False

        # Handle negative indices: h[-2] (UnaryOp) or positive indices: h[2] (Constant)
        if isinstance(node.value.slice, ast.UnaryOp) and isinstance(node.value.slice.operand, ast.Constant):
            # Negative index case: h[-2] or h[-1]
            slice_idx = -node.value.slice.operand.value if isinstance(node.value.slice.op, ast.USub) else node.value.slice.operand.value
            if slice_idx not in [-2, -1]:
                return False
            suffix = "__forward" if slice_idx == -2 else "__backward"
        elif isinstance(node.value.slice, ast.Constant):
            # Positive index case: h[2] or h[3] for 2-layer bidirectional
            slice_idx = node.value.slice.value

            # Find the base module name to get num_layers
            # layer_lookup_name might be 'lstm_layer_1', need to find 'lstm' in rnn_num_layers
            base_module_name = None
            for module_name in self.rnn_num_layers:
                if layer_lookup_name in self.multi_layer_rnns.get(module_name, []):
                    base_module_name = module_name
                    break

            if base_module_name:
                num_layers = self.rnn_num_layers[base_module_name]
            else:
                num_layers = lyr_obj.num_layers if hasattr(lyr_obj, 'num_layers') else 1

            # Calculate expected indices for last layer
            # For num_layers=N bidirectional: indices are [0,1,...,2N-2,2N-1]
            # Last layer forward = 2*N-2, backward = 2*N-1
            expected_forward_idx = 2 * num_layers - 2
            expected_backward_idx = 2 * num_layers - 1

            if slice_idx == expected_forward_idx:
                suffix = "__forward"
            elif slice_idx == expected_backward_idx:
                suffix = "__backward"
            else:
                return False
        else:
            return False

        self.module_of_output[result_var] = layer_lookup_name + suffix
        self.variable_aliases[result_var] = subscripted_var

        # CRITICAL FIX: Store the user's variable names on layer object
        # so generator uses them instead of generating synthetic names
        if suffix == "__forward":
            lyr_obj.forward_output_var = result_var
        elif suffix == "__backward":
            lyr_obj.backward_output_var = result_var

        self.previous_assign = node
        return True

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

        if subscripted_var not in self.module_of_output:
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        prev_module_name = self.module_of_output[subscripted_var]
        layer_lookup_name = self._get_layer_lookup_name(prev_module_name)
        lyr_obj = self._get_layer_by_name(layer_lookup_name)

        if not lyr_obj or not hasattr(lyr_obj, 'return_type'):
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        if self._determine_rnn_slice_return_type(node, lyr_obj, prev_module_name, subscripted_var, result_var):
            return

        if self._handle_bidirectional_rnn_slice(node, lyr_obj, layer_lookup_name, subscripted_var, result_var):
            return

        self.module_of_output[result_var] = prev_module_name
        self.variable_aliases[result_var] = subscripted_var
        # if result_var.startswith('_subscript_temp_'):
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
        lyr_obj = self._get_layer_by_name(lyr_name)
        if not lyr_obj:
            return
        lyr_type = lyr_obj.__class__.__name__
        if (lyr_type in channels_first_layers and len(self.buml_model.tensor_ops)!=0):

            if (isinstance(self.previous_assign.targets[0], ast.Name) and
                isinstance(self.previous_assign.value, ast.Call)):
                # Safe check: ensure func.value exists and is a Name before accessing .id
                if (hasattr(self.previous_assign.value.func, 'value') and
                    isinstance(self.previous_assign.value.func.value, ast.Name) and
                    self.previous_assign.value.func.value.id != "self"):
                    ops_name = self.previous_assign.value.func.attr
                    if ops_name in ["permute", "transpose"]:
                        # lyr_obj already retrieved at line 1664, use that instead of searching buml_model.layers
                        # (which may not have the layer yet since it's added later in forward() processing)
                        lyr_obj.permute_in = True
                        # Update variable tracking when removing transpose
                        # Map transpose output variable back to transpose input's source layer
                        transpose_output_var = self.previous_assign.targets[0].id
                        transpose_input_var = self.previous_assign.value.func.value.id
                        if transpose_input_var in self.module_of_output:
                            transpose_input_source = self.module_of_output[transpose_input_var]
                            self.module_of_output[transpose_output_var] = transpose_input_source

                            # The current layer (lyr_name) was expecting transpose output, now should use transpose input
                            # CRITICAL: Also update layer object attribute
                            lyr_obj.input_var = transpose_input_var

                            # Update the layer object's name_module_input to point to the source of transpose input
                            if hasattr(lyr_obj, 'name_module_input'):
                                lyr_obj.name_module_input = transpose_input_source

                        self.buml_model.tensor_ops.pop()
                        self.buml_model.modules.pop()


    def _extract_inline_base_var(self, call_node, parent_node):
        """Extract base variable from inline operation, handling chained calls."""
        if isinstance(call_node.func.value, ast.Name):
            return call_node.func.value.id
        elif isinstance(call_node.func.value, ast.Call):
            return self.extract_inline_tensorop(call_node.func.value, parent_node)
        else:
            if hasattr(call_node.func, 'value') and hasattr(call_node.func.value, 'id'):
                return call_node.func.value.id
            return "x"

    def _extract_inline_op_params(self, call_node, op_type):
        """Extract operation parameters for inline TensorOp."""
        if op_type == 'repeat':
            repeat_counts = [self.param_value(arg) for arg in call_node.args]
            return {"repeat_dim": repeat_counts}
        else:
            dim_value = None
            if len(call_node.args) > 0:
                dim_value = self.param_value(call_node.args[0])
            else:
                for kw in call_node.keywords:
                    if kw.arg == "dim":
                        dim_value = self.param_value(kw.value)
            return {"reduce_dim": dim_value}

    def _is_noop_rnn_squeeze_unsqueeze(self, op_type, op_params, base_layer):
        """Check if squeeze(0)/unsqueeze(0) on single-layer RNN is a no-op."""
        if op_type not in ('squeeze', 'unsqueeze'):
            return False
        if op_params.get('reduce_dim') != 0:
            return False
        if not base_layer or not (base_layer.endswith('__hidden') or base_layer.endswith('__cell')):
            return False

        rnn_layer_name = base_layer.replace('__hidden', '').replace('__cell', '')
        rnn_layer = self._get_layer_by_name(rnn_layer_name)
        num_layers = getattr(rnn_layer, 'num_layers', None) if rnn_layer else None
        return (rnn_layer and
                rnn_layer.__class__.__name__ in ('SimpleRNNLayer', 'LSTMLayer', 'GRULayer') and
                (num_layers is None or num_layers == 1))

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
        if not isinstance(call_node.func, ast.Attribute):
            return "x"

        op_type = call_node.func.attr
        base_var = self._extract_inline_base_var(call_node, parent_node)

        if op_type not in ['squeeze', 'unsqueeze', 'repeat']:
            return None if base_var == 'self' else base_var

        base_layer = self.module_of_output.get(base_var)
        if base_layer is None:
            return base_var

        if op_type in ('squeeze', 'unsqueeze'):
            op_params = self._extract_inline_op_params(call_node, op_type)
            if self._is_noop_rnn_squeeze_unsqueeze(op_type, op_params, base_layer):
                output_var = self._extract_output_var_from_target(parent_node.targets[0])
                if output_var and base_layer:
                    self.module_of_output[output_var] = base_layer
                return base_var

        intermediate_var = self.TEMP_INLINE.format(op_type, self.tensor_op_counter)
        tensorop_param = {
            "tns_type": op_type,
            "layers_of_tensors": [base_layer],
            "name": f"op_{self.tensor_op_counter}"
        }
        tensorop_param.update(self._extract_inline_op_params(call_node, op_type))

        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1
        self.module_of_output[intermediate_var] = tensorop_param["name"]

        return intermediate_var

    def _check_and_mark_residual_input(self, tns_obj, call_node):
        """Check if tensorop's input is saved for residual connection and mark it."""
        if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
            source_var = call_node.func.value.id
            if hasattr(self, '_variables_saved_for_residual') and source_var in self._variables_saved_for_residual:
                tns_obj.input_reused = True
                self._variables_saved_for_residual.discard(source_var)

    def _extract_output_var_from_target(self, target):
        """Extract output variable from assignment target."""
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, ast.Tuple):
            first_elem = target.elts[0]
            return first_elem.id if isinstance(first_elem, ast.Name) else None
        return None

    def _create_and_track_split_tensorop(self, tensorop_param, call_node, node):
        """
        Create and track split TensorOp with tuple outputs.

        For: x1, x2 = torch.split(x, 32, dim=1)
        Tracks each output variable with __split_N suffix.
        """
        op_name = f"op_{self.tensor_op_counter}"
        tensorop_param["name"] = op_name
        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)

        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1

        # Extract output variable names from tuple target
        output_vars = []
        for elt in node.targets[0].elts:
            if isinstance(elt, ast.Name):
                output_vars.append(elt.id)

        # Track each output variable with __split_N suffix
        for i, out_var in enumerate(output_vars):
            split_suffix = f"__split_{i}"
            self.module_of_output[out_var] = op_name + split_suffix

        # Extract input variable for the split operation
        input_var = self._extract_tensorop_input_var(call_node, node)

        # Store main operation input/output (for the operation itself)
        # The output is a comma-joined tuple of all output variables
        # main_output = ", ".join(output_vars) if output_vars else None
        # if input_var and main_output:

        # NEW: Set input and output_vars (list) on tensorop object
        tns_obj.input_var = input_var
        tns_obj.output_vars = output_vars  # Set as list, not comma-joined string
        # DO NOT set tns_obj.output_var for split ops - use output_vars instead

        # Update prev_layer_output (use last output variable)
        if output_vars:
            self.prev_layer_output = output_vars[-1]

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
        # Note: We used to skip squeeze(0) on RNN hidden states, but this causes shape mismatches
        # when the squeezed value is later unsqueezed and used. TensorFlow will handle squeeze(0)
        # on [B,H] as a no-op automatically, so we should generate both squeeze and unsqueeze.

        op_name = f"op_{self.tensor_op_counter}"
        tensorop_param["name"] = op_name
        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)

        # Note: is_rnn_initial_state flag is set later when we detect this tensorop
        # is used as hx_source for an RNN layer (see handle_rnn_layer)
        # Don't mark unsqueeze as skippable just because it operates on RNN hidden state

        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1

        self._check_and_mark_residual_input(tns_obj, call_node)

        # Detect parallel operations for TensorOps (similar to layers)
        input_var = self._extract_tensorop_input_var(call_node, node)
        if input_var:
            self._detect_parallel_operations(tns_obj, input_var)

        output_var = self._extract_output_var_from_target(node.targets[0])
        if output_var:
            self.module_of_output[output_var] = op_name
            # This ensures h_squeezed appears in output, not just op_1
            input_var = self._extract_tensorop_input_var(call_node, node)

            # Special handling for concatenate: extract original variable names (not resolved)
            if tns_obj.tns_type == 'concatenate' and hasattr(call_node, 'args') and call_node.args:
                # torch.cat([var1, var2, ...]) - extract vars from the list
                if isinstance(call_node.args[0], ast.List):
                    original_vars = []
                    all_are_names = True
                    for elt in call_node.args[0].elts:
                        if isinstance(elt, ast.Name):
                            original_vars.append(elt.id)
                        else:
                            # Inline call like self.dropout(x), subscript, or other expression
                            all_are_names = False
                            break
                    # CRITICAL FIX: Only set input_var if ALL elements are simple names
                    # If there are inline calls/expressions, let generator resolve from layers_of_tensors
                    if original_vars and all_are_names and len(original_vars) == len(call_node.args[0].elts):
                        input_var = ", ".join(original_vars)  # "last_output, last_hidden"
                    else:
                        # Don't set input_var - generator will resolve from layers_of_tensors
                        input_var = None

            # NEW: Also set on tensorop object
            if input_var is not None:
                tns_obj.input_var = input_var
            tns_obj.output_var = output_var
            # Update prev_layer_output so next operation can detect branching
            self.prev_layer_output = output_var

    def _extract_tensorop_input_var(self, call_node, node):
        """Extract input variable from TensorOp call."""
        # Check if it's a method call (e.g., x.mean())
        if hasattr(call_node, 'func') and hasattr(call_node.func, 'value'):
            if isinstance(call_node.func.value, ast.Name):
                val_id = call_node.func.value.id
                # Skip torch/F/nn as these are library modules, get from args instead
                if val_id not in ['torch', 'F', 'nn', 'functional']:
                    return val_id

        # For library calls (torch.mean(x)) or fallback, get from arguments
        if hasattr(call_node, 'args') and call_node.args:
            if isinstance(call_node.args[0], ast.Name):
                return call_node.args[0].id

        return None

    def _extract_activation_input_var(self, node):
        """Extract input variable from activation function call."""
        if not node.value.args or not isinstance(node.value.args[0], ast.Name):
            self.migration_warnings.append(
                f"Line {node.lineno}: Could not extract input for activation function. Using default input variable 'x'."
            )
            return "x"
        return node.value.args[0].id

    def _check_prev_is_tensorop(self, input_var):
        """Check if previous module is a TensorOp. Returns (is_tensorop, prev_lyr_obj, prev_lyr_name)."""
        if input_var not in self.module_of_output:
            return False, None, None

        prev_lyr_name = self.module_of_output[input_var]
        prev_lyr_obj = self._get_module_by_name(prev_lyr_name)
        prev_cls = prev_lyr_obj.__class__.__name__ if prev_lyr_obj else "None"
        is_tensorop = prev_lyr_obj and prev_cls == "TensorOp"
        return is_tensorop, prev_lyr_obj, prev_lyr_name

    def _create_standalone_functional_activation(self, node, module_name, input_var):
        """Create standalone activation layer for functional API.

        Reuses existing activation layer of same type if found, otherwise creates new one.
        """
        actv_func = actv_fun_mapping[module_name]
        output_var = node.targets[0].id

        # Look for existing activation layer with same activation function
        # Use lowercase activation function name as base layer name (e.g., 'relu', 'gelu')
        base_actv_name = actv_func.lower()
        existing_actv = self._get_layer_by_name(base_actv_name)

        if existing_actv and existing_actv in self.buml_model.modules:
            # Reuse existing activation layer
            import copy
            reused_actv = copy.deepcopy(existing_actv)
            reused_actv.name = base_actv_name  # Reset to base name before adding suffix
            reused_actv.is_layer_call = True
            # Reset permute flags - reused layers shouldn't have format conversion flags
            reused_actv.permute_in = False
            reused_actv.permute_out = False
            self._add_layer_with_tracking(reused_actv)
            # NEW: Also set on layer object
            reused_actv.input_var = input_var
            reused_actv.output_var = output_var
            self.module_of_output[output_var] = reused_actv.name
        else:
            # First occurrence - create new activation layer with clean name
            actv_lyr = mm_classes.GeneralLayer(name=base_actv_name, actv_func=actv_func)
            self._add_layer_with_tracking(actv_lyr)
            # NEW: Also set on layer object
            actv_lyr.input_var = input_var
            actv_lyr.output_var = output_var
            self.module_of_output[output_var] = actv_lyr.name

    def _can_merge_activation(self, layer_obj):
        """Check if a layer supports activation function merging."""
        if not layer_obj:
            return False

        layer_class = layer_obj.__class__.__name__
        parent_class = layer_obj.__class__.mro()[1].__name__ if len(layer_obj.__class__.mro()) > 1 else None

        # TensorFlow doesn't support activation parameter for these layer types:
        # - NormalizationLayer (BatchNorm, LayerNorm)
        # - PoolingLayer
        # - LayerModifier (Dropout)
        # - EmbeddingLayer
        # - FlattenLayer
        non_mergeable_types = ['PoolingLayer', 'EmbeddingLayer', 'FlattenLayer']
        non_mergeable_parents = ['NormalizationLayer', 'LayerModifier']

        return layer_class not in non_mergeable_types and parent_class not in non_mergeable_parents

    def _handle_functional_activation(self, node, module_name):
        """Handle functional API activation functions."""
        input_var = self._extract_activation_input_var(node)
        is_tensorop, prev_lyr_obj, prev_lyr_name = self._check_prev_is_tensorop(input_var)

        # Check if previous layer can merge activation
        can_merge = self._can_merge_activation(prev_lyr_obj) and not is_tensorop

        # If layer already has an activation, don't merge - create standalone instead
        layer_already_has_activation = prev_lyr_obj and hasattr(prev_lyr_obj, 'actv_func') and prev_lyr_obj.actv_func is not None

        if is_tensorop or not can_merge or layer_already_has_activation:
            # Treat as standalone activation
            self._create_standalone_functional_activation(node, module_name, input_var)
        else:
            # Merge into previous layer
            output_var = node.targets[0].id
            if prev_lyr_obj:
                prev_lyr_obj.actv_func = actv_fun_mapping[module_name]
                # Update the layer's output variable to preserve the activation's output variable
                # NEW: Also update attribute on layer object
                prev_lyr_obj.output_var = output_var
            if input_var in self.module_of_output:
                self.module_of_output[output_var] = self.module_of_output[input_var]

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
            self.buml_model.add_layer(lyr_obj)  # This also appends to modules

            # NEW: Set on layer object
            lyr_obj.input_var = node.value.args[0].id
            lyr_obj.output_var = node.targets[0].id
            self.module_of_output[node.targets[0].id] = synthetic_name
            # Note: No need to append to modules here, add_layer already did it
        except ValueError as e:
            self.migration_warnings.append(f"Functional layer '{synthetic_name}': {str(e)}")

    def _try_merge_activation(self, node, module_name, input_source_module=None):
        """Try to merge activation into previous layer. Returns (merged, prev_lyr_name).

        Args:
            node: The AST node for the activation call
            module_name: Name of the activation module
            input_source_module: The module name (with suffix) that produced the input variable.
                                This is captured before module_of_output gets overwritten.
        """
        if hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer:
            return False, None

        # Use input_source_module if provided (captures the layer before overwrite)
        if input_source_module:
            prev_lyr_name_with_suffix = input_source_module
        else:
            # Fallback to old logic for backward compatibility
            if not (self.previous_assign and isinstance(self.previous_assign.value, ast.Call)):
                return False, None

            if not hasattr(self.previous_assign.value.func, 'attr'):
                return False, None

            # Get the output variable from previous assignment
            prev_output_var = self.previous_assign.targets[0].id
            # Look up the actual module name (with suffix) that produced this variable
            prev_lyr_name_with_suffix = self.module_of_output.get(prev_output_var)
            if not prev_lyr_name_with_suffix:
                return False, None

        # Get the module object using the suffixed name
        prev_lyr_obj = self._get_module_by_name(prev_lyr_name_with_suffix)
        if not prev_lyr_obj:
            return False, None

        # Check if it's a layer that can merge activation
        if not self._can_merge_activation(prev_lyr_obj):
            return False, None

        # If layer already has activation, don't merge
        if hasattr(prev_lyr_obj, 'actv_func') and prev_lyr_obj.actv_func is not None:
            return False, None

        actv = self.activation_functions[module_name]
        actv_func = actv_fun_mapping.get(actv)
        if actv_func is None:
            self.migration_warnings.append(
                f"Unsupported activation function '{actv}'. This activation will be skipped in the migration."
            )
            return False, None

        # Merge activation into previous layer
        prev_lyr_obj.actv_func = actv_func
        output_var = node.targets[0].id
        # Point output to the previous layer (skip standalone activation)
        self.module_of_output[output_var] = prev_lyr_name_with_suffix
        return True, prev_lyr_name_with_suffix

    def _create_standalone_activation(self, node, module_name, input_source_module=None):
        """Create standalone activation layer.

        Args:
            node: AST node for the activation call
            module_name: Name of the activation module
            input_source_module: The module that produces the input to this activation.
                                 Must be passed in, as module_of_output may already be overwritten.
        """
        # Use the actual module name from PyTorch to preserve layer names
        actv = self.activation_functions[module_name]
        actv_func = actv_fun_mapping.get(actv)
        if actv_func is None:
            self.migration_warnings.append(
                f"Unsupported activation function '{actv}'. This activation will be skipped in the migration."
            )
            return

        input_var = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else "x"
        output_var = node.targets[0].id

        # If input_source_module wasn't passed, try to get it from module_of_output
        # (though it may already be overwritten by this point)
        if input_source_module is None:
            input_source_module = self.module_of_output.get(input_var)

        # Check if activation layer already exists (layer reuse)
        existing_actv = self._get_layer_by_name(module_name)
        if existing_actv and existing_actv in self.buml_model.modules:
            # This is a reused activation - clone and add with suffix
            import copy
            reused_actv = copy.deepcopy(existing_actv)
            reused_actv.name = module_name  # Reset to base name before adding suffix
            reused_actv.is_layer_call = True
            # Reset permute flags - reused layers shouldn't have format conversion flags
            reused_actv.permute_in = False
            reused_actv.permute_out = False

            # Set name_module_input from the captured input source
            if input_source_module:
                reused_actv.name_module_input = input_source_module

            self._add_layer_with_tracking(reused_actv)
            # NEW: Also set on layer object
            reused_actv.input_var = input_var
            reused_actv.output_var = output_var
            self.module_of_output[output_var] = reused_actv.name
        else:
            # First occurrence - create and add with tracking
            actv_lyr = mm_classes.GeneralLayer(name=module_name, actv_func=actv_func)

            # Set name_module_input from the captured input source
            if input_source_module:
                actv_lyr.name_module_input = input_source_module

            self._add_layer_with_tracking(actv_lyr)
            # NEW: Also set on layer object
            actv_lyr.input_var = input_var
            actv_lyr.output_var = output_var
            self.module_of_output[output_var] = actv_lyr.name

    def _handle_module_activation(self, node, module_name, input_source_module=None):
        """Handle module API activation functions with merge logic."""
        merged, _ = self._try_merge_activation(node, module_name, input_source_module)
        if not merged:
            self._create_standalone_activation(node, module_name, input_source_module)

    def _handle_module_layer_reuse(self, node, module_name, module_obj):
        """
        Handle layer reuse - same layer instance called multiple times.
        Create a new layer in BUML with is_layer_call=True.
        The layer gets a unique suffix from _add_layer_with_tracking.
        """
        # Extract current call's input/output (set by _process_module_api before this)
        # Use object attributes instead
        current_input = module_obj.input_var if hasattr(module_obj, 'input_var') else None
        current_output = module_obj.output_var if hasattr(module_obj, 'output_var') else None

            # Restore object attributes instead

        if not current_input:
            return

        # Clone the original layer and mark it as a layer call (reuse)
        import copy
        call_layer = copy.deepcopy(module_obj)
        # Keep the BASE layer name (without suffix) - _add_layer_with_tracking will add suffix
        call_layer.name = module_name
        # Mark as layer call so generator skips it in __init__
        call_layer.is_layer_call = True
        # Reset permute flags - reused layers shouldn't have format conversion flags
        call_layer.permute_in = False
        call_layer.permute_out = False

        # Add layer with tracking (this adds suffix to name AND adds to both layers and modules)
        self._add_layer_with_tracking(call_layer)

        # Now call_layer.name has suffix, use it for tracking
        # NEW: Also set on layer object
        call_layer.input_var = current_input
        call_layer.output_var = current_output
        self.module_of_output[current_output] = call_layer.name

    def _detect_parallel_operations(self, module_obj, input_var):
        """Detect and mark parallel operations and TensorOp usage."""
        if not module_obj:
            return

        # If input comes from a TensorOp or previous layer output doesn't match current input
        if input_var in self.module_of_output:
            source_module = self.module_of_output[input_var]
            if source_module.startswith("op_"):
                module_obj.input_reused = True
                if self.prev_layer_output and input_var != self.prev_layer_output:
                    module_obj.name_module_input = source_module
            elif self.prev_layer_output and input_var != self.prev_layer_output:
                module_obj.input_reused = True
        # Handle case where input is network input 'x' used by non-first layer
        elif self.prev_layer_output and input_var != self.prev_layer_output:
            module_obj.input_reused = True
            # For network input, set name_module_input to 'INPUT' marker
            if input_var == 'x':
                module_obj.name_module_input = 'INPUT'

    def _check_residual_connection(self, module_obj, input_var):
        """Check and mark residual connections."""
        if hasattr(self, '_variables_saved_for_residual') and input_var in self._variables_saved_for_residual:
            if module_obj:
                module_obj.input_reused = True
            self._variables_saved_for_residual.discard(input_var)

    def _process_module_api(self, node, module_name):
        """Process module API calls (self.layer)."""
        if module_name in self.multi_layer_rnns:
            self.expand_multi_layer_rnn_call(node, module_name, is_tuple=False)
            return

        input_var = self._extract_rnn_input_arg(node)
        output_var = node.targets[0].id

        # Save input source BEFORE overwriting module_of_output (for layer reuse)
        input_source_module = self.module_of_output.get(input_var) if input_var in self.module_of_output else None

        self.module_of_output[output_var] = module_name

        module_obj = self._get_layer_by_name(module_name)
        # NEW: Also set on module object
        # BUT: Don't set if this is a layer reuse (module already in buml_model.modules)
        # because we'll set it on the cloned layer later
        if module_obj and module_obj not in self.buml_model.modules:
            module_obj.input_var = input_var
            module_obj.output_var = output_var
        # Set name_module_input using saved input source (not overwritten value)
        # BUT: Skip activation functions - they're handled in _handle_module_activation
        # and shouldn't be modified here (layer reuse would overwrite the first occurrence)
        if module_obj and input_source_module and module_name not in self.activation_functions:
            module_obj.name_module_input = input_source_module
        self._detect_parallel_operations(module_obj, input_var)
        self.prev_layer_output = node.targets[0].id
        self._check_residual_connection(module_obj, input_var)

        is_subnn_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)

        if module_name in self.activation_functions:
            self._handle_module_activation(node, module_name, input_source_module)
        elif not is_subnn_obj:
            self.is_permute_before_cnn(module_name)

        if module_name not in self.activation_functions:
            module_obj = self._get_layer_by_name(module_name)
            is_subnn = False
            if not module_obj:
                module_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)
                is_subnn = module_obj is not None

            # For layer reuse: clone the layer and add with new suffix
            # The counter suffix ensures unique keys in modules_details
            # Generator strips suffix to reference same layer instance
            # NOTE: SubNNs (Sequential/ModuleList) don't get suffix tracking
            if module_obj and module_obj in self.buml_model.modules and not is_subnn:
                import copy
                # Clone the layer with is_layer_call=True to skip __init__ definition
                reused_layer = copy.deepcopy(module_obj)
                reused_layer.name = module_name  # Reset to base name before adding
                reused_layer.is_layer_call = True
                # Reset permute flags - reused layers shouldn't have format conversion flags
                reused_layer.permute_in = False
                reused_layer.permute_out = False
                # Add with tracking - this adds counter suffix and appends to modules
                self._add_layer_with_tracking(reused_layer)
                # Copy them to the new suffixed name
                # CRITICAL: Also update reused layer object attributes (use values set at line 2401)
                reused_layer.input_var = input_var
                reused_layer.output_var = output_var
                # Note: No need to restore first layer attributes here!
                # The first layer's attributes were set correctly when it was first processed
                # and we prevented overwriting them at line 2422 by checking if module already in buml_model.modules
                # Update module_of_output to point to new suffixed name
                self.module_of_output[node.targets[0].id] = reused_layer.name
            elif module_obj and not is_subnn:
                # Add first occurrence with suffix tracking (only for Layers, not SubNNs)
                self._add_layer_with_tracking(module_obj)
                # CRITICAL: Layer attributes may have been updated (e.g., by is_permute_before_cnn)
                # AFTER they were initially set at line 2441. So we should NOT overwrite them here.
                # The attributes were already set at line 2441, and any subsequent modifications
                # (like transpose removal) should be preserved.
                # NO-OP: Attributes already set correctly
                # Update module_of_output to point to suffixed name
                self.module_of_output[node.targets[0].id] = module_obj.name
            elif is_subnn and module_obj not in self.buml_model.modules:
                # SubNN (Sequential/ModuleList) called in forward - add to modules for generator
                # No suffix tracking needed for SubNNs
                # NEW: Set input/output vars on SubNN object
                module_obj.input_var = input_var
                module_obj.output_var = output_var
                if input_source_module:
                    module_obj.name_module_input = input_source_module
                self._add_module_with_tracking(module_obj)

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
        """Extract transpose operation parameters.

        Detects and skips format conversion transposes that swap dimensions
        to convert from features-last to channels-first format before Conv layers.
        """
        transpose_dim = [op_args[i].value for i in range(len(op_args))]
        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None

        # Check if this is a dimension-swapping transpose for format conversion
        # Pattern: transpose(d1, d2) where d2 = d1 + 1 (swapping adjacent dimensions)
        # Common cases: transpose(1, 2), transpose(2, 3), etc.
        if (len(transpose_dim) == 2 and
            transpose_dim[1] == transpose_dim[0] + 1 and
            transpose_dim[0] > 0):  # Skip batch dimension

            modules = self.buml_model.modules
            prev_module = modules[-1] if modules else None

            if isinstance(prev_module, Layer):
                lyr_type = prev_module.__class__.__name__
                # Transpose after Embedding or pass-through layers (Dropout, etc.)
                # that preserve format is converting to channels-first for PyTorch Conv
                # TensorFlow Conv uses same format, so skip
                # Pass-through layers: Dropout, BatchNorm don't change dimensionality
                if lyr_type in ["EmbeddingLayer", "DropoutLayer", "BatchNormLayer"]:
                    prev_module.permute_out = True
                    return None

        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None
        return {"tns_type": "transpose", "transpose_dim": transpose_dim, "layers_of_tensors": source_layers}

    def _extract_source_layers(self, var_node):
        """Extract source layers from variable node."""
        if not isinstance(var_node, ast.Name):
            return None

        source_var = var_node.id
        if source_var in self.module_of_output:
            return [self.module_of_output[source_var]]
        else:
            return ['INPUT']

    def _process_reshape_arg(self, arg):
        """Process single reshape argument, handling .size() calls and variables."""
        if isinstance(arg, ast.Call) and hasattr(arg.func, 'attr') and arg.func.attr == 'size':
            if len(arg.args) > 0:
                dim_idx = self.param_value(arg.args[0])
                source_layers = self._extract_source_layers(arg.func.value) if hasattr(arg.func, 'value') else None

                op_name = f"op_{self.tensor_op_counter}"
                shape_tensorop_param = {"tns_type": "shape_dim", "reduce_dim": dim_idx,
                                       "layers_of_tensors": source_layers, "name": op_name}
                tns_obj = getattr(mm_classes, "TensorOp")(**shape_tensorop_param)
                self.buml_model.add_tensor_op(tns_obj)
                self.tensor_op_counter += 1

                # This ensures the generator creates an intermediate variable instead of reusing 'x'
                synthetic_var = f"_{op_name}"
                # NEW: Also set on tensorop object
                tns_obj.input_var = None  # Shape extraction from constant or expression
                tns_obj.output_var = synthetic_var

                return op_name
            else:
                return self.param_value(arg)
        elif isinstance(arg, ast.Name) and arg.id in self.module_of_output:
            layer_name = self.module_of_output[arg.id]
            if layer_name.startswith('op_'):
                return layer_name
            else:
                return self.param_value(arg)
        else:
            return self.param_value(arg)

    def _extract_op_reshape(self, call_node, node, op_args):
        """Extract reshape/view operation parameters."""
        reshape_dim = [self._process_reshape_arg(arg) for arg in op_args]
        source_layers = self._extract_source_layers(call_node.func.value) if hasattr(call_node.func, 'value') else None

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

        # Extract source layers to track which variable is being squeezed
        source_layers = self._extract_source_layers(call_node.func.value) if hasattr(call_node.func, 'value') else None
        tensorop_param = {"tns_type": "squeeze", "reduce_dim": squeeze_dim}
        if source_layers:
            tensorop_param["layers_of_tensors"] = source_layers
        return tensorop_param

    def _extract_op_unsqueeze(self, call_node, op_args):
        """Extract unsqueeze operation parameters."""
        unsqueeze_dim = None
        if len(op_args) > 0:
            unsqueeze_dim = self.param_value(op_args[0])
        else:
            for kw in call_node.keywords:
                if kw.arg == "dim":
                    unsqueeze_dim = self.param_value(kw.value)

        # Extract source layers to track which variable is being unsqueezed
        source_layers = self._extract_source_layers(call_node.func.value) if hasattr(call_node.func, 'value') else None
        tensorop_param = {"tns_type": "unsqueeze", "reduce_dim": unsqueeze_dim}
        if source_layers:
            tensorop_param["layers_of_tensors"] = source_layers
        return tensorop_param

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

    def _extract_flatten_dims(self, op_args, keywords):
        """Extract start_dim and end_dim from flatten operation."""
        start_dim = 1
        end_dim = -1

        if len(op_args) > 0:
            start_dim = self.param_value(op_args[0])
        if len(op_args) > 1:
            end_dim = self.param_value(op_args[1])

        for kw in keywords:
            if kw.arg == "start_dim":
                start_dim = self.param_value(kw.value)
            elif kw.arg == "end_dim":
                end_dim = self.param_value(kw.value)

        return start_dim, end_dim

    def _create_flatten_layer(self, layer_name, start_dim, end_dim, name_module_input):
        """Create FlattenLayer with given parameters."""
        flatten_params = {
            "name": layer_name,
            "start_dim": start_dim,
            "end_dim": end_dim,
            "name_module_input": name_module_input
        }
        flatten_layer = getattr(mm_classes, "FlattenLayer")(**flatten_params)
        self.buml_model.add_layer(flatten_layer)
        return flatten_layer

    def _extract_op_flatten(self, call_node, node, op_args):
        """Extract flatten operation and create FlattenLayer (not a TensorOp)."""
        start_dim, end_dim = self._extract_flatten_dims(op_args, call_node.keywords)

        source_var = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
        name_module_input = self.module_of_output.get(source_var, source_var)

        layer_name = f"flatten_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        self._create_flatten_layer(layer_name, start_dim, end_dim, name_module_input)

        if isinstance(node.targets[0], ast.Name):
            output_var = node.targets[0].id
            self.module_of_output[output_var] = layer_name

        return "flatten_created"

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

    def _get_call_node_from_value(self, node_value):
        """Handle .values attribute accessor and return call node."""
        if isinstance(node_value, ast.Attribute) and node_value.attr == 'values':
            return node_value.value
        return node_value

    def _extract_op_type_and_args(self, call_node, node):
        """Extract operation type and arguments from call node."""
        if not hasattr(call_node.func, 'attr'):
            self.migration_warnings.append(
                f"Line {node.lineno}: Unknown tensor operation encountered. This operation will be skipped in the migration."
            )
            return None, None
        return call_node.func.attr, call_node.args

    def _get_op_handler(self, op_type, node):
        """Get operation handler from mapping."""
        handler = self._op_handlers.get(op_type)
        if not handler:
            self.migration_warnings.append(
                f"Unrecognized tensor operation '{op_type}'. This operation will be skipped in the migration."
            )
        return handler

    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """
        call_node = self._get_call_node_from_value(node.value)

        if not isinstance(call_node, ast.Call):
            return

        op_type, op_args = self._extract_op_type_and_args(call_node, node)
        if op_type is None:
            return

        handler = self._get_op_handler(op_type, node)
        if not handler:
            return

        tensorop_param = handler(call_node, node, op_args)

        if tensorop_param == "flatten_created":
            return

        # Handle training-aware dropout as a layer instead of tensorop
        if (tensorop_param and
            tensorop_param.get('tns_type') == 'dropout' and
            tensorop_param.get('dropout_training_aware')):
            self._create_dropout_layer(tensorop_param, node)
            return

        # Handle split/chunk with tuple outputs specially
        if (tensorop_param and tensorop_param.get('tns_type') == 'split' and
            isinstance(node.targets[0], ast.Tuple)):
            self._create_and_track_split_tensorop(tensorop_param, call_node, node)
            return

        if tensorop_param:
            self._create_and_track_tensorop(tensorop_param, call_node, node)


    def _handle_layer_reuse_in_concat(self, layer_obj, arg, layer_name):
        """Handle layer reuse in concatenation."""
        self.layer_reuse_count[layer_name] = self.layer_reuse_count.get(layer_name, 0) + 1
        use_count = self.layer_reuse_count[layer_name]

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
        # NOTE: Don't set reuse_layer.output_var - let generator determine it from input_reused
        reuse_layer.input_var = input_var if 'input_var' in locals() else None
        return temp_name

    def _handle_first_layer_use_in_concat(self, layer_obj, arg, layer_name, is_inline=False):
        """Handle first layer use in concatenation.

        Args:
            layer_obj: The layer object
            arg: The argument AST node
            layer_name: Name of the layer
            is_inline: True if this is an inline layer call (torch.cat([self.fc1(x), ...]))
                      False if layer was already assigned to variable (a=self.fc1(x); torch.cat([a, ...]))
        """
        # Only inline layer calls need separate output variables
        if is_inline:
            layer_obj.input_reused = True

            # Track input for inline concat: use 'INPUT' marker if input is 'x'
            if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
                input_var = arg.args[0].id
                if input_var == 'x':
                    layer_obj.name_module_input = 'INPUT'
                elif input_var in self.module_of_output:
                    layer_obj.name_module_input = self.module_of_output[input_var]
        else:
            # Regular layer call: handle normally
            if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
                input_var = arg.args[0].id
                if input_var in self.module_of_output:
                    layer_obj.name_module_input = self.module_of_output[input_var]
                    layer_obj.input_reused = True

        self.buml_model.modules.append(layer_obj)
        temp_name = self.TEMP_NESTED.format(self.tensor_op_counter)
        self.tensor_op_counter += 1
        self.module_of_output[temp_name] = layer_name
        # NOTE: Don't set layer_obj.output_var - let generator determine it from input_reused
        layer_obj.input_var = input_var if 'input_var' in locals() else None
        return temp_name

    def _extract_layer_call_variable(self, arg, layer_name):
        """Extract variable from layer call in concatenation."""
        layer_obj = self._get_layer_by_name(layer_name)

        if not layer_obj:
            self.migration_warnings.append(
                f"Concatenation operation: Layer '{layer_name}' referenced but not found in the model. Check if it was defined earlier."
            )
            return None

        if layer_obj in self.buml_model.modules:
            return self._handle_layer_reuse_in_concat(layer_obj, arg, layer_name)
        else:
            # This is an inline layer call: torch.cat([self.fc1(x), ...])
            return self._handle_first_layer_use_in_concat(layer_obj, arg, layer_name, is_inline=True)

    def _extract_inline_call_variable(self, arg, node):
        """Extract variable from inline operation call in concatenation."""
        result = self.extract_inline_tensorop(arg, node)
        if result is None:
            self.migration_warnings.append(
                f"Concatenation operation: Could not process inline operation '{ast.unparse(arg)}'. This argument will be skipped."
            )
        return result

    def _extract_subscript_variable(self, arg, node):
        """Extract variable from subscript operation in concatenation."""
        temp_name = self.TEMP_SUBSCRIPT.format(self.tensor_op_counter)
        self.tensor_op_counter += 1
        self.handle_subscript_operation(arg, temp_name, node)
        return temp_name

    def _extract_concat_arg_variable(self, arg, node):
        """Extract variable name from concatenation argument, handling layer calls and inline ops."""
        if isinstance(arg, ast.Name):
            return arg.id
        elif isinstance(arg, ast.Call):
            if (isinstance(arg.func, ast.Attribute) and
                isinstance(arg.func.value, ast.Name) and
                arg.func.value.id == 'self'):
                return self._extract_layer_call_variable(arg, arg.func.attr)
            else:
                # Check for no-op squeeze(0) on single-layer RNN hidden states
                if (isinstance(arg.func, ast.Attribute) and
                    arg.func.attr == 'squeeze' and
                    isinstance(arg.func.value, ast.Name)):
                    base_var = arg.func.value.id
                    # Check if squeeze(0)
                    if (len(arg.args) > 0 and isinstance(arg.args[0], ast.Constant) and
                        arg.args[0].value == 0):
                        # Check if base_var is an RNN hidden state
                        if base_var in self.module_of_output:
                            base_layer = self.module_of_output[base_var]
                            if base_layer and (base_layer.endswith('__hidden') or base_layer.endswith('__cell')):
                                rnn_layer_name = base_layer.replace('__hidden', '').replace('__cell', '')
                                rnn_layer = self._get_layer_by_name(rnn_layer_name)
                                num_layers = getattr(rnn_layer, 'num_layers', None) if rnn_layer else None
                                if (rnn_layer and
                                    rnn_layer.__class__.__name__ in ('SimpleRNNLayer', 'LSTMLayer', 'GRULayer') and
                                    (num_layers is None or num_layers == 1)):
                                    # No-op squeeze: return base variable directly
                                    return base_var
                return self._extract_inline_call_variable(arg, node)
        elif isinstance(arg, ast.Subscript):
            # CRITICAL FIX: Check if subscript is on RNN hidden state with return_type='hidden'
            # In TF, hidden states from return_state=True are 2D tensors (no time dimension)
            # So subscripts like h[-1] are no-ops and should be skipped
            if isinstance(arg.value, ast.Name):
                base_var = arg.value.id
                if base_var in self.module_of_output:
                    module_name = self.module_of_output[base_var]
                    if module_name and module_name.endswith('__hidden'):
                        rnn_layer_name = module_name.replace('__hidden', '')
                        rnn_layer = self._get_layer_by_name(rnn_layer_name)
                        # Check if this RNN has return_type='hidden' (only hidden state, no sequence)
                        if rnn_layer and hasattr(rnn_layer, 'return_type') and rnn_layer.return_type == 'hidden':
                            # Skip subscript - return base variable directly
                            return base_var
            return self._extract_subscript_variable(arg, node)
        else:
            return None

    def _is_bidirectional_hidden_rnn(self, source_layer):
        """Check if layer is a bidirectional RNN with hidden return type."""
        return (source_layer and
                hasattr(source_layer, 'bidirectional') and source_layer.bidirectional and
                hasattr(source_layer, 'return_type') and source_layer.return_type in ('hidden', 'both'))

    def _extract_subscript_indices(self, ops_args):
        """Extract indices from subscript operations."""
        indices = []
        for arg in ops_args:
            if isinstance(arg.slice, ast.UnaryOp) and isinstance(arg.slice.op, ast.USub):
                indices.append(-arg.slice.operand.value)
            elif isinstance(arg.slice, ast.Constant):
                indices.append(arg.slice.value)
            else:
                return None
        return indices

    def _handle_bidirectional_concat_output(self, node, source_layer_name):
        """Track output variable for bidirectional concat to preserve original name."""
        output_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if output_var:
            # Store the original PyTorch concat variable name so it can be used
            # by the RNN layer as its hidden output variable
            concat_module = "bidirectional_concat_" + source_layer_name
            self.module_of_output[output_var] = concat_module

            # Mark that this layer should use the original concat variable name
            # This will be picked up when generating layer details
            if not hasattr(self, '_bidirectional_concat_var_names'):
                self._bidirectional_concat_var_names = {}
            self._bidirectional_concat_var_names[source_layer_name] = output_var
        return output_var is not None

    def _check_bidirectional_rnn_concat(self, ops_args, layers_of_tensors, node):
        """Check if this is a bidirectional RNN concatenation pattern and handle it."""
        # Strip __forward/__backward suffixes for comparison
        base_layers = [name.replace("__forward", "").replace("__backward", "") for name in layers_of_tensors]

        if not (len(ops_args) == 2 and len(set(base_layers)) == 1):
            return False

        # Use the first base layer name (without suffixes) as source
        source_layer_name = base_layers[0]
        # But check if we need to keep the original for proper layer lookup
        # If the original has no suffix, use it; otherwise use base
        orig_name = layers_of_tensors[0]
        if not (orig_name.endswith("__forward") or orig_name.endswith("__backward")):
            source_layer_name = orig_name

        source_layer = self._get_layer_by_name(source_layer_name)

        if not self._is_bidirectional_hidden_rnn(source_layer):
            return False

        if all(isinstance(arg, ast.Subscript) for arg in ops_args):
            indices = self._extract_subscript_indices(ops_args)
            if indices:
                # Accept negative indices (-2, -1) or positive indices for last layer
                # For num_layers=N bidirectional: last layer is [2*N-2, 2*N-1]
                # Find base module name to get num_layers
                base_module_name = None
                for module_name in self.rnn_num_layers:
                    if source_layer_name in self.multi_layer_rnns.get(module_name, []):
                        base_module_name = module_name
                        break

                if base_module_name:
                    num_layers = self.rnn_num_layers[base_module_name]
                else:
                    num_layers = source_layer.num_layers if hasattr(source_layer, 'num_layers') else 1

                expected_positive_indices = {2 * num_layers - 2, 2 * num_layers - 1}

                if set(indices) == {-2, -1} or set(indices) == expected_positive_indices:
                    return self._handle_bidirectional_concat_output(node, source_layer_name)

        # For name-based patterns (h_forward, h_backward), also skip to avoid duplicate concat
        # The template automatically generates concat for bidirectional RNN hidden states
        if all(isinstance(arg, ast.Name) for arg in ops_args):
            return self._handle_bidirectional_concat_output(node, source_layer_name)

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

    def _update_prev_layer_return_type_if_subscript(self, ops_args):
        """Update previous layer return_type if first arg is subscript."""
        if isinstance(ops_args[0], ast.Subscript):
            if (self.previous_assign and
                hasattr(self.previous_assign, 'value') and
                isinstance(self.previous_assign.value, ast.Call) and
                hasattr(self.previous_assign.value, 'func') and
                hasattr(self.previous_assign.value.func, 'attr')):
                prev_lyr_name = self.previous_assign.value.func.attr
                lyr_obj = self._get_layer_by_name(prev_lyr_name)
                if lyr_obj:
                    lyr_obj.return_type = "hidden"

    def _extract_concat_variables(self, ops_args, node):
        """Extract all variables from concatenation arguments."""
        variables = []
        for arg in ops_args:
            var = self._extract_concat_arg_variable(arg, node)
            if var is None:
                self.migration_warnings.append(
                    f"Line {node.lineno}: Cannot extract variable from concatenation argument. Make sure all concat arguments are valid variables or layer outputs."
                )
                return None
            variables.append(var)
        return variables

    def _resolve_concat_variables(self, variables):
        """Resolve variable aliases to actual variables."""
        actual_vars = []
        for var in variables:
            actual_var = var if var in self.module_of_output else self.variable_aliases.get(var, var)
            actual_vars.append(actual_var)
        return actual_vars

    def _is_bidirectional_rnn_subscript_concat(self, ops_args):
        """Check if this is a bidirectional RNN concat with subscript args (h[-2], h[-1])."""
        if not (len(ops_args) == 2 and
                all(isinstance(arg, ast.Subscript) for arg in ops_args)):
            return False

        # Both subscripts should be on the same variable
        if not (isinstance(ops_args[0].value, ast.Name) and
                isinstance(ops_args[1].value, ast.Name) and
                ops_args[0].value.id == ops_args[1].value.id):
            return False

        var_name = ops_args[0].value.id
        if var_name not in self.module_of_output:
            return False

        source_module = self.module_of_output[var_name]
        # Strip __hidden suffix if present
        if source_module.endswith("__hidden"):
            source_module = source_module[:-8]

        source_layer = self._get_layer_by_name(source_module)
        if not self._is_bidirectional_hidden_rnn(source_layer):
            return False

        # Check indices are [-2, -1]
        indices = self._extract_subscript_indices(ops_args)
        return indices and set(indices) == {-2, -1}

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

        # Early check for bidirectional RNN concat to avoid creating subscript tensorops
        is_bidir_subscript = self._is_bidirectional_rnn_subscript_concat(ops_args)
        if is_bidir_subscript:
            # Skip creating TensorOp: template already generates bidirectional concat
            # Track output variable to use as the concat result variable name
            output_var = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
            if output_var:
                var_name = ops_args[0].value.id
                source_module = self.module_of_output.get(var_name, "NOT_FOUND")
                base_module = source_module.replace("__hidden", "")

                # Store the original PyTorch concat variable name so it can be used
                # by the RNN layer as its hidden output variable
                concat_module = "bidirectional_concat_" + base_module
                self.module_of_output[output_var] = concat_module

                # Mark that this layer should use the original concat variable name
                # This will be picked up when generating layer details
                if not hasattr(self, '_bidirectional_concat_var_names'):
                    self._bidirectional_concat_var_names = {}
                self._bidirectional_concat_var_names[base_module] = output_var
            return None

        self._update_prev_layer_return_type_if_subscript(ops_args)

        variables = self._extract_concat_variables(ops_args, node)
        if variables is None:
            return None

        actual_vars = self._resolve_concat_variables(variables)

        layers_of_tensors = [self.module_of_output[actual_var] for actual_var in actual_vars]
        cat_dim = self.param_value(node.value.keywords[0].value)

        if self._check_bidirectional_rnn_concat(ops_args, layers_of_tensors, node):
            return None

        # Check for duplicate sources from bidirectional RNN hidden states
        # This happens when h_f = h_n[-2]; h_b = h_n[-1]; h = torch.cat([h_f, h_b])
        # Both h_f and h_b point to the same __hidden module
        if (len(layers_of_tensors) == 2 and
            layers_of_tensors[0] == layers_of_tensors[1] and
            layers_of_tensors[0].endswith("__hidden")):
            # Check if source is a bidirectional RNN
            base_layer_name = layers_of_tensors[0].replace("__hidden", "")
            source_layer = self._get_layer_by_name(base_layer_name)
            if self._is_bidirectional_hidden_rnn(source_layer):
                # Let the RNN template handle the concat - don't create a TensorOp
                # But register the output variable so it can be used in later operations
                output_var = self._extract_output_var_from_target(node.targets[0])
                if output_var:
                    concat_module = "bidirectional_concat_" + base_layer_name
                    self.module_of_output[output_var] = concat_module
                    if not hasattr(self, '_bidirectional_concat_var_names'):
                        self._bidirectional_concat_var_names = {}
                    self._bidirectional_concat_var_names[base_layer_name] = output_var
                return None

        var_types = self._determine_rnn_var_types(layers_of_tensors, actual_vars)

        result = {
            "tns_type": "concatenate",
            "layers_of_tensors": layers_of_tensors,
            "concatenate_dim": cat_dim,
            "actual_vars": var_types
        }
        return result


    def _extract_permute_dimensions(self, ops_args):
        """Extract permute dimensions from operation arguments."""
        permute_dim = []
        for arg in ops_args:
            if isinstance(arg, ast.Constant):
                permute_dim.append(arg.value)
            elif isinstance(arg, ast.Num):  # Python < 3.8
                permute_dim.append(arg.n)
        return permute_dim

    def extract_tensorop_permute(self, ops_args):
        """
        It extracts the permute tensorop information.

        Returns:
            The tensorop parameters.
        """
        modules = self.buml_model.modules
        prev_module = modules[-1] if modules else None

        if isinstance(prev_module, Layer):
            lyr_type = prev_module.__class__.__name__
            if lyr_type in channels_first_layers:
                prev_module.permute_out = True
                return None

        permute_dim = self._extract_permute_dimensions(ops_args)
        return {"tns_type": "permute", "permute_dim": permute_dim}

    def _extract_op_interpolate(self, call_node, node, ops_args):
        """Extract F.interpolate operation parameters."""
        size = None
        scale_factor = None
        mode = 'nearest'

        # Extract from keywords
        for kw in call_node.keywords:
            if kw.arg == 'size':
                if isinstance(kw.value, ast.Constant):
                    size = kw.value.value
                elif isinstance(kw.value, (ast.Tuple, ast.List)):
                    size = tuple(elt.value if isinstance(elt, ast.Constant) else None
                               for elt in kw.value.elts)
            elif kw.arg == 'scale_factor':
                if isinstance(kw.value, ast.Constant):
                    scale_factor = kw.value.value
            elif kw.arg == 'mode':
                if isinstance(kw.value, ast.Constant):
                    mode = kw.value.value

        return {
            "tns_type": "interpolate",
            "interpolate_size": size,
            "interpolate_scale": scale_factor,
            "interpolate_mode": mode
        }

    def _extract_op_pad(self, call_node, node, ops_args):
        """Extract F.pad operation parameters."""
        pad = None
        mode = 'constant'
        value = 0

        # Second positional arg is pad tuple (first is input)
        if len(ops_args) > 1:
            pad_arg = ops_args[1]
            if isinstance(pad_arg, (ast.Tuple, ast.List)):
                pad = tuple(elt.value if isinstance(elt, ast.Constant) else 0
                          for elt in pad_arg.elts)

        # Extract from keywords
        for kw in call_node.keywords:
            if kw.arg == 'mode':
                if isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            elif kw.arg == 'value':
                if isinstance(kw.value, ast.Constant):
                    value = kw.value.value

        return {
            "tns_type": "pad",
            "pad_amount": pad,
            "pad_mode": mode,
            "pad_value": value
        }

    def _extract_op_dropout(self, call_node, node, ops_args):
        """Extract F.dropout operation parameters."""
        p = 0.5
        training_aware = False

        # Extract from keywords
        for kw in call_node.keywords:
            if kw.arg == 'p':
                if isinstance(kw.value, ast.Constant):
                    p = kw.value.value
            elif kw.arg == 'training':
                # F.dropout with training parameter (e.g., training=self.training)
                # means dropout should be training-aware
                training_aware = True

        return {
            "tns_type": "dropout",
            "dropout_rate": p,
            "dropout_training_aware": training_aware
        }

    def _create_dropout_layer(self, tensorop_param, node):
        """Create a Dropout layer from F.dropout with training parameter."""
        # Generate layer name following convention (dropout_0, dropout_1, etc.)
        dropout_name = f"dropout_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        # Get input variable
        input_var = None
        if isinstance(node.value, ast.Call) and node.value.args:
            if isinstance(node.value.args[0], ast.Name):
                input_var = node.value.args[0].id

        # Determine input source module
        name_module_input = None
        if input_var and input_var in self.module_of_output:
            name_module_input = self.module_of_output[input_var]

        # Create DropoutLayer using BUML
        layer_obj = getattr(mm_classes, "DropoutLayer")(
            name=dropout_name,
            rate=tensorop_param.get('dropout_rate', 0.5)
        )

        # Set input source if available
        if name_module_input:
            layer_obj.name_module_input = name_module_input

        # Add layer to model and tracking
        self._add_layer_with_tracking(layer_obj)

        # Track output mapping
        output_var = self._extract_output_var_from_target(node.targets[0])
        if output_var:
            self.module_of_output[output_var] = dropout_name
            self.prev_layer_output = output_var

    def _extract_op_zeros_like(self, call_node, op_args):
        """Extract zeros_like operation parameters."""
        source_var = None

        # Extract source variable from first argument
        if len(op_args) > 0 and isinstance(op_args[0], ast.Name):
            source_var = op_args[0].id

        # Determine source layers
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        return {
            "tns_type": "zeros_like",
            "layers_of_tensors": source_layers,
            "input_reused": True  # Force new variable name to avoid overwriting previous outputs
        }

    def _extract_op_split(self, call_node, node, op_args):
        """Extract split operation parameters.

        Handles: x1, x2 = torch.split(x, split_size, dim=1)
        or: x1, x2 = torch.split(x, [size1, size2], dim=1)

        Note: torch.split(x, split_size_or_sections, dim) where:
        - If split_size_or_sections is int: splits into chunks of that size
        - If split_size_or_sections is list: splits into chunks with those sizes
        """
        source_var = None
        split_size = None
        split_dim = 0  # Default dimension
        num_outputs = 1

        # Extract source variable from first argument
        if len(op_args) > 0 and isinstance(op_args[0], ast.Name):
            source_var = op_args[0].id

        # Count number of output variables from assignment target (x1, x2 = ...)
        if isinstance(node.targets[0], ast.Tuple):
            num_outputs = len(node.targets[0].elts)

        # Extract split size from second argument
        if len(op_args) > 1:
            if isinstance(op_args[1], ast.Constant):
                # PyTorch: split_size (chunk size), but we need num_splits for TF
                # Use num_outputs from assignment to determine this
                split_size = num_outputs
            elif isinstance(op_args[1], ast.List):
                # Handle list of split sizes [size1, size2, ...]
                split_size = [elt.value for elt in op_args[1].elts if isinstance(elt, ast.Constant)]

        # Extract dim from keyword arguments or third positional argument
        if hasattr(call_node, 'keywords'):
            for kw in call_node.keywords:
                if kw.arg == 'dim' and isinstance(kw.value, ast.Constant):
                    split_dim = kw.value.value

        if len(op_args) > 2 and isinstance(op_args[2], ast.Constant):
            split_dim = op_args[2].value

        # Determine source layers
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        return {
            "tns_type": "split",
            "layers_of_tensors": source_layers,
            "split_dim": split_dim,
            "split_sizes": split_size,
            "input_reused": True  # Force new variable names for split outputs
        }

    def _extract_op_chunk(self, call_node, node, op_args):
        """Extract chunk operation parameters.

        Handles: x1, x2, x3 = torch.chunk(x, 3, dim=1)

        Note: torch.chunk(x, chunks, dim) where:
        - chunks: number of chunks to split into
        - dim: dimension along which to split
        """
        source_var = None
        num_chunks = None
        split_dim = 0  # Default dimension

        # Extract source variable from first argument
        if len(op_args) > 0 and isinstance(op_args[0], ast.Name):
            source_var = op_args[0].id

        # Extract number of chunks from second argument
        if len(op_args) > 1 and isinstance(op_args[1], ast.Constant):
            num_chunks = op_args[1].value

        # Extract dim from keyword arguments or third positional argument
        if hasattr(call_node, 'keywords'):
            for kw in call_node.keywords:
                if kw.arg == 'dim' and isinstance(kw.value, ast.Constant):
                    split_dim = kw.value.value

        if len(op_args) > 2 and isinstance(op_args[2], ast.Constant):
            split_dim = op_args[2].value

        # Determine source layers
        if source_var and source_var in self.module_of_output:
            source_layers = [self.module_of_output[source_var]]
        elif source_var:
            source_layers = ['INPUT']
        else:
            source_layers = None

        # torch.chunk(x, N, dim) splits tensor into N chunks
        # This maps to split with num_splits = N
        return {
            "tns_type": "split",
            "layers_of_tensors": source_layers,
            "split_dim": split_dim,
            "split_sizes": num_chunks,
            "input_reused": True  # Force new variable names for chunk outputs
        }

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

    def visit_Expr(self, node: ast.Expr):
        """
        Visit expression statements to extract epochs from train function calls.
        We see train(model, train_loader, criterion, optimizer, 10) and need to extract 10.
        The train function signature is: train(model, train_loader, criterion, optimizer, epochs)
        So epochs is at index 4 (5th parameter).
        """
        if (hasattr(self, 'inside_nn_class') and self.inside_nn_class) or \
           (hasattr(self, 'inside_method') and self.inside_method):
            self.generic_visit(node)
            return

        # Check if it's a function call to train/train_model
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            if func_name in ['train', 'train_model']:
                # Check keyword argument first (explicit epochs=N)
                for keyword in node.value.keywords:
                    if keyword.arg == "epochs" and isinstance(keyword.value, ast.Constant):
                        self.data_config["config"]["epochs"] = keyword.value.value
                        break

                # If not found in keywords, check positional arguments
                # Standard signature: train(model, train_loader, criterion, optimizer, epochs)
                # epochs is the 5th parameter (index 4)
                if "epochs" not in self.data_config["config"] and len(node.value.args) >= 5:
                    epochs_arg = node.value.args[4]
                    if isinstance(epochs_arg, ast.Constant):
                        self.data_config["config"]["epochs"] = epochs_arg.value

        self.generic_visit(node)



def transform_layer(lyr_type: str, lyr_params: dict,
                    lyr_name: str | None = None):
    """
    It transforms layers and their params from PyTorch to BUML.

    Parameters:
        lyr_type (str): The type of the layer (PyTorch).
        lyr_params (dict): A dictionary storing the layer parameters and
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
        lyr_params (dict): A dictionary storing the layer parameters and
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

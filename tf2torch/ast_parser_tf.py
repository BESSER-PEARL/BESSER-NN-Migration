"""
Module to extract information from the AST of a neural network
written in TensorFlow and transforms it to a BUML model.
It also extracts data and model configuration attributes.
"""


import ast
import copy
import sys
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
import besser.BUML.metamodel.nn as mm_classes
from besser.BUML.metamodel.nn import NN, Layer, TensorOp
from ast_parser_nn import ASTParser
from definitions import (
    layers_mapping, params_mapping, static_params, rnn_layers,
    pos_params, int2list_params, lyrs_of_int2list_params, loss_func_mapping,
    excluded_params, actv_fun_mapping
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
            # Extract dimension from class name (Conv1D yields 1D)
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


def infer_layernorm_params(buml_model):
    """
    Infer LayerNormalization normalized_shape from previous layers.

    LayerNorm normalizes over the last dimension(s). We infer the shape
    from the output of the previous layer.

    Parameters:
        buml_model: The BUML model with already added layers.

    Returns:
        dict with 'normalized_shape' param as a list.
    """
    # Search backwards through layers for the most recent layer with output shape
    for layer in reversed(buml_model.layers):
        layer_class = layer.__class__.__name__

        # For RNN layers, use hidden_size
        if layer_class in ["SimpleRNNLayer", "LSTMLayer", "GRULayer"]:
            return {"normalized_shape": [layer.hidden_size]}

        # For Linear/Dense layers, use out_features
        if layer_class == "LinearLayer":
            return {"normalized_shape": [layer.out_features]}

        # For Conv layers, use out_channels
        if layer_class in ["Conv1D", "Conv2D", "Conv3D"]:
            return {"normalized_shape": [layer.out_channels]}

    # Default: will be inferred dynamically at runtime
    return {"normalized_shape": [1]}


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

    # Template names for temporary variables
    TEMP_SUBSCRIPT_OP = "_subscript_op_{}"
    TEMP_SUBSCRIPT = "_subscript_temp_{}"
    TEMP_BINOP = "_binop_temp_{}"
    TEMP_INLINE = "_inline_{}_{}"
    TEMP_NESTED = "_nested_temp_{}"

    def __init__(self, input_nn_type: str, only_nn:bool):
        super().__init__(input_nn_type, only_nn)

        self.padding_amount: int | None = None
        # Track RNN output vs hidden variables separately
        self.rnn_output_vars = {}  # {module_name: output_var}
        self.rnn_hidden_vars = {}  # {module_name: hidden_var}
        # Track variable aliases and layer lookups
        self.variable_aliases = {}  # {alias_var: source_var}
        self.layer_by_name = {}  # {layer_name: layer_obj} for O(1) lookup
        self.module_by_name = {}  # {module_name: module_obj} for O(1) lookup of layers and sub_nns
        self.layer_reuse_count = {}  # {layer_name: reuse_count} for layer reuse tracking
        # Track standalone activation layers (layers.ReLU, layers.Activation, etc.)
        self.activation_functions = {}  # {module_name: actv_func_name}
        # Counter for adding unique suffix to ALL layer names
        self.layer_counter = 0

    def _get_layer_by_name(self, layer_name):
        """Get layer by name using O(1) lookup, fallback to linear search if not in dict."""
        if layer_name in self.layer_by_name:
            return self.layer_by_name[layer_name]
        # Fallback to linear search and update dict (for layers added before tracking)
        layer_obj = next((obj for obj in self.buml_model.layers if obj.name == layer_name), None)
        if layer_obj:
            self.layer_by_name[layer_name] = layer_obj
        return layer_obj

    def _add_layer_with_tracking(self, layer_obj):
        """
        Add layer to model with counter suffix and update lookup dict for O(1) access.
        All layers get a suffix like _c1, _c2, etc. to ensure unique keys in modules_details.
        The generator strips this suffix when creating layer syntax and uses is_layer_call
        to skip duplicate definitions in __init__.
        """
        if hasattr(layer_obj, 'name'):
            base_name = layer_obj.name
            # Store base name for lookups (before adding suffix)
            if base_name not in self.layer_by_name:
                self.layer_by_name[base_name] = layer_obj

            # Add suffix to ALL layers using _c prefix to distinguish from other _N suffixes
            self.layer_counter += 1
            layer_obj.name = f"{base_name}_c{self.layer_counter}"

            # Store in lookup dicts using the SUFFIXED name
            self.layer_by_name[layer_obj.name] = layer_obj

        self.buml_model.add_layer(layer_obj)

    def _add_module_with_tracking(self, module_obj):
        """Add module (layer or sub_nn) to model and update lookup dict for O(1) access."""
        self.buml_model.modules.append(module_obj)
        if hasattr(module_obj, 'name'):
            self.module_by_name[module_obj.name] = module_obj

    def visit_AugAssign(self, node: ast.AugAssign):
        """
        Handle augmented assignment (in-place operations like +=, -=, *=, /=, //=).
        Converts them to regular assignment: x += y becomes x = x + y

        This enables support for residual connections and other in-place operations.
        """
        # Only process if we're inside a class definition
        if not self.in_class:
            return

        # Create BinOp node: x + y
        binop = ast.BinOp(
            left=ast.Name(id=node.target.id, ctx=ast.Load()),
            op=node.op,
            right=node.value
        )

        # Create Assign node: x = (x + y)
        assign = ast.Assign(targets=[node.target], value=binop)
        assign.lineno = node.lineno
        assign.col_offset = node.col_offset

        # Process as regular assignment
        self.visit_Assign(assign)

    def _get_tf_binop_operation_type(self, binop_node, node):
        """Map Python AST binop to BUML tensor operation type."""
        op_map = {
            'Add': 'binop_add',
            'Sub': 'binop_subtract',
            'Mult': 'binop_multiply',
            'Div': 'binop_divide',
            'FloorDiv': 'binop_floor_divide'
        }
        op_type = binop_node.op.__class__.__name__

        if op_type not in op_map:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported binary operation '{op_type}'. This operation will be skipped."
            )
            return None
        return op_map[op_type]

    def _mark_residual_source_layers(self, tns_type, left_layer, right_layer):
        """Mark source layers for residual connections (addition operations)."""
        if tns_type != "binop_add":
            return
        if not isinstance(left_layer, str) or not isinstance(right_layer, str):
            return

        for layer_name in [left_layer, right_layer]:
            if layer_name and not isinstance(layer_name, (int, float)):
                layer_obj = self._get_layer_by_name(layer_name)
                if layer_obj:
                    layer_obj.input_reused = True

    def handle_forward_binop(self, node: ast.Assign):
        """
        Handle binary operations in forward method (e.g., x + y, x * 2, a // b).
        Creates TensorOp objects for arithmetic operations.
        """
        binop_node = node.value
        tns_type = self._get_tf_binop_operation_type(binop_node, node)
        if tns_type is None:
            return

        left_var = self._extract_binop_operand(binop_node.left, node, "left")
        right_var = self._extract_binop_operand(binop_node.right, node, "right")

        if left_var is None or right_var is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: Could not extract operands for binary operation. Operation skipped."
            )
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

        target_var = node.targets[0].id
        tensorop_param = {
            "name": f"op_{self.tensor_op_counter}",
            "tns_type": tns_type,
            "layers_of_tensors": [left_layer, right_layer],
            "actual_vars": var_types
        }
        self.tensor_op_counter += 1

        try:
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.module_of_output[target_var] = tensorop_param["name"]

            tns_obj.input_var = left_var if isinstance(left_var, str) else str(left_var)
            tns_obj.output_var = target_var
            self.prev_layer_output = target_var

            self._mark_residual_source_layers(tns_type, left_layer, right_layer)

        except Exception as e:
            self.migration_warnings.append(
                f"Line {node.lineno}: Failed to create binary operation TensorOp: {str(e)}"
            )

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

    def _determine_concat_var_types(self, lyr1, lyr2, var1, var2):
        """Determine var types (output/hidden) for RNN concatenation operands."""
        # Resolve variable aliases
        actual_var1 = self.variable_aliases.get(var1, var1)
        actual_var2 = self.variable_aliases.get(var2, var2)

        var_types = []

        # Check first variable
        if lyr1 in self.rnn_hidden_vars and actual_var1 == self.rnn_hidden_vars[lyr1]:
            var_types.append("hidden")
        elif lyr1 in self.rnn_output_vars and actual_var1 == self.rnn_output_vars[lyr1]:
            var_types.append("output")
        else:
            var_types.append("output")  # Default to output

        # Check second variable
        if lyr2 in self.rnn_hidden_vars and actual_var2 == self.rnn_hidden_vars[lyr2]:
            var_types.append("hidden")
        elif lyr2 in self.rnn_output_vars and actual_var2 == self.rnn_output_vars[lyr2]:
            var_types.append("output")
        else:
            var_types.append("output")  # Default to output

        return var_types

    def _determine_rnn_var_types_multi(self, layers_of_tensors, actual_vars):
        """Determine var types for N RNN concatenation operands."""
        var_types = []
        for lyr_name, actual_var in zip(layers_of_tensors, actual_vars):
            # Check if lyr_name has __hidden or __cell suffix
            if lyr_name.endswith("__hidden"):
                var_types.append("hidden")
            elif lyr_name.endswith("__cell"):
                var_types.append("hidden")  # Cell states are also hidden states
            elif lyr_name in self.rnn_hidden_vars and actual_var == self.rnn_hidden_vars[lyr_name]:
                var_types.append("hidden")
            elif lyr_name in self.rnn_output_vars and actual_var == self.rnn_output_vars[lyr_name]:
                var_types.append("output")
            else:
                var_types.append("output")
        return var_types

    def _is_bidirectional_hidden_concat(self, layers_of_tensors, var_types):
        """Check if concat is for bidirectional RNN hidden states (already handled by template)."""
        # Must have exactly 2 operands
        if len(layers_of_tensors) != 2:
            return False

        # Both must be hidden states
        if not all(vtype == "hidden" for vtype in var_types):
            return False

        # Both must come from the same base layer
        base_layer1 = layers_of_tensors[0].replace("__hidden", "").replace("__cell", "")
        base_layer2 = layers_of_tensors[1].replace("__hidden", "").replace("__cell", "")

        if base_layer1 != base_layer2:
            return False

        # Check if the layer is bidirectional
        lyr_obj = next((obj for obj in self.buml_model.modules if obj.name == base_layer1), None)
        if lyr_obj and hasattr(lyr_obj, 'bidirectional') and lyr_obj.bidirectional:
            return True

        return False

    def _handle_module_layer_reuse(self, node, module_name, module_obj, input_var, output_var):
        """
        Handle layer reuse by cloning the layer with is_layer_call=True.
        This marks it to skip __init__ definition but be called in forward.
        Now receives input_var and output_var for THIS use of the reused layer.
        """
        import copy

        # Clone the layer with is_layer_call=True to skip __init__ definition
        reused_layer = copy.deepcopy(module_obj)
        reused_layer.name = module_name  # Reset to base name before adding suffix
        reused_layer.is_layer_call = True
        # Set the input/output vars for THIS reused instance
        reused_layer.input_var = input_var
        reused_layer.output_var = output_var
        # Add with tracking - this adds counter suffix and appends to modules
        self._add_layer_with_tracking(reused_layer)
        # Set module input and mark split reuse
        self._set_module_input(reused_layer, input_var)
        self._mark_split_input_reuse(reused_layer, input_var)
        # Update module_of_output to point to new suffixed name
        self.module_of_output[node.targets[0].id] = reused_layer.name

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
        return temp_name

    def _handle_first_layer_use_in_concat(self, layer_obj, arg, layer_name):
        """Handle first layer use in concatenation."""
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

    def _extract_layer_call_variable(self, arg, layer_name):
        """Extract variable from layer call in concatenation - returns INLINE_CALL marker."""
        # Get input variable
        if len(arg.args) > 0 and isinstance(arg.args[0], ast.Name):
            input_var = arg.args[0].id
        else:
            input_var = "x"

        # Check if layer exists
        layer_obj = next((lyr for lyr in self.buml_model.layers if lyr.name == layer_name), None)
        if not layer_obj:
            self.migration_warnings.append(
                f"Concatenation operation: Layer '{layer_name}' referenced but not found in model."
            )
            return None

        # Add layer to modules so it appears in __init__
        # Mark as inline_only so it's not executed separately in forward
        if layer_obj not in self.buml_model.modules:
            layer_obj.inline_only = True  # Flag for generator
            self.buml_model.modules.append(layer_obj)

        return f"INLINE_CALL:{layer_name}:{input_var}"

    def _extract_inline_call_variable(self, arg, node):
        """Extract variable from inline operation call in concatenation."""
        result = self.extract_inline_tensorop(arg, node)
        if result is None:
            self.migration_warnings.append(
                f"Concatenation operation: Could not process inline operation."
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
                return self._extract_inline_call_variable(arg, node)
        elif isinstance(arg, ast.Subscript):
            return self._extract_subscript_variable(arg, node)
        else:
            return None

    def _extract_concat_variables(self, ops_args, node):
        """Extract all variables from concatenation arguments."""
        variables = []
        for arg in ops_args:
            var = self._extract_concat_arg_variable(arg, node)
            if var is None:
                self.migration_warnings.append(
                    f"Line {node.lineno}: Cannot extract variable from concatenation argument."
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

    def _extract_binop_operand(self, operand_node, node, side):
        """
        Extract operand from binary operation. Handles:
        - Variable names (ast.Name)
        - Numeric constants (ast.Constant, ast.Num)
        - Inline tensor operations (ast.Call) - e.g., x.squeeze(0)
        - Subscript operations (ast.Subscript) - e.g., x[:, -1]
        - Nested binary operations (ast.BinOp) - e.g., (a + b)

        Parameters:
            operand_node: AST node representing the operand
            node: Parent assignment node (for line number tracking)
            side: "left" or "right" (for warning messages)

        Returns:
            str or number: Variable name, numeric constant, or temp variable name
        """
        if isinstance(operand_node, ast.Name):
            # Variable reference
            return operand_node.id
        elif isinstance(operand_node, ast.Constant):
            # Numeric constant
            return operand_node.value
        elif isinstance(operand_node, ast.Num):  # Python 3.7 compatibility
            return operand_node.n
        elif isinstance(operand_node, ast.Call):
            # Inline tensor operation (e.g., x.squeeze(0))
            return self.extract_inline_tensorop(operand_node, node)
        elif isinstance(operand_node, ast.Subscript):
            # Subscript operation (e.g., x[:, last_index])
            temp_name = self.TEMP_SUBSCRIPT.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            self.handle_subscript_operation(operand_node, temp_name, node)
            return temp_name
        elif isinstance(operand_node, ast.BinOp):
            # Nested binary operation: recursively handle it
            temp_name = self.TEMP_BINOP.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            temp_target = ast.Name(id=temp_name, ctx=ast.Store())
            binop_assign = ast.Assign(targets=[temp_target], value=operand_node)
            binop_assign.lineno = node.lineno
            binop_assign.col_offset = node.col_offset
            self.handle_forward_binop(binop_assign)
            return temp_name
        else:
            # Unsupported operand type
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported operand type '{operand_node.__class__.__name__}' on {side} side of binary operation."
            )
            return None

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

        if op_type not in ['squeeze', 'unsqueeze', 'repeat', 'expand_dims']:
            if base_var == 'self':
                return None
            return base_var

        base_layer = self.module_of_output.get(base_var)
        if base_layer is None:
            return base_var

        # Map TensorFlow operation names to PyTorch equivalents
        tns_type = 'unsqueeze' if op_type == 'expand_dims' else op_type

        intermediate_var = self.TEMP_INLINE.format(tns_type, self.tensor_op_counter)
        tensorop_param = {
            "tns_type": tns_type,
            "layers_of_tensors": [base_layer],
            "name": f"op_{self.tensor_op_counter}"
        }
        tensorop_param.update(self._extract_inline_op_params(call_node, op_type))

        tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.tensor_op_counter += 1
        self.module_of_output[intermediate_var] = tensorop_param["name"]

        return intermediate_var

    def _extract_inline_base_var(self, call_node, parent_node):
        """Extract base variable from inline operation, handling chained calls."""
        if isinstance(call_node.func.value, ast.Name):
            # Check if this is a functional call (tf.operation) or method call (tensor.operation)
            if call_node.func.value.id == 'tf':
                # Functional call like tf.squeeze(out, axis=1): tensor is in args[0]
                if call_node.args:
                    first_arg = call_node.args[0]
                    if isinstance(first_arg, ast.Name):
                        return first_arg.id
                    elif isinstance(first_arg, ast.Call):
                        # Nested inline: tf.squeeze(tf.expand_dims(h, axis=1), axis=1)
                        return self.extract_inline_tensorop(first_arg, parent_node)
                return "x"
            else:
                # Method call like out.squeeze(1): tensor is the object
                return call_node.func.value.id
        elif isinstance(call_node.func.value, ast.Call):
            return self.extract_inline_tensorop(call_node.func.value, parent_node)
        else:
            if hasattr(call_node.func, 'value') and hasattr(call_node.func.value, 'id'):
                return call_node.func.value.id
            return "x"

    def _extract_inline_op_params(self, call_node, op_type):
        """Extract operation parameters for inline TensorOp."""
        # Check if this is a TensorFlow functional call
        is_tf_functional = (isinstance(call_node.func, ast.Attribute) and
                           isinstance(call_node.func.value, ast.Name) and
                           call_node.func.value.id == 'tf')

        if op_type == 'repeat':
            # For tf.tile(tensor, multiples=[...]), multiples is in keywords
            if is_tf_functional:
                for kw in call_node.keywords:
                    if kw.arg == 'multiples':
                        if isinstance(kw.value, ast.List):
                            repeat_counts = [self.param_value(e) for e in kw.value.elts]
                            return {"repeat_dim": repeat_counts}
                return {"repeat_dim": []}
            else:
                # Method call: tensor.repeat(dim0, dim1, ...)
                repeat_counts = [self.param_value(arg) for arg in call_node.args]
                return {"repeat_dim": repeat_counts}
        else:
            # squeeze/unsqueeze operations
            dim_value = None
            if is_tf_functional:
                # TensorFlow: tf.squeeze(tensor, axis=1) or tf.expand_dims(tensor, axis=1)
                # Dimension is in keywords as 'axis'
                for kw in call_node.keywords:
                    if kw.arg == "axis":
                        dim_value = self.param_value(kw.value)
            else:
                # Method call: tensor.squeeze(1) or tensor.unsqueeze(1)
                # Dimension is in args[0]
                if len(call_node.args) > 0:
                    dim_value = self.param_value(call_node.args[0])
                else:
                    for kw in call_node.keywords:
                        if kw.arg == "dim":
                            dim_value = self.param_value(kw.value)
            return {"reduce_dim": dim_value}

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
        subscript_assign = self._create_temp_assignment(temp_name, subscript_node, node)

        if self._is_rnn_subscript(subscripted_var):
            self.handle_forward_slicing(subscript_assign)
        else:
            subscript_pattern = self.extract_subscript_pattern(subscript_node)
            self.create_subscript_tensorop(subscripted_var, subscript_pattern, temp_name)

        self.previous_assign = subscript_assign

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
                # Multi-dimensional slicing like [:, last_index, :]
                elements = [slice_to_string(elt) for elt in slice_node.elts]
                return ", ".join(elements)
            else:
                # Single index like last_index or 0
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
        if source_module.endswith("__hidden"):
            source_module = source_module[:-8]
        elif source_module.endswith("__cell"):
            source_module = source_module[:-6]

        subscript_op = self._create_subscript_op(op_name, source_module, subscript_pattern)
        self.buml_model.modules.append(subscript_op)
        subscript_op.input_var = resolved_var
        subscript_op.output_var = output_var
        self.module_of_output[output_var] = op_name

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

    def handle_forward_variable_assignment(self, node: ast.Assign):
        """
        Handle simple variable assignments like inp = x or r = x (residual).
        Creates an identity TensorOp to preserve the assignment in generated code.

        Parameters:
            node (ast.Assign): The AST node representing the assignment

        Returns:
            None, but creates a TensorOp and updates tracking dictionaries
        """
        target_var = node.targets[0].id
        source_var = node.value.id

        # Create an identity operation to preserve this assignment
        # This ensures "residual = x" becomes "residual = x" in PyTorch
        from besser.BUML.metamodel.nn import TensorOp

        # Create identity tensorop with layers_of_tensors to track source
        source_module = self.module_of_output.get(source_var, 'INPUT')
        identity_op = TensorOp(
            name=target_var,
            tns_type='identity',
            layers_of_tensors=[source_module],
            input_reused=True  # Mark as reusing input so BESSER uses tensorop name as output var
        )

        self.buml_model.modules.append(identity_op)
        self.module_of_output[target_var] = target_var  # Identity op outputs to target_var
        identity_op.input_var = source_var
        identity_op.output_var = target_var

        # Track that this variable has been saved for later use (residual connection)
        if not hasattr(self, '_variables_saved_for_residual'):
            self._variables_saved_for_residual = set()
        self._variables_saved_for_residual.add(source_var)

        self.previous_assign = node

    def _configure_rnn_module(self, module_obj, module_name, rnn_in):
        """Configure RNN module with hx_source and input_reused settings."""
        if not module_obj:
            return

        if hasattr(self, '_current_rnn_initial_hidden_source') and self._current_rnn_initial_hidden_source:
            module_obj.hx_source = self._current_rnn_initial_hidden_source

        if hasattr(self, 'prev_layer_output') and self.prev_layer_output:
            if rnn_in != self.prev_layer_output:
                module_obj.input_reused = True

        dropout_layer_name = f"{module_name}_dropout"
        dropout_module = next((obj for obj in self.buml_model.layers if obj.name == dropout_layer_name), None)
        if dropout_module:
            self.buml_model.modules.append(dropout_module)

    def _append_rnn_module(self, module_obj, node, module_name):
        """Append RNN module to buml_model, handling reuse if needed."""
        if module_obj and module_obj in self.buml_model.modules:
            self._handle_module_layer_reuse(node, module_name, module_obj)
        elif module_obj:
            self.buml_model.modules.append(module_obj)

    def handle_forward_tuple_assignment(self, node: ast.Assign):
        """
        Handle RNN tuple assignments such as:
        - out, h = self.rnn(x)  # SimpleRNN/GRU with return_state=True
        - out, h, c = self.lstm(x)  # LSTM with return_state=True

        Parameters:
            node (ast.Assign): The AST node representing the tuple assignment

        Returns:
            None, but populates the BUML model and tracks RNN variables
        """
        if hasattr(node.value.func, 'value') and isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id != "self":
                self.extract_tensorop(node)
                return

        module_name = node.value.func.attr
        module_obj = self._get_layer_by_name(module_name)

        self._extract_tuple_target_vars(node, module_name, module_obj)
        rnn_out = self._determine_rnn_output_var(node)
        rnn_in = self._extract_rnn_input_arg(node)

        if module_obj:
            module_obj.input_var = rnn_in
            module_obj.output_var = rnn_out

        self._configure_rnn_module(module_obj, module_name, rnn_in)
        self._append_rnn_module(module_obj, node, module_name)

        self.prev_layer_output = rnn_out
        self.previous_assign = node

    def _get_var_id(self, target_elt):
        """Safely extract variable ID from AST target element."""
        return target_elt.id if isinstance(target_elt, ast.Name) else None

    def _track_rnn_hidden_var(self, var, module_name):
        """Track RNN hidden state variable."""
        if var and var != "_":
            self.module_of_output[var] = module_name + "__hidden"
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.hidden_state_var = var
                lyr_obj.hidden_unused = False
        elif var == "_":
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.hidden_state_var = "_"
                lyr_obj.hidden_unused = True

    def _track_rnn_output_var(self, var, module_name):
        """Track RNN output variable."""
        if var and var != "_":
            self.rnn_output_vars[module_name] = var
            self.module_of_output[var] = module_name

    def _track_rnn_hidden_var(self, var, module_name):
        """Track RNN hidden state variable."""
        if var and var != "_":
            self.rnn_hidden_vars[module_name] = var
            self.module_of_output[var] = module_name + "__hidden"
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.hidden_state_var = var
                lyr_obj.hidden_unused = False
        elif var == "_":
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.hidden_state_var = "_"
                lyr_obj.hidden_unused = True

    def _track_lstm_cell_var(self, var, module_name):
        """Track LSTM cell state variable."""
        if var and var != "_":
            self.module_of_output[var] = module_name + "__cell"
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.cell_state_var = var
                lyr_obj.cell_unused = False
        elif var == "_":
            lyr_obj = self._get_layer_by_name(module_name)
            if lyr_obj:
                lyr_obj.cell_state_var = "_"
                lyr_obj.cell_unused = True

    def _handle_2_target_rnn(self, node, module_name):
        """Handle SimpleRNN or GRU: out, h = self.rnn(x)"""
        var1 = self._get_var_id(node.targets[0].elts[0])
        var2 = self._get_var_id(node.targets[0].elts[1])

        self._track_rnn_output_var(var1, module_name)
        self._track_rnn_hidden_var(var2, module_name)

    def _handle_bidirectional_rnn_3_targets(self, node, module_name, module_obj):
        """Handle Bidirectional RNN: out, forward_h, backward_h = self.encoder(x)"""
        var1, var2, var3 = (self._get_var_id(node.targets[0].elts[i]) for i in range(3))

        self._track_rnn_output_var(var1, module_name)

        uses_separate_components = (var2 and var2 != "_") or (var3 and var3 != "_")
        if uses_separate_components and module_obj:
            module_obj.skip_hidden_concat = True

        if var2 and var2 != "_":
            self.module_of_output[var2] = module_name + "__hidden_forward"
        if var3 and var3 != "_":
            self.module_of_output[var3] = module_name + "__hidden_backward"

    def _handle_lstm_3_targets(self, node, module_name):
        """Handle LSTM: out, h, c = self.lstm(x)"""
        var1, var2, var3 = (self._get_var_id(node.targets[0].elts[i]) for i in range(3))

        self._track_rnn_output_var(var1, module_name)
        self._track_rnn_hidden_var(var2, module_name)
        self._track_lstm_cell_var(var3, module_name)

    def _handle_bidirectional_lstm_5_targets(self, node, module_name, module_obj):
        """Handle Bidirectional LSTM: out, fwd_h, fwd_c, bwd_h, bwd_c = self.bilstm(x)"""
        vars = [self._get_var_id(node.targets[0].elts[i]) for i in range(5)]
        var1, var2, var3, var4, var5 = vars

        self._track_rnn_output_var(var1, module_name)

        uses_separate_components = (var2 and var2 != "_") or (var4 and var4 != "_")
        if uses_separate_components and module_obj:
            module_obj.skip_hidden_concat = True

        if var2 and var2 != "_":
            self.module_of_output[var2] = module_name + "__hidden_forward"
        if var4 and var4 != "_":
            self.module_of_output[var4] = module_name + "__hidden_backward"
        if var3 and var3 != "_":
            self.module_of_output[var3] = module_name + "__cell_forward"
        if var5 and var5 != "_":
            self.module_of_output[var5] = module_name + "__cell_backward"

    def _extract_tuple_target_vars(self, node, module_name, module_obj=None):
        """Extract and track output/hidden variables from tuple assignment targets."""
        num_targets = len(node.targets[0].elts)

        if num_targets == 2:
            self._handle_2_target_rnn(node, module_name)
        elif num_targets == 3:
            is_bidirectional = module_obj and hasattr(module_obj, 'bidirectional') and module_obj.bidirectional
            if is_bidirectional:
                self._handle_bidirectional_rnn_3_targets(node, module_name, module_obj)
            else:
                self._handle_lstm_3_targets(node, module_name)
        elif num_targets == 5:
            self._handle_bidirectional_lstm_5_targets(node, module_name, module_obj)

    def _determine_rnn_output_var(self, node):
        """Determine which variable holds the main RNN output."""
        # In TensorFlow, the first element is always the output sequence
        # Second (and third for LSTM) are hidden states
        first_elem = node.targets[0].elts[0]

        if isinstance(first_elem, ast.Name) and first_elem.id != "_":
            return first_elem.id
        else:
            # If first element is _, check second element (hidden state becomes output)
            if len(node.targets[0].elts) > 1:
                second_elem = node.targets[0].elts[1]
                if isinstance(second_elem, ast.Name) and second_elem.id != "_":
                    return second_elem.id

        return "x"  # Fallback

    def _extract_hidden_var_from_initial_state(self, kw_value):
        """Extract hidden state variable from initial_state argument."""
        if isinstance(kw_value, ast.List) and len(kw_value.elts) >= 1:
            return kw_value.elts[0].id if isinstance(kw_value.elts[0], ast.Name) else None
        elif isinstance(kw_value, ast.Name):
            return kw_value.id
        return None

    def _track_initial_state_source(self, h_var, node):
        """Track source layer for RNN initial_state and add migration warning."""
        if not h_var or h_var not in self.module_of_output:
            return

        source_module = self.module_of_output[h_var].replace("__hidden", "").replace("__cell", "")
        self._current_rnn_initial_hidden_source = source_module
        self.migration_warnings.append(
            f"Line {node.lineno}: RNN called with initial_state from layer '{source_module}'. "
            f"This will be migrated to PyTorch using the corresponding hidden state variable."
        )

    def _extract_rnn_input_arg(self, node):
        """
        Extract input argument from RNN call, handling both variables and inline operations.

        Also detects initial hidden state parameter for seq2seq/decoder patterns.
        Returns: input_var (str)
        Side effect: Sets self._current_rnn_initial_hidden if initial_state is provided
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

        # Check for initial_state keyword argument (TensorFlow pattern)
        self._current_rnn_initial_hidden = None
        self._current_rnn_initial_hidden_source = None
        for kw in node.value.keywords:
            if kw.arg == "initial_state":
                h_var = self._extract_hidden_var_from_initial_state(kw.value)
                self._track_initial_state_source(h_var, node)
                break

        # Check for second positional argument (less common in TF but possible)
        if len(node.value.args) > 1:
            self.migration_warnings.append(
                f"Line {node.lineno}: RNN called with additional positional arguments. "
                f"TensorFlow Keras RNNs typically use initial_state as a keyword argument. "
                f"Manual review recommended."
            )

        return input_var

    def _handle_tf_shape_slicing(self, node: ast.Assign):
        """Handle tf.shape(x)[dim] pattern to create shape_dim tensorop."""
        result_var = node.targets[0].id

        # Extract dimension index from subscript
        if isinstance(node.value.slice, ast.Constant):
            dim_idx = node.value.slice.value
        elif isinstance(node.value.slice, ast.Index) and isinstance(node.value.slice.value, ast.Constant):
            dim_idx = node.value.slice.value.value
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported slice type in tf.shape subscript"
            )
            return

        # Extract source tensor from tf.shape() arguments
        if len(node.value.value.args) == 0:
            self.migration_warnings.append(
                f"Line {node.lineno}: tf.shape() call missing argument"
            )
            return

        source_arg = node.value.value.args[0]
        if not isinstance(source_arg, ast.Name):
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported source type in tf.shape()"
            )
            return

        source_var = source_arg.id
        # Determine source layers
        source_layers = [self.module_of_output[source_var]] if source_var in self.module_of_output else ['INPUT']

        # Create shape_dim tensorop
        shape_tensorop_param = {
            "tns_type": "shape_dim",
            "reduce_dim": dim_idx,
            "layers_of_tensors": source_layers,
            "name": result_var
        }
        tns_obj = getattr(mm_classes, "TensorOp")(**shape_tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.module_of_output[result_var] = result_var

    def _is_tf_shape_slicing(self, node: ast.Assign):
        """Check if this is a tf.shape(x)[dim] pattern."""
        return (isinstance(node.value.value, ast.Call) and
                hasattr(node.value.value.func, 'attr') and
                node.value.value.func.attr == 'shape' and
                hasattr(node.value.value.func, 'value') and
                isinstance(node.value.value.func.value, ast.Name) and
                node.value.value.func.value.id == 'tf')

    def _get_layer_lookup_name(self, module_name):
        """Strip __hidden or __cell suffix from module name."""
        if module_name.endswith("__hidden"):
            return module_name[:-8]
        elif module_name.endswith("__cell"):
            return module_name[:-6]
        return module_name

    def _process_rnn_slicing(self, node, subscripted_var, result_var, prev_module_name):
        """Process slicing on RNN layer outputs."""
        layer_lookup_name = self._get_layer_lookup_name(prev_module_name)
        lyr_obj = self._get_layer_by_name(layer_lookup_name)

        if not lyr_obj or not hasattr(lyr_obj, 'return_type'):
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        self._handle_non_rnn_slicing(node, subscripted_var, result_var)

        if result_var not in self.module_of_output:
            self.module_of_output[result_var] = prev_module_name
        self.variable_aliases[result_var] = subscripted_var

    def handle_forward_slicing(self, node: ast.Assign):
        """
        Handle slicing operations such as:
        - x = out[:, -1]  # Last timestep of RNN output
        - x = out[:, -1, :]  # Last timestep with explicit feature dimension
        - x = tensor[0]  # Regular tensor slicing
        - batch_size = tf.shape(x)[0]  # Shape dimension extraction

        Parameters:
            node (ast.Assign): The AST node representing the slicing assignment

        Returns:
            None, but populates the BUML model
        """
        if self._is_tf_shape_slicing(node):
            self._handle_tf_shape_slicing(node)
            return

        subscripted_var = node.value.value.id if isinstance(node.value.value, ast.Name) else None
        result_var = node.targets[0].id

        if not subscripted_var or subscripted_var not in self.module_of_output:
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        prev_module_name = self.module_of_output[subscripted_var]
        self._process_rnn_slicing(node, subscripted_var, result_var, prev_module_name)
        self.previous_assign = node

    def _handle_non_rnn_slicing(self, node, subscripted_var, result_var):
        """Create subscript TensorOp for non-RNN or regular slicing operations."""
        subscript_pattern = self.extract_subscript_pattern(node.value)
        self.create_subscript_tensorop(subscripted_var if subscripted_var else "unknown",
                                      subscript_pattern, result_var)

    def _is_timestep_selection(self, node):
        """Check if this is a timestep selection pattern like [:, -1] or [:, -1, :]."""
        slice_node = node.value.slice

        # Check for patterns like [:, -1] or [:, -1, :]
        if isinstance(slice_node, ast.Tuple):
            # Multi-dimensional slicing
            for i, elt in enumerate(slice_node.elts):
                if isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                    # Negative index like -1
                    if isinstance(elt.operand, ast.Constant) and elt.operand.value == 1:
                        return True
                elif isinstance(elt, ast.Constant) and elt.value == -1:
                    return True

        # Single dimension slicing like [-1]
        if isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.USub):
            if isinstance(slice_node.operand, ast.Constant) and slice_node.operand.value == 1:
                return True
        elif isinstance(slice_node, ast.Constant) and slice_node.value == -1:
            return True

        return False

    def _handle_init_activation_layer(self, module_name, lyr_type, node):
        """Handle standalone activation layers in __init__."""
        if lyr_type == "Activation":
            if len(node.value.args) > 0 and isinstance(node.value.args[0], ast.Constant):
                actv_name = node.value.args[0].value
            else:
                actv_name = "relu"
            self.activation_functions[module_name] = actv_name
        else:
            self.activation_functions[module_name] = actv_fun_mapping[lyr_type]

    def _extract_bidirectional_layer(self, node, lyr_type, lyr_params):
        """Extract bidirectional RNN layer parameters."""
        if len(node.value.args) > 0:
            if (isinstance(node.value.args[0], ast.Call) and
                node.value.func.attr == "Bidirectional"):
                lyr = node.value.args[0]
                lyr_type, lyr_params = self.extract_layer(lyr)
                lyr_params["bidirectional"] = True
        return lyr_type, lyr_params

    def _add_init_layer_with_params(self, lyr_type, lyr_params, module_name, dropout_rate):
        """Add layer with optional dropout and parameter inference."""
        if dropout_rate is not None:
            dropout_layer = getattr(mm_classes, "DropoutLayer")(
                name=f"{module_name}_dropout",
                rate=dropout_rate
            )
            self.buml_model.add_layer(dropout_layer)

        if lyr_type == "BatchNormLayer":
            lyr_params.update(infer_batchnorm_params(self.buml_model))
        elif lyr_type == "LayerNormLayer":
            lyr_params.update(infer_layernorm_params(self.buml_model))

        buml_layer = getattr(mm_classes, lyr_type)(**lyr_params)
        self.buml_model.add_layer(buml_layer)

    def handle_init(self, node: ast.Assign):
        """
        It retrieves the sub_nn layers and stores them in the 'sub_nn'
        dict. It also retrieves the layers and their parameters and
        stores them in the 'layers' dict.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        module_name = node.targets[0].attr

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "Sequential":
                self.handle_sequential_layers(node, module_name)

        elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if hasattr(node.value.func, 'attr') and node.value.func.attr == "Sequential":
                self.handle_sequential_layers(node, module_name)
            else:
                lyr_type, lyr_params = self.extract_layer(node.value)

                # Skip unsupported Model constructor (Functional API)
                if lyr_type == "Model":
                    return

                if lyr_type in actv_fun_mapping:
                    self._handle_init_activation_layer(module_name, lyr_type, node)
                else:
                    lyr_type, lyr_params = self._extract_bidirectional_layer(node, lyr_type, lyr_params)

                    lyr_type, lyr_params, padding_amount, dropout_rate, has_recurrent_dropout = transform_layer(
                        lyr_type, lyr_params, self.padding_amount, module_name
                    )
                    self.padding_amount = padding_amount

                    if has_recurrent_dropout:
                        self.migration_warnings.append(
                            f"TensorFlow layer '{module_name}' has recurrent_dropout parameter, "
                            f"which has no equivalent in PyTorch LSTM/GRU/SimpleRNN. This parameter is not migrated."
                        )

                    if not lyr_type.startswith("ZeroPadding"):
                        self._add_init_layer_with_params(lyr_type, lyr_params, module_name, dropout_rate)

        self.buml_model.modules.clear()


    def _flatten_nested_sequential(self, seq_name, layer_id, elt, subnn):
        """Process and flatten a nested Sequential layer."""
        nested_seq_name = f"{seq_name}_nested_{layer_id}"
        nested_node = ast.Assign(targets=[ast.Name(id=nested_seq_name)], value=elt)
        self.handle_sequential_layers(nested_node, nested_seq_name)

        nested_subnn = next((obj for obj in self.buml_model.sub_nns if
                            obj.name == nested_seq_name), None)
        if nested_subnn:
            for nested_layer in nested_subnn.layers:
                nested_layer.name = f"layer_{layer_id}"
                subnn.add_layer(nested_layer)
                layer_id += 1
            self.buml_model.sub_nns.remove(nested_subnn)
        return layer_id

    def _handle_sequential_activation(self, lyr_type, layer_id, subnn):
        """Handle activation layer in Sequential model."""
        actv_func = actv_fun_mapping[lyr_type]
        if subnn.layers and hasattr(subnn.layers[-1], 'actv_func'):
            subnn.layers[-1].actv_func = actv_func
            return layer_id
        else:
            actv_layer = getattr(mm_classes, "GeneralLayer")(
                name=f"layer_{layer_id}",
                actv_func=actv_func
            )
            subnn.add_layer(actv_layer)
            return layer_id + 1

    def _extract_bidirectional_params(self, elt, lyr_type, lyr_params):
        """Extract parameters for bidirectional RNN layers."""
        if len(elt.args) > 0:
            if (isinstance(elt.args[0], ast.Call) and
                isinstance(elt.func, ast.Attribute) and
                elt.func.attr == "Bidirectional"):
                lyr = elt.args[0]
                lyr_type, lyr_params = self.extract_layer(lyr)
                lyr_params["bidirectional"] = True
        return lyr_type, lyr_params

    def _add_sequential_layer(self, lyr_type, lyr_params, layer_id, dropout_rate, subnn):
        """Add a layer to the Sequential subnn, including optional dropout."""
        if dropout_rate is not None:
            dropout_layer = getattr(mm_classes, "DropoutLayer")(
                name=f"layer_{layer_id}",
                rate=dropout_rate
            )
            subnn.add_layer(dropout_layer)
            layer_id += 1

        lyr_params["name"] = f"layer_{layer_id}"

        if lyr_type == "BatchNormLayer":
            lyr_params.update(infer_batchnorm_params(subnn))
        elif lyr_type == "LayerNormLayer":
            lyr_params.update(infer_layernorm_params(subnn))

        subnn_layer = getattr(mm_classes, lyr_type)(**lyr_params)
        subnn.add_layer(subnn_layer)
        return layer_id + 1

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

        for elt in node.value.args[0].elts:
            if isinstance(elt, ast.Call):
                lyr_type, lyr_params = self.extract_layer(elt)

                if lyr_type == "Sequential":
                    layer_id = self._flatten_nested_sequential(seq_name, layer_id, elt, subnn)
                    continue

                if lyr_type in actv_fun_mapping:
                    layer_id = self._handle_sequential_activation(lyr_type, layer_id, subnn)
                    continue

                lyr_type, lyr_params = self._extract_bidirectional_params(elt, lyr_type, lyr_params)

                lyr_type, lyr_params, padding_amount, dropout_rate, has_recurrent_dropout = transform_layer(
                    lyr_type, lyr_params, self.padding_amount
                )
                self.padding_amount = padding_amount

                if has_recurrent_dropout:
                    self.migration_warnings.append(
                        "Sequential layer has recurrent_dropout parameter, "
                        "which has no equivalent in PyTorch LSTM/GRU/SimpleRNN. This parameter is not migrated."
                    )

                if not lyr_type.startswith("ZeroPadding"):
                    layer_id = self._add_sequential_layer(lyr_type, lyr_params, layer_id, dropout_rate, subnn)

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

    def _process_chained_call(self, node, chain):
        """Process a chained call by decomposing into individual calls."""
        for i, call in enumerate(chain):
            is_last = (i == len(chain) - 1)
            if is_last:
                synthetic_node = ast.Assign(targets=node.targets, value=call)
            else:
                temp_name = f"_chain_temp_{self.tensor_op_counter}_{i}"
                temp_target = ast.Name(id=temp_name, ctx=ast.Store())
                synthetic_node = ast.Assign(targets=[temp_target], value=call)

                if i + 1 < len(chain):
                    next_call = chain[i + 1]
                    next_func_value = next_call.func.value

                    if isinstance(next_func_value, ast.Name):
                        if next_call.args:
                            next_call.args[0] = ast.Name(id=temp_name, ctx=ast.Load())
                    elif isinstance(next_func_value, ast.Call):
                        next_call.func.value = ast.Name(id=temp_name, ctx=ast.Load())
            synthetic_node.lineno = node.lineno
            synthetic_node.col_offset = node.col_offset
            self.process_single_call(synthetic_node)
            self.previous_assign = synthetic_node

    def _extract_nested_call(self, node):
        """Extract nested call from arguments and process it."""
        inner_call = node.value.args[0]
        # Use the outer output variable instead of creating a temp variable
        # For x_1 = fc3(fc2(fc1(h))), use 'x_1' throughout
        outer_var = node.targets[0].id

        inner_node = ast.Assign(
            targets=[ast.Name(id=outer_var, ctx=ast.Store())],
            value=inner_call
        )
        inner_node.lineno = node.lineno
        inner_node.col_offset = node.col_offset
        self.handle_forward_simple_call(inner_node)
        self.previous_assign = inner_node

        node.value.args[0] = ast.Name(id=outer_var, ctx=ast.Load())

    def _extract_subscript_arg(self, node):
        """Extract subscript from arguments and process it."""
        subscript_node = node.value.args[0]
        temp_name = f"_subscript_arg_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        subscript_assign = ast.Assign(
            targets=[ast.Name(id=temp_name, ctx=ast.Store())],
            value=subscript_node
        )
        subscript_assign.lineno = node.lineno
        subscript_assign.col_offset = node.col_offset
        self.handle_forward_slicing(subscript_assign)
        self.previous_assign = subscript_assign

        node.value.args[0] = ast.Name(id=temp_name, ctx=ast.Load())

    def handle_forward_simple_call(self, node: ast.Assign):
        """
        Processes forward method assignments including chained calls.
        """
        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'value'):
            if isinstance(node.value.func.value, ast.Call):
                chain = self.decompose_chained_call(node)
                self._process_chained_call(node, chain)
                return

        is_nested_call = False
        if isinstance(node.value, ast.Call) and node.value.args:
            if isinstance(node.value.args[0], ast.Call):
                self._extract_nested_call(node)
                is_nested_call = True
            elif isinstance(node.value.args[0], ast.Subscript):
                self._extract_subscript_arg(node)
                is_nested_call = True

        if is_nested_call:
            self.is_processing_nested_outer = True

        self.process_single_call(node)
        self.previous_assign = node

        if is_nested_call:
            self.is_processing_nested_outer = False

    def _create_shape_dim_tensorop_for_reshape(self, source_var, dim_idx):
        """Create shape_dim tensorop for reshape argument and return operation name."""
        source_layers = [self.module_of_output[source_var]] if source_var in self.module_of_output else ['INPUT']

        op_name = f"op_{self.tensor_op_counter}"
        shape_tensorop_param = {
            "tns_type": "shape_dim",
            "reduce_dim": dim_idx,
            "layers_of_tensors": source_layers,
            "name": op_name
        }
        tns_obj = getattr(mm_classes, "TensorOp")(**shape_tensorop_param)
        self.buml_model.add_tensor_op(tns_obj)
        self.module_of_output[op_name] = op_name
        self.tensor_op_counter += 1
        return op_name

    def _extract_shape_subscript_info(self, arg, node):
        """Extract dimension index and source variable from tf.shape(x)[dim] pattern."""
        # Extract dimension index from subscript
        if isinstance(arg.slice, ast.Constant):
            dim_idx = arg.slice.value
        elif isinstance(arg.slice, ast.Index) and isinstance(arg.slice.value, ast.Constant):
            dim_idx = arg.slice.value.value
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported slice type in tf.shape subscript"
            )
            return None, None

        # Extract source tensor from tf.shape() arguments
        if len(arg.value.args) == 0:
            self.migration_warnings.append(
                f"Line {node.lineno}: tf.shape() call missing argument"
            )
            return None, None

        source_arg = arg.value.args[0]
        if not isinstance(source_arg, ast.Name):
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported source type in tf.shape()"
            )
            return None, None

        return dim_idx, source_arg.id

    def _process_reshape_arg(self, arg, node):
        """
        Process single reshape argument, handling tf.shape()[dim] calls and variables.

        Similar to torch2tf's _process_reshape_arg but for TensorFlow patterns.
        Detects tf.shape(x)[dim] subscript patterns and creates shape_dim tensorops.

        Args:
            arg: AST node representing a reshape dimension argument
            node: Parent assignment node for line number context

        Returns:
            Dimension value (int, str variable name, or op name for shape_dim)
        """
        # Handle List nodes (e.g., [batch_size, -1] in tf.reshape(x, [batch_size, -1]))
        if isinstance(arg, ast.List):
            return [self._process_reshape_arg(el, node) for el in arg.elts]

        # Handle tf.shape(x)[dim] subscript pattern
        if isinstance(arg, ast.Subscript):
            if (isinstance(arg.value, ast.Call) and
                hasattr(arg.value.func, 'attr') and
                arg.value.func.attr == 'shape' and
                hasattr(arg.value.func, 'value') and
                isinstance(arg.value.func.value, ast.Name) and
                arg.value.func.value.id == 'tf'):

                dim_idx, source_var = self._extract_shape_subscript_info(arg, node)
                if dim_idx is not None and source_var is not None:
                    return self._create_shape_dim_tensorop_for_reshape(source_var, dim_idx)
                return -1

        # Handle Name nodes (variables that might reference shape dims)
        if isinstance(arg, ast.Name):
            var_name = arg.id
            if var_name in self.module_of_output:
                layer_name = self.module_of_output[var_name]
                return layer_name if layer_name.startswith('op_') else self.param_value(arg)
            return var_name

        # Handle Constant nodes (literal values like -1, 64, etc.)
        if isinstance(arg, ast.Constant):
            return arg.value

        # Handle older Python AST Num nodes
        if isinstance(arg, ast.Num):
            return arg.n

        # Fallback to param_value for other cases
        return self.param_value(arg)

    def _handle_initial_state_keyword(self, module_obj, module_name, kw):
        """Handle initial_state keyword argument for RNN layers."""
        if isinstance(kw.value, ast.List) and len(kw.value.elts) >= 1:
            h_var = kw.value.elts[0].id if isinstance(kw.value.elts[0], ast.Name) else None
        elif isinstance(kw.value, ast.Name):
            h_var = kw.value.id
        else:
            return

        if not h_var or h_var not in self.module_of_output:
            return

        source_module = self.module_of_output[h_var].replace("__hidden", "").replace("__cell", "")
        module_obj.hx_source = source_module

        # Mark source layer output as reused
        source_layer = self._get_layer_by_name(source_module)
        if source_layer:
            source_layer.input_reused = True
            if not hasattr(self, '_reserved_tf_vars'):
                self._reserved_tf_vars = set()
            self._reserved_tf_vars.add(h_var)

    def _set_module_input(self, module_obj, input_var):
        """Set name_module_input for a module based on input variable."""
        if input_var in self.module_of_output:
            module_obj.name_module_input = self.module_of_output[input_var]
        elif input_var == "x":
            module_obj.name_module_input = "INPUT"

    def _handle_dropout_before_module(self, module_obj, module_name):
        """Add Dropout module before RNN/layer if dropout layer exists."""
        dropout_layer_name = f"{module_name}_dropout"
        dropout_module = next((obj for obj in self.buml_model.layers if obj.name == dropout_layer_name), None)
        if dropout_module:
            dropout_module.name_module_input = module_obj.name_module_input
            module_obj.name_module_input = dropout_layer_name
            self.buml_model.modules.append(dropout_module)

    def _extract_call_input_var(self, node):
        """Extract input variable from call arguments."""
        if not node.value.args:
            return "x"

        arg = node.value.args[0]
        if isinstance(arg, ast.Name):
            return arg.id
        elif isinstance(arg, ast.Call):
            # After nested call extraction, the arg becomes the outer var name
            return node.targets[0].id
        else:
            return "x"

    def _find_module_by_name(self, module_name):
        """Find module object by name in layers or sub_nns."""
        # First check the tracking dict for O(1) lookup
        module_obj = self._get_layer_by_name(module_name)
        if not module_obj:
            module_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)
        return module_obj

    def _mark_split_input_reuse(self, module_obj, input_var):
        """Mark input_reused if layer consumes a split output."""
        if input_var in self.module_of_output:
            source = self.module_of_output[input_var]
            if isinstance(source, str) and "__split_" in source:
                module_obj.input_reused = True

    def _mark_rnn_hidden_usage(self, module_obj, input_var):
        """Mark if layer uses RNN hidden state as input."""
        if not (module_obj and input_var in self.module_of_output):
            return

        source_module = self.module_of_output[input_var]
        if source_module in self.rnn_hidden_vars and self.rnn_hidden_vars[source_module] == input_var:
            module_obj.use_rnn_hidden = True

    def _is_tf_nn_activation(self, node):
        """Check if node is a tf.nn.activation call."""
        return (isinstance(node.value.func.value, ast.Attribute) and
                node.value.func.value.attr == "nn" and
                isinstance(node.value.func.value.value, ast.Name) and
                node.value.func.value.value.id == "tf")

    def _process_self_module_call(self, node, module_name):
        """Process a call to a module defined in self (e.g., self.layer(x))."""
        input_var = self._extract_call_input_var(node)
        output_var = node.targets[0].id

        module_obj = self._find_module_by_name(module_name)

        # Check if this is a sub_nn (Sequential) - same pattern as torch2tf
        is_subnn = False
        if module_obj and module_obj in self.buml_model.sub_nns:
            is_subnn = True

        # Check if this is layer reuse BEFORE modifying module_obj
        is_layer_reuse = module_obj and module_obj in self.buml_model.modules and not is_subnn

        if module_obj and not is_layer_reuse:
            module_obj.input_var = input_var
            module_obj.output_var = output_var
            self._set_module_input(module_obj, input_var)
            self._mark_split_input_reuse(module_obj, input_var)

        self.module_of_output[output_var] = module_name

        if module_obj and hasattr(module_obj, 'return_type'):
            for kw in node.value.keywords:
                if kw.arg == "initial_state":
                    self._handle_initial_state_keyword(module_obj, module_name, kw)
                    break

        self._mark_rnn_hidden_usage(module_obj, input_var)

        if module_obj:
            self._handle_dropout_before_module(module_obj, module_name)

        if module_name in self.activation_functions:
            self._handle_module_activation(node, module_name)
        elif is_layer_reuse:
            # Handle layer reuse - pass the current input/output vars for this use
            self._handle_module_layer_reuse(node, module_name, module_obj, input_var, output_var)
        elif module_obj and not is_subnn:
            # Add first occurrence with suffix tracking (only for layers, not sub_nns)
            self._add_layer_with_tracking(module_obj)
            # Update module_of_output to point to suffixed name
            self.module_of_output[node.targets[0].id] = module_obj.name
        elif is_subnn and module_obj not in self.buml_model.modules:
            # SubNN (Sequential) called in forward - add to modules for generator
            # No suffix tracking needed for SubNNs
            self._add_module_with_tracking(module_obj)

    def process_single_call(self, node: ast.Assign):
        """
        Process a single (non-chained) call operation.
        This contains the original logic from handle_forward_simple_call.
        """
        # Check if node.value.func has a 'value' attribute (guard against functional API nested calls)
        if not hasattr(node.value.func, 'value'):
            # Functional API pattern in forward - skip it
            return

        if isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id == "self":
                module_name = node.value.func.attr
                self._process_self_module_call(node, module_name)
            else:
                if self._is_tf_nn_activation(node):
                    func_name = node.value.func.attr
                    if func_name in tf_actv_func_mapping:
                        self.handle_tf_activation(node, func_name)
                    else:
                        self.extract_tensorop(node)
                else:
                    self.extract_tensorop(node)
        elif isinstance(node.value.func.value, ast.Attribute):
            if node.value.func.value.value.id == "tf":
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
            actv_lyr.input_var = input_var
            actv_lyr.output_var = output_var
            self.module_of_output[output_var] = actv_lyr_name
        else:
            # Attach activation to previous layer
            if prev_lyr_obj:
                prev_lyr_obj.actv_func = tf_actv_func_mapping[func_name]
                # Update layer output for nested calls
                if hasattr(self, 'is_processing_nested_outer') and self.is_processing_nested_outer:
                    prev_lyr_obj.output_var = node.targets[0].id
            # Update module_of_output
            if input_var in self.module_of_output:
                self.module_of_output[node.targets[0].id] = self.module_of_output[input_var]

    def _handle_module_activation(self, node, module_name):
        """
        Handle module API activation layers.

        For TensorFlow: standalone activation layers (layers.ReLU(), layers.Activation('sigmoid'))
        are never merged with previous layers to preserve their original layer names.
        Only inline activations (Dense(activation='relu')) would already be part of the layer.
        """
        # Preserve original module name for standalone activations
        actv = self.activation_functions[module_name]
        # actv might already be the BUML name (relu) or need mapping
        actv_func = actv if actv in tf_actv_func_mapping.values() else actv

        actv_lyr = mm_classes.GeneralLayer(name=module_name, actv_func=actv_func)
        self.buml_model.modules.append(actv_lyr)

        input_var = node.value.args[0].id if node.value.args and isinstance(node.value.args[0], ast.Name) else "x"
        output_var = node.targets[0].id
        actv_lyr.input_var = input_var
        actv_lyr.output_var = output_var
        self.module_of_output[output_var] = module_name

    def _extract_source_variable(self, op_args, node, op_type):
        """
        Extract source variable from tensorop arguments.

        Returns:
            list: Source layers, or None if extraction failed
        """
        if len(op_args) == 0:
            self.migration_warnings.append(
                f"Line {node.lineno}: {op_type} requires an input tensor."
            )
            return None

        if not isinstance(op_args[0], ast.Name):
            self.migration_warnings.append(
                f"Line {node.lineno}: {op_type} operand type not supported."
            )
            return None

        source_var = op_args[0].id
        if source_var in self.module_of_output:
            return [self.module_of_output[source_var]]
        elif source_var == 'x':
            return ['INPUT']
        else:
            self.migration_warnings.append(
                f"Line {node.lineno}: {op_type} source variable '{source_var}' not found."
            )
            return None

    def _extract_zeros_like(self, node, op_args):
        """Extract zeros_like tensorop parameters."""
        source_layers = self._extract_source_variable(op_args, node, "zeros_like")
        if source_layers is None:
            return None

        return {
            "tns_type": "zeros_like",
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_squeeze(self, node, op_args):
        """Extract squeeze tensorop parameters."""
        # Extract dimension parameter (optional)
        reduce_dim = None
        for kw in node.value.keywords:
            if kw.arg == "axis":
                reduce_dim = self.param_value(kw.value)
                break

        source_layers = self._extract_source_variable(op_args, node, "squeeze")
        if source_layers is None:
            return None

        return {
            "tns_type": "squeeze",
            "reduce_dim": reduce_dim,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_expand_dims(self, node, op_args):
        """Extract expand_dims tensorop parameters."""
        # Extract dimension parameter (required)
        reduce_dim = None
        for kw in node.value.keywords:
            if kw.arg == "axis":
                reduce_dim = self.param_value(kw.value)
                break

        # axis can also be positional (second argument)
        if reduce_dim is None and len(op_args) > 1:
            reduce_dim = self.param_value(op_args[1])

        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: expand_dims requires an 'axis' parameter."
            )
            return None

        source_layers = self._extract_source_variable(op_args, node, "expand_dims")
        if source_layers is None:
            return None

        return {
            "tns_type": "unsqueeze",
            "reduce_dim": reduce_dim,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_reduce_mean(self, node, op_args):
        """Extract reduce_mean tensorop parameters."""
        # Extract dimension parameter
        reduce_dim = None
        for kw in node.value.keywords:
            if kw.arg == "axis":
                reduce_dim = self.param_value(kw.value)
                break

        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: reduce_mean requires an 'axis' parameter."
            )
            return None

        source_layers = self._extract_source_variable(op_args, node, "reduce_mean")
        if source_layers is None:
            return None

        return {
            "tns_type": "mean",
            "reduce_dim": reduce_dim,
            "layers_of_tensors": source_layers
        }

    def _extract_reduce_max(self, node, op_args):
        """Extract reduce_max tensorop parameters."""
        # Extract dimension parameter
        reduce_dim = None
        keepdims = False
        for kw in node.value.keywords:
            if kw.arg == "axis":
                reduce_dim = self.param_value(kw.value)
            elif kw.arg == "keepdims":
                keepdims = self.param_value(kw.value)

        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: reduce_max requires an 'axis' parameter."
            )
            return None

        source_layers = self._extract_source_variable(op_args, node, "reduce_max")
        if source_layers is None:
            return None

        return {
            "tns_type": "max",
            "reduce_dim": reduce_dim,
            "reduce_keepdims": keepdims,
            "layers_of_tensors": source_layers
        }

    def _extract_l2_normalize(self, node, op_args):
        """Extract l2_normalize tensorop parameters."""
        # Extract dimension parameter
        reduce_dim = None
        for kw in node.value.keywords:
            if kw.arg == "axis":
                reduce_dim = self.param_value(kw.value)
                break

        if reduce_dim is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: l2_normalize requires an 'axis' parameter."
            )
            return None

        source_layers = self._extract_source_variable(op_args, node, "l2_normalize")
        if source_layers is None:
            return None

        return {
            "tns_type": "normalize",
            "reduce_dim": reduce_dim,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_tile(self, node, op_args):
        """Extract tile tensorop parameters."""
        # Extract multiples/repeat counts
        repeat_counts = []
        if len(op_args) > 1:
            multiples_arg = op_args[1]
            if isinstance(multiples_arg, (ast.List, ast.Tuple)):
                for elt in multiples_arg.elts:
                    repeat_counts.append(self.param_value(elt))
            else:
                repeat_counts.append(self.param_value(multiples_arg))

        source_layers = self._extract_source_variable(op_args, node, "tile")
        if source_layers is None:
            return None

        return {
            "tns_type": "repeat",
            "repeat_dim": repeat_counts,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_binary_op(self, node, op_args, op_type):
        """Extract binary operation (add, subtract, multiply, divide, etc.)."""
        op_map = {
            "add": "binop_add",
            "subtract": "binop_subtract",
            "multiply": "multiply",
            "divide": "binop_divide",
            "floor_divide": "binop_floor_divide",
            "matmul": "matmultiply"
        }
        tns_type = op_map[op_type]

        # Extract operands (handles both variables and scalar constants)
        left_var = self._extract_binop_operand(op_args[0], node, "left")
        right_var = self._extract_binop_operand(op_args[1], node, "right")

        if left_var is None or right_var is None:
            return None

        # Get layer names or keep scalar values
        left_layer = left_var if isinstance(left_var, (int, float)) else self.module_of_output.get(left_var)
        right_layer = right_var if isinstance(right_var, (int, float)) else self.module_of_output.get(right_var)

        if left_layer is None or right_layer is None:
            return None

        layers_of_tensors = [left_layer, right_layer]
        return {"tns_type": tns_type, "layers_of_tensors": layers_of_tensors}

    def _extract_transpose(self, node, op_args):
        """Extract transpose tensorop parameters."""
        op_args_kw = node.value.keywords[0].value.elts
        permute_dim = [op_args_kw[0].value, op_args_kw[1].value, op_args_kw[2].value]
        return {"tns_type": "permute", "permute_dim": permute_dim}

    def _extract_reshape(self, node, op_args):
        """Extract reshape tensorop parameters."""
        source_layers = self._extract_source_variable(op_args, node, "reshape")
        if source_layers is None:
            return None

        # Process reshape arguments
        if len(op_args) == 2 and isinstance(op_args[1], ast.List):
            reshape_dim = self._process_reshape_arg(op_args[1], node)
        else:
            reshape_dim = [self._process_reshape_arg(op_args[i], node)
                           for i in range(1, len(op_args))]

        return {
            "tns_type": "reshape",
            "reshape_dim": reshape_dim,
            "layers_of_tensors": source_layers
        }

    def _extract_pad(self, node, op_args):
        """Extract pad tensorop parameters."""
        pad_amount = None
        pad_mode = 'CONSTANT'

        # Second argument is paddings
        if len(op_args) > 1:
            paddings_arg = op_args[1]
            if isinstance(paddings_arg, (ast.List, ast.Tuple)):
                pad_amount = []
                for elt in paddings_arg.elts:
                    if isinstance(elt, (ast.List, ast.Tuple)):
                        pad_amount.append([self.param_value(e) for e in elt.elts])
                    else:
                        pad_amount.append(self.param_value(elt))

        # Extract mode from keywords
        for kw in node.value.keywords:
            if kw.arg == "mode":
                pad_mode = self.param_value(kw.value)
                break

        source_layers = self._extract_source_variable(op_args, node, "pad")
        if source_layers is None:
            return None

        return {
            "tns_type": "pad",
            "pad_amount": pad_amount,
            "pad_mode": pad_mode,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_dropout(self, node, op_args):
        """Extract dropout tensorop parameters."""
        dropout_rate = 0.5  # Default

        for kw in node.value.keywords:
            if kw.arg == "rate":
                dropout_rate = self.param_value(kw.value)
                break

        source_layers = self._extract_source_variable(op_args, node, "dropout")
        if source_layers is None:
            return None

        return {
            "tns_type": "dropout",
            "dropout_rate": dropout_rate,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_resize(self, node, op_args):
        """Extract resize/interpolate tensorop parameters."""
        resize_size = None
        resize_mode = 'bilinear'

        # Second argument is size
        if len(op_args) > 1:
            size_arg = op_args[1]
            if isinstance(size_arg, (ast.List, ast.Tuple)):
                resize_size = tuple(self.param_value(elt) for elt in size_arg.elts)
            else:
                resize_size = self.param_value(size_arg)

        # Extract method from keywords
        for kw in node.value.keywords:
            if kw.arg == "method":
                resize_mode = self.param_value(kw.value)
                break

        source_layers = self._extract_source_variable(op_args, node, "resize")
        if source_layers is None:
            return None

        return {
            "tns_type": "interpolate",
            "interpolate_size": resize_size,
            "interpolate_mode": resize_mode,
            "layers_of_tensors": source_layers,
            "input_reused": True
        }

    def _extract_split(self, node, op_args):
        """Extract split tensorop parameters for tf.split(value, num_or_size_splits, axis)."""
        # tf.split signature: tf.split(value, num_or_size_splits, axis=0, num=None, name=None)
        # Example: chunk1, chunk2, chunk3 = tf.split(last, num_or_size_splits=3, axis=1)

        # Extract source layers using the common helper
        source_layers = self._extract_source_variable(op_args, node, "split")
        if source_layers is None:
            return None

        # Get num_or_size_splits from positional arg or keyword
        num_splits = None
        if len(op_args) >= 2:
            num_splits = self.param_value(op_args[1])
        else:
            for kw in node.value.keywords:
                if kw.arg == "num_or_size_splits":
                    num_splits = self.param_value(kw.value)
                    break

        if num_splits is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: tf.split missing num_or_size_splits argument"
            )
            return None

        # Get split axis from positional arg or keyword
        split_axis = None
        if len(op_args) >= 3:
            split_axis = self.param_value(op_args[2])
        else:
            for kw in node.value.keywords:
                if kw.arg == "axis":
                    split_axis = self.param_value(kw.value)
                    break

        if split_axis is None:
            split_axis = 0  # Default axis is 0 in TensorFlow

        # Track output variables from tuple assignment
        if isinstance(node.targets[0], ast.Tuple):
            output_vars = [elt.id for elt in node.targets[0].elts if isinstance(elt, ast.Name)]
        else:
            output_vars = [node.targets[0].id]

        return {
            "tns_type": "split",
            "split_sizes": num_splits,
            "split_dim": split_axis,
            "layers_of_tensors": source_layers,
            "output_vars": output_vars
        }

    def _extract_concat(self, node, op_args):
        """Extract concat tensorop parameters."""
        ops_args = node.value.args[0].elts

        # Extract variables
        variables = self._extract_concat_variables(ops_args, node)
        if variables is None:
            return None

        # Resolve aliases
        actual_vars = self._resolve_concat_variables(variables)

        # Get source layers
        layers_of_tensors = []
        for var in actual_vars:
            if isinstance(var, str) and var.startswith("INLINE_CALL:"):
                layers_of_tensors.append(var)
            elif var in self.module_of_output:
                layers_of_tensors.append(self.module_of_output[var])
            else:
                self.migration_warnings.append(
                    f"Line {node.lineno}: Variable '{var}' not found in module outputs."
                )
                return None

        # Get concatenation dimension
        cat_dim = self.param_value(node.value.keywords[0].value)

        # Track RNN var types
        var_types = self._determine_rnn_var_types_multi(layers_of_tensors, actual_vars)

        # Check if bidirectional RNN concat (handled by template)
        if self._is_bidirectional_hidden_concat(layers_of_tensors, var_types):
            output_var = node.targets[0].id
            self.variable_aliases[output_var] = actual_vars[0]
            self.module_of_output[output_var] = layers_of_tensors[0]
            return None

        # Check if concat output variable is different from input variables
        # If so, set input_reused to get a unique output variable name
        output_var = node.targets[0].id
        input_reused = output_var not in actual_vars

        return {
            "tns_type": "concatenate",
            "layers_of_tensors": layers_of_tensors,
            "concatenate_dim": cat_dim,
            "actual_vars": var_types,
            "input_reused": input_reused
        }

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
        if op_type == "split":
            tensorop_param = self._extract_split(node, op_args)
        elif op_type == "concat":
            tensorop_param = self._extract_concat(node, op_args)
        elif op_type in ["add", "subtract", "multiply", "divide", "floor_divide", "matmul"]:
            tensorop_param = self._extract_binary_op(node, op_args, op_type)
        elif op_type == "transpose":
            tensorop_param = self._extract_transpose(node, op_args)
        elif op_type == "reshape":
            tensorop_param = self._extract_reshape(node, op_args)
        elif op_type == "reduce_mean":
            tensorop_param = self._extract_reduce_mean(node, op_args)
        elif op_type == "zeros_like":
            tensorop_param = self._extract_zeros_like(node, op_args)
        elif op_type == "squeeze":
            tensorop_param = self._extract_squeeze(node, op_args)
        elif op_type == "expand_dims":
            tensorop_param = self._extract_expand_dims(node, op_args)
        elif op_type == "reduce_max":
            tensorop_param = self._extract_reduce_max(node, op_args)
        elif op_type == "l2_normalize":
            tensorop_param = self._extract_l2_normalize(node, op_args)
        elif op_type == "tile":
            tensorop_param = self._extract_tile(node, op_args)
        elif op_type == "pad":
            tensorop_param = self._extract_pad(node, op_args)
        elif op_type == "dropout":
            tensorop_param = self._extract_dropout(node, op_args)
        elif op_type == "resize":
            tensorop_param = self._extract_resize(node, op_args)
        else:
            self.migration_warnings.append(
                f"TensorOp type '{op_type}' is not recognized and will be skipped."
            )

        # Handle dropout as a layer instead of tensorop
        if (tensorop_param and
            tensorop_param.get('tns_type') == 'dropout'):
            self._create_dropout_layer(tensorop_param, node)
            return

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name

            # Special handling for split: remove output_vars before creating TensorOp
            # (output vars are tracked via layer/tensorop attributes instead)
            output_vars = tensorop_param.pop("output_vars", None)

            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1

            # Track output variable(s)
            if hasattr(node, 'targets') and len(node.targets) > 0:
                if output_vars and len(output_vars) > 1:
                    # Multiple outputs (e.g., split): track each with suffix like RNN does
                    for idx, var in enumerate(output_vars):
                        self.module_of_output[var] = f"{op_name}__split_{idx}"

                    # Track output for split: output is comma-joined tuple of actual vars
                    # Find input variable
                    first_layer = tensorop_param.get('layers_of_tensors', [None])[0]
                    input_var = None
                    if first_layer:
                        for var, mod_name in self.module_of_output.items():
                            if mod_name == first_layer:
                                input_var = var
                                break
                    if input_var is None:
                        input_var = 'x'

                    # Track output for split: output is comma-joined tuple of actual vars
                    tns_obj.input_var = input_var
                    tns_obj.output_vars = output_vars  # Set as list, not comma-joined string
                    # DO NOT set tns_obj.output_var for split ops
                else:
                    # Single output: normal tracking
                    target_var = node.targets[0].id
                    self.module_of_output[target_var] = op_name

                    # Track inputs/outputs for all tensorops using actual TF variable names
                    # This preserves original code patterns (e.g., x = tf.add(x, y) → input='x', output='x')
                    if 'layers_of_tensors' in tensorop_param and tensorop_param['layers_of_tensors']:
                        # Find the variable name that the first input layer produces
                        first_layer = tensorop_param['layers_of_tensors'][0]
                        input_var = None

                        # Reverse lookup: search module_of_output to find which variable this layer produces
                        for var, mod_name in self.module_of_output.items():
                            if mod_name == first_layer:
                                input_var = var
                                break

                        # Fallback if not found (shouldn't happen, but be safe)
                        if input_var is None:
                            input_var = 'x'

                        tns_obj.input_var = input_var
                        tns_obj.output_var = target_var

    def _create_dropout_layer(self, tensorop_param, node):
        """Create a Dropout layer from tf.nn.dropout."""
        # Generate layer name following convention
        dropout_name = f"dropout_{self.tensor_op_counter}"
        self.tensor_op_counter += 1

        # Get source layers from tensorop_param
        source_layers = tensorop_param.get('layers_of_tensors', [])
        name_module_input = source_layers[0] if source_layers else None

        # Create DropoutLayer using BUML
        layer_obj = getattr(mm_classes, "DropoutLayer")(
            name=dropout_name,
            rate=tensorop_param.get('dropout_rate', 0.5)
        )

        # Set input source if available
        if name_module_input:
            layer_obj.name_module_input = name_module_input

        # Add layer and track output
        self.buml_model.add_layer(layer_obj)
        self.buml_model.modules.append(layer_obj)

        # Track output variable
        if hasattr(node, 'targets') and len(node.targets) > 0:
            output_var = node.targets[0].id
            self.module_of_output[output_var] = dropout_name

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
                batch_size_kwarg = next(
                    (k for k in node.value.keywords if k.arg == "batch_size"),
                    None
                )
                if batch_size_kwarg is not None:
                    # Handle both constant values and variable references
                    if isinstance(batch_size_kwarg.value, ast.Constant):
                        config["batch_size"] = batch_size_kwarg.value.value
                    elif isinstance(batch_size_kwarg.value, ast.Name):
                        # Variable reference - we'll need to track it from outer scope
                        # For now, skip setting batch_size in config
                        pass

    def visit_Expr(self, node: ast.Expr):
        """
        Visit expression statements to extract epochs from train function calls.
        TF code has: train(model, train_loader, criterion, optimizer, epochs, num_classes)
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
                # Standard signature: train(model, train_loader, criterion, optimizer, epochs, ...)
                # epochs is the 5th parameter (index 4)
                if "epochs" not in self.data_config["config"] and len(node.value.args) >= 5:
                    epochs_arg = node.value.args[4]
                    if isinstance(epochs_arg, ast.Constant):
                        self.data_config["config"]["epochs"] = epochs_arg.value

        self.generic_visit(node)

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
        cnns = ["Conv1D", "Conv2D", "Conv3D", "PoolingLayer", "BatchNormLayer"]
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


    def add_squeeze_for_global_pooling(self):
        """
        Add squeeze TensorOps after global pooling layers to remove the
        size-1 dimension added by AdaptiveAvgPool.
        """
        def add_squeeze_ops(modules):
            """Insert squeeze TensorOps after global pooling layers."""
            i = 0
            while i < len(modules):
                module = modules[i]
                if (isinstance(module, Layer) and
                    module.__class__.__name__ == "PoolingLayer" and
                    hasattr(module, 'pooling_type') and
                    module.pooling_type.startswith("global")):
                    # Create squeeze TensorOp to remove the size-1 dimension
                    squeeze_op = mm_classes.TensorOp(
                        name=f"{module.name}_squeeze",
                        tns_type='squeeze',
                        reduce_dim=-1
                    )
                    # Insert squeeze op right after the pooling layer
                    modules.insert(i + 1, squeeze_op)
                    i += 1  # Skip the newly inserted squeeze op
                i += 1

        if self.buml_model.sub_nns:
            for subnn in self.buml_model.sub_nns:
                add_squeeze_ops(subnn.modules)

        add_squeeze_ops(self.buml_model.modules)


    def _is_cnn_module(self, module, cnns):
        """Check if module is a CNN-type layer or spatial tensorop."""
        if isinstance(module, Layer):
            return module.__class__.__name__ in cnns
        elif isinstance(module, TensorOp):
            return module.tns_type in ("interpolate", "pad")
        return False

    def _is_batchnorm_on_spatial_data(self, module, modules, cnns):
        """Check if BatchNormLayer operates on spatial data (follows a CNN layer)."""
        if not module.name_module_input:
            return False

        prev_module = next((obj for obj in modules if obj.name == module.name_module_input), None)
        if not prev_module:
            return False

        if isinstance(prev_module, Layer):
            return prev_module.__class__.__name__ in cnns
        elif isinstance(prev_module, TensorOp):
            return prev_module.tns_type in ("interpolate", "pad")
        return False

    def _should_treat_next_as_cnn(self, next_module, cnns):
        """Determine if next module should be treated as CNN for permute logic."""
        if isinstance(next_module, Layer):
            return next_module.__class__.__name__ in cnns
        elif isinstance(next_module, TensorOp):
            if next_module.tns_type in ("interpolate", "pad"):
                return True
            if next_module.tns_type == "permute" and hasattr(next_module, 'permute_dim'):
                canceling_permutes = [[0, 2, 1], [0, 2, 3, 1], [0, 2, 3, 4, 1]]
                return next_module.permute_dim in canceling_permutes
        return False

    def _update_prev_module_reference(self, module, modules):
        """Update prev_module reference based on module's input."""
        if isinstance(module, Layer) and module.name_module_input:
            return next((obj for obj in modules if obj.name == module.name_module_input), None)
        elif isinstance(module, TensorOp) and module.layers_of_tensors:
            # For TensorOp, use the first input from layers_of_tensors
            input_name = module.layers_of_tensors[0] if isinstance(module.layers_of_tensors[0], str) else None
            if input_name:
                return next((obj for obj in modules if obj.name == input_name), None)
        return None

    def _set_input_permute(self, module, prev_cnn, prev_module, lyr_out_permuted):
        """Set input permute flags on module based on previous module state."""
        needs_input_permute = False

        if not prev_cnn and (prev_module is None or prev_module.name not in lyr_out_permuted):
            needs_input_permute = True
        elif prev_cnn and prev_module and prev_module.name in lyr_out_permuted:
            needs_input_permute = True

        if needs_input_permute:
            if isinstance(module, (Layer, TensorOp)):
                module.permute_in = True
            if prev_module is not None and prev_module.name not in lyr_out_permuted:
                lyr_out_permuted.append(prev_module.name)

    def _set_output_permute(self, module, next_cnn, lyr_out_permuted):
        """Set output permute flags on module based on next module state."""
        if next_cnn:
            return

        is_global_pooling = (isinstance(module, Layer) and
                           module.__class__.__name__ == "PoolingLayer" and
                           hasattr(module, 'pooling_type') and
                           module.pooling_type.startswith("global"))

        if not is_global_pooling and isinstance(module, (Layer, TensorOp)):
            module.permute_out = True

        if module.name not in lyr_out_permuted:
            lyr_out_permuted.append(module.name)

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
        next_cnn = self._should_treat_next_as_cnn(next_module, cnns)

        current_cnn = False
        if isinstance(module, Layer) and module.__class__.__name__ in cnns:
            if module.__class__.__name__ == "BatchNormLayer":
                current_cnn = self._is_batchnorm_on_spatial_data(module, modules, cnns)
            else:
                current_cnn = True

            if module.name_module_input:
                prev_module = self._update_prev_module_reference(module, modules)

            if isinstance(next_module, Layer) and next_module.name_module_input:
                if next_module.name_module_input != module.name:
                    next_cnn = False
        elif isinstance(module, TensorOp) and module.tns_type in ("interpolate", "pad"):
            current_cnn = True
            if module.layers_of_tensors:
                prev_module = self._update_prev_module_reference(module, modules)

        prev_cnn = self._is_cnn_module(prev_module, cnns) if prev_module else False

        if current_cnn:
            self._set_input_permute(module, prev_cnn, prev_module, lyr_out_permuted)
            self._set_output_permute(module, next_cnn, lyr_out_permuted)

        return module



def transform_layer(lyr_type: str, lyr_params: dict,
                    padding_amount: int | None, layer_name=None):
    """
    It transforms layers and their params from TensorFlow to BUML.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionary storing the layer parameters and
            their values.
        padding_amount (int | None): It  keeps track of padding in
            ZeroPadding layer. In TF, padding is added to conv layers
            using a separate layer, but in PyTorch and BUML it is defined
            as an attribute of the conv layer.
        lyr_name (str): The name of the layer.

    Returns:
        Tuple of (lyr_type, lyr_params, padding_amount, dropout_rate, has_recurrent_dropout)
    """
    process_positional_params(lyr_type, lyr_params, pos_params)

    param_to_list(lyr_type, lyr_params, int2list_params,
                  lyrs_of_int2list_params)

    lyr_params, padding_amount = set_conv_padding(lyr_type, lyr_params,
                                                  padding_amount)

    lyr_params, dropout_rate, has_recurrent_dropout = process_params(lyr_type, lyr_params)
    lyr_params["name"] = layer_name
    if not lyr_type.startswith("ZeroPadding"):
        lyr_type = layers_mapping[lyr_type]

    return lyr_type, lyr_params, padding_amount, dropout_rate, has_recurrent_dropout



def _process_units_param(lyr_type, lyr_params, updated_params):
    """Handle units parameter for Dense and RNN layers."""
    lyrs_units = rnn_layers + ["Dense"]
    if lyr_type in lyrs_units and "units" in lyr_params:
        param = "out_features" if lyr_type == "Dense" else "hidden_size"
        updated_params[param] = lyr_params["units"]

def _extract_dropout_params(lyr_type, lyr_params):
    """Extract dropout parameters from RNN layers.

    TensorFlow LSTM dropout applies to inputs at each timestep.
    PyTorch LSTM dropout parameter only works between layers (num_layers > 1).
    For single-layer PyTorch LSTM, we need a separate Dropout layer BEFORE the LSTM.
    """
    dropout_rate = None
    has_recurrent_dropout = False

    if lyr_type in rnn_layers:
        # Extract dropout to create a separate Dropout layer before the LSTM
        # This preserves TF behavior since PyTorch LSTM dropout doesn't work for single-layer
        if "dropout" in lyr_params and lyr_params["dropout"] > 0:
            dropout_rate = lyr_params["dropout"]
            # Remove from LSTM params - will be replaced by separate Dropout layer
            del lyr_params["dropout"]

        if "recurrent_dropout" in lyr_params and lyr_params["recurrent_dropout"] > 0:
            has_recurrent_dropout = True

    return dropout_rate, has_recurrent_dropout

def _map_param(param, value, updated_params, lyr_type=None):
    """Map a single parameter from TF to BUML."""
    if param == "activation":
        # LSTM and GRU have tanh hardcoded internally - skip their activation parameter
        # to avoid creating extra activation layers
        if lyr_type in ["LSTM", "GRU"]:
            # Skip - internal activation, not a separate layer
            pass
        else:
            # For SimpleRNN (maps to nonlinearity param) and Dense/other layers (separate activation)
            updated_params["actv_func"] = value
    elif param in ["return_sequences", "return_state", "dropout", "recurrent_dropout", "units", "positional_params", "name"]:
        pass
    elif param == "mask_zero":
        if value is True:
            updated_params["padding_idx"] = 0
    elif param in params_mapping:
        updated_params[params_mapping[param]] = value

def _determine_rnn_return_type(lyr_type, lyr_params, updated_params):
    """Determine RNN return_type from return_sequences and return_state."""
    if lyr_type not in rnn_layers:
        return

    has_return_sequences = lyr_params.get("return_sequences", False)
    has_return_state = lyr_params.get("return_state", False)

    if has_return_sequences and has_return_state:
        updated_params["return_type"] = "both"
    elif has_return_sequences:
        updated_params["return_type"] = "full"
    elif has_return_state:
        updated_params["return_type"] = "hidden"
    else:
        updated_params["return_type"] = "last"

def process_params(lyr_type: str, lyr_params: dict):
    """
    It processes and transforms the layers' parameters.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionary storing the layer parameters and
            their values.

    Returns:
        Tuple of (transformed_params, dropout_rate, has_recurrent_dropout)
        - transformed_params: The parameters transformed into BUML
        - dropout_rate: Dropout rate if present on RNN layer, None otherwise
        - has_recurrent_dropout: True if recurrent_dropout param was present
    """
    updated_lyr_params = {}

    _process_units_param(lyr_type, lyr_params, updated_lyr_params)
    dropout_rate, has_recurrent_dropout = _extract_dropout_params(lyr_type, lyr_params)

    for param, value in lyr_params.items():
        _map_param(param, value, updated_lyr_params, lyr_type)

    _determine_rnn_return_type(lyr_type, lyr_params, updated_lyr_params)
    set_static_params(lyr_type, updated_lyr_params, static_params)

    return updated_lyr_params, dropout_rate, has_recurrent_dropout


def set_conv_padding(layer_type, lyr_params, padding_amount):
    """
    If padding is used before conv layers, its amount is stored in
    the 'padding_amount' attribute. This function adds the padding
    to the conv layer. Also handles conversion of TF padding='same'
    to PyTorch numeric padding.

    Parameters:
        lyr_type (str): The type of the layer (TensorFlow).
        lyr_params (dict): A dictionary storing the layer parameters and
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
    elif layer_type.startswith("Conv") or layer_type.startswith("MaxPooling") or layer_type.startswith("MaxPool") or layer_type.startswith("AveragePooling") or layer_type.startswith("AveragePool"):
        # Check if there's accumulated padding from ZeroPadding layer first
        if padding_amount is not None:
            lyr_params["padding_amount"] = padding_amount
            padding_amount = None
            # Remove any padding parameter since we're using accumulated padding
            if "padding" in lyr_params:
                del lyr_params["padding"]
        # Handle TF padding='same' -> calculate PyTorch padding amount
        # Note: At this point params haven't been mapped yet, so use TF names
        elif "padding" in lyr_params and lyr_params["padding"] == "same":
            # For 'same' padding with stride=1: padding = (kernel_size minus 1) // 2
            # This formula works for most common cases
            # Use pool_size for pooling layers, kernel_size for conv layers
            size_param = "pool_size" if layer_type.startswith(("MaxPooling", "MaxPool", "AveragePooling", "AveragePool")) else "kernel_size"
            if size_param in lyr_params:
                size_value = lyr_params[size_param]
                # Calculate padding for each dimension
                if isinstance(size_value, list):
                    padding = [(k - 1) // 2 for k in size_value]
                    # If all dimensions have same padding, use single value
                    if len(set(padding)) == 1:
                        padding = padding[0]
                else:
                    padding = (size_value - 1) // 2
                lyr_params["padding_amount"] = padding
                # Remove padding since we converted to padding_amount
                del lyr_params["padding"]
        # Handle padding='valid' -> padding=0
        elif "padding" in lyr_params and lyr_params["padding"] == "valid":
            lyr_params["padding_amount"] = 0
            del lyr_params["padding"]

    return lyr_params, padding_amount

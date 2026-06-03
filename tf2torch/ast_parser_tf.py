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
        self.layer_reuse_count = {}  # {layer_name: reuse_count} for layer reuse tracking

    def _get_layer_by_name(self, layer_name):
        """Get layer by name using O(1) lookup, fallback to linear search if not in dict."""
        if layer_name in self.layer_by_name:
            return self.layer_by_name[layer_name]
        # Fallback to linear search and update dict (for layers added before tracking)
        layer_obj = next((obj for obj in self.buml_model.layers if obj.name == layer_name), None)
        if layer_obj:
            self.layer_by_name[layer_name] = layer_obj
        return layer_obj

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

    def handle_forward_binop(self, node: ast.Assign):
        """
        Handle binary operations in forward method (e.g., x + y, x * 2, a // b).
        Creates TensorOp objects for arithmetic operations.
        """
        binop_node = node.value
        op_type = binop_node.op.__class__.__name__

        # Map Python AST op types to BUML tns_type
        op_map = {
            'Add': 'binop_add',
            'Sub': 'binop_subtract',
            'Mult': 'binop_multiply',
            'Div': 'binop_divide',
            'FloorDiv': 'binop_floor_divide'
        }

        if op_type not in op_map:
            self.migration_warnings.append(
                f"Line {node.lineno}: Unsupported binary operation '{op_type}'. This operation will be skipped."
            )
            return

        tns_type = op_map[op_type]

        # Extract operands (variable names or constant values)
        left_var = self._extract_binop_operand(binop_node.left, node, "left")
        right_var = self._extract_binop_operand(binop_node.right, node, "right")

        if left_var is None or right_var is None:
            self.migration_warnings.append(
                f"Line {node.lineno}: Could not extract operands for binary operation. Operation skipped."
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

        # Create TensorOp
        target_var = node.targets[0].id
        tensorop_param = {
            "name": f"op_{self.tensor_op_counter}",
            "tns_type": tns_type,
            "layers_of_tensors": [left_layer, right_layer],
            "actual_vars": var_types  # Track which component (output/hidden) each refers to
        }
        self.tensor_op_counter += 1

        try:
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.module_of_output[target_var] = tensorop_param["name"]

            # Update prev_layer_output for TensorOps
            self.prev_layer_output = target_var

            # Mark source layers for residual connections
            # When we have x = a + b where a and b are from different layers,
            # mark both source layers so their inputs are preserved
            if tns_type == "binop_add" and isinstance(left_layer, str) and isinstance(right_layer, str):
                for layer_name in [left_layer, right_layer]:
                    if layer_name and not isinstance(layer_name, (int, float)):
                        layer_obj = self._get_layer_by_name(layer_name)
                        if layer_obj:
                            layer_obj.input_reused = True

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
            if lyr_name in self.rnn_hidden_vars and actual_var == self.rnn_hidden_vars[lyr_name]:
                var_types.append("hidden")
            elif lyr_name in self.rnn_output_vars and actual_var == self.rnn_output_vars[lyr_name]:
                var_types.append("output")
            else:
                var_types.append("output")
        return var_types

    def _handle_layer_reuse_in_concat(self, layer_obj, arg, layer_name):
        """Handle layer reuse in concatenation."""
        import copy
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
            # Subscript operation (e.g., x[:, -1])
            temp_name = self.TEMP_SUBSCRIPT.format(self.tensor_op_counter)
            self.tensor_op_counter += 1
            self.handle_subscript_operation(operand_node, temp_name, node)
            return temp_name
        elif isinstance(operand_node, ast.BinOp):
            # Nested binary operation - recursively handle it
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

        if op_type not in ['squeeze', 'unsqueeze', 'repeat']:
            if base_var == 'self':
                return None
            return base_var

        base_layer = self.module_of_output.get(base_var)
        if base_layer is None:
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

        subscript_op = self._create_subscript_op(op_name, source_module, subscript_pattern)
        self.buml_model.modules.append(subscript_op)
        self.inputs_outputs[op_name] = [resolved_var, output_var]
        self.module_of_output[output_var] = op_name

    def _is_rnn_subscript(self, subscripted_var):
        """Check if subscript is on an RNN layer."""
        if not (subscripted_var and subscripted_var in self.module_of_output):
            return False

        src_module = self.module_of_output[subscripted_var]
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
        # Check if this is a method call (e.g., x.max(dim=1)) vs module call (e.g., self.rnn(x))
        if hasattr(node.value.func, 'value') and isinstance(node.value.func.value, ast.Name):
            caller_id = node.value.func.value.id
            if caller_id != "self":
                self.extract_tensorop(node)
                return

        module_name = node.value.func.attr

        # Extract and track tuple target variables
        self._extract_tuple_target_vars(node, module_name)

        # Determine main output variable
        rnn_out = self._determine_rnn_output_var(node)

        # Extract input argument
        rnn_in = self._extract_rnn_input_arg(node)

        # Update tracking structures
        self.inputs_outputs[module_name] = [rnn_in, rnn_out]
        module_obj = self._get_layer_by_name(module_name)
        if not module_obj:
            module_obj = next((obj for obj in self.buml_model.sub_nns if obj.name == module_name), None)

        if module_obj:
            self.buml_model.modules.append(module_obj)

        self.previous_assign = node

    def _extract_tuple_target_vars(self, node, module_name):
        """Extract and track output/hidden variables from tuple assignment targets."""
        # Handle cases like:
        # out, h = self.rnn(x)  # 2 elements
        # out, h, c = self.lstm(x)  # 3 elements (LSTM)

        num_targets = len(node.targets[0].elts)

        if num_targets == 2:
            # SimpleRNN or GRU: out, h = self.rnn(x)
            var1 = node.targets[0].elts[0].id if isinstance(node.targets[0].elts[0], ast.Name) else None
            var2 = node.targets[0].elts[1].id if isinstance(node.targets[0].elts[1], ast.Name) else None

            if var1 and var1 != "_":
                self.rnn_output_vars[module_name] = var1
                self.module_of_output[var1] = module_name
            if var2 and var2 != "_":
                self.rnn_hidden_vars[module_name] = var2
                self.module_of_output[var2] = module_name

        elif num_targets == 3:
            # LSTM: out, h, c = self.lstm(x)
            var1 = node.targets[0].elts[0].id if isinstance(node.targets[0].elts[0], ast.Name) else None
            var2 = node.targets[0].elts[1].id if isinstance(node.targets[0].elts[1], ast.Name) else None
            var3 = node.targets[0].elts[2].id if isinstance(node.targets[0].elts[2], ast.Name) else None

            if var1 and var1 != "_":
                self.rnn_output_vars[module_name] = var1
                self.module_of_output[var1] = module_name
            if var2 and var2 != "_":
                self.rnn_hidden_vars[module_name] = var2
                self.module_of_output[var2] = module_name
            if var3 and var3 != "_":
                # Track cell state for LSTM
                self.module_of_output[var3] = module_name

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

    def _extract_rnn_input_arg(self, node):
        """
        Extract input argument from RNN call, handling both variables and inline operations.

        Also detects initial hidden state parameter for seq2seq/decoder patterns.
        Returns: input_var (str)
        Side effect: May add migration warnings for initial states
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
        # In TensorFlow Keras, initial_state parameter is typically passed as keyword argument
        # But we check positional args too for completeness
        if len(node.value.args) > 1:
            self.migration_warnings.append(
                f"Line {node.lineno}: RNN called with additional positional arguments. "
                f"TensorFlow Keras RNNs typically use initial_state as a keyword argument. "
                f"Manual review recommended."
            )

        return input_var

    def handle_forward_slicing(self, node: ast.Assign):
        """
        Handle slicing operations such as:
        - x = out[:, -1]  # Last timestep of RNN output
        - x = out[:, -1, :]  # Last timestep with explicit feature dimension
        - x = tensor[0]  # Regular tensor slicing

        Parameters:
            node (ast.Assign): The AST node representing the slicing assignment

        Returns:
            None, but populates the BUML model
        """
        if not isinstance(node.value.value, ast.Name):
            # Complex subscript base, handle as regular subscript
            subscripted_var = None
        else:
            subscripted_var = node.value.value.id

        result_var = node.targets[0].id

        # Look up which module produced this variable
        if not subscripted_var or subscripted_var not in self.module_of_output:
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        prev_module_name = self.module_of_output[subscripted_var]
        lyr_obj = self._get_layer_by_name(prev_module_name)

        # If not an RNN layer, handle as regular subscript
        if not lyr_obj or not hasattr(lyr_obj, 'return_type'):
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
            return

        # For RNN layers, determine if this is a timestep selection
        if self._is_timestep_selection(node):
            # This might indicate we need return_type='last'
            # But in TensorFlow, return_type is already set during layer creation
            # We just create the subscript TensorOp
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)
        else:
            # Regular slicing on RNN output
            self._handle_non_rnn_slicing(node, subscripted_var, result_var)

        # Track the result variable
        self.module_of_output[result_var] = prev_module_name
        self.variable_aliases[result_var] = subscripted_var
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
            ops_args = node.value.args[0].elts

            # Extract variables (handles subscripts, calls, N-args)
            variables = self._extract_concat_variables(ops_args, node)
            if variables is None:
                return None

            # Resolve aliases
            actual_vars = self._resolve_concat_variables(variables)

            # Get source layers
            layers_of_tensors = []
            for var in actual_vars:
                # Handle inline layer call markers
                if isinstance(var, str) and var.startswith("INLINE_CALL:"):
                    layers_of_tensors.append(var)  # Pass marker directly
                elif var in self.module_of_output:
                    layers_of_tensors.append(self.module_of_output[var])
                else:
                    self.migration_warnings.append(
                        f"Line {node.lineno}: Variable '{var}' not found in module outputs."
                    )
                    return None

            # Get concatenation dimension
            cat_dim = self.param_value(node.value.keywords[0].value)

            # Track RNN var types for N variables
            var_types = self._determine_rnn_var_types_multi(layers_of_tensors, actual_vars)

            tensorop_param = {
                "tns_type": "concatenate",
                "layers_of_tensors": layers_of_tensors,
                "concatenate_dim": cat_dim,
                "actual_vars": var_types
            }
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
        elif op_type == "reduce_mean":
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

            # Extract source variable
            if len(op_args) > 0:
                if isinstance(op_args[0], ast.Name):
                    source_var = op_args[0].id
                    if source_var in self.module_of_output:
                        source_layers = [self.module_of_output[source_var]]
                    elif source_var == 'x':
                        source_layers = ['INPUT']
                    else:
                        self.migration_warnings.append(
                            f"Line {node.lineno}: Source variable '{source_var}' not found."
                        )
                        return None
                else:
                    self.migration_warnings.append(
                        f"Line {node.lineno}: reduce_mean operand type not supported."
                    )
                    return None
            else:
                self.migration_warnings.append(
                    f"Line {node.lineno}: reduce_mean requires an input tensor."
                )
                return None

            tensorop_param = {
                "tns_type": "mean",
                "reduce_dim": reduce_dim,
                "layers_of_tensors": source_layers
            }
        else:
            print(f"{op_type} is not recognized!")

        if tensorop_param:
            op_name = f"op_{self.tensor_op_counter}"
            tensorop_param["name"] = op_name
            tns_obj = getattr(mm_classes, "TensorOp")(**tensorop_param)
            self.buml_model.add_tensor_op(tns_obj)
            self.tensor_op_counter+=1

            # Track output variable
            if hasattr(node, 'targets') and len(node.targets) > 0:
                target_var = node.targets[0].id
                self.module_of_output[target_var] = op_name

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

    # Track return_sequences and return_state to determine return_type
    has_return_sequences = lyr_params.get("return_sequences", False)
    has_return_state = lyr_params.get("return_state", False)

    for param in lyr_params:
        if param == "activation":
            updated_lyr_params["actv_func"] = lyr_params[param]
        elif param in ["return_sequences", "return_state"]:
            # Skip these, handled below to determine return_type
            pass
        elif param in params_mapping:
            param_name = params_mapping[param]
            updated_lyr_params[param_name] = lyr_params[param]
        elif param == "units":
            pass
        else:
            print(f"parameter {param} of layer {lyr_type} is not found!")

    # Determine return_type based on return_sequences and return_state
    if lyr_type in rnn_layers:
        if has_return_sequences and has_return_state:
            updated_lyr_params["return_type"] = "both"
        elif has_return_sequences:
            updated_lyr_params["return_type"] = "full"
        elif has_return_state:
            updated_lyr_params["return_type"] = "hidden"
        else:
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

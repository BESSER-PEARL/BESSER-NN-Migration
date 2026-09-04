"""
Module providing a class that extracts information from the AST of a 
neural network written in TensorFlow or PyTorch and transforms it to
a BUML model.
It also extracts data and model configuration attributes.
"""

import ast
import sys
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
from abc import abstractmethod
from besser.BUML.metamodel.nn import NN, Layer
from transform_code import set_remaining_params


class ASTParser(ast.NodeVisitor):
    """
    Class visiting and parsing the AST.

    Attributes:
        input_nn_type (str): The type of the nn input architecture.
        only_nn (str): Whether to process only the model definition or also
            its configuration and dataset.
        buml_model (NN): The BUML NN model.
        previous_assign (ast.AST | None): It keeps track of the previous 
            module in the forward method.
        data_config (dict): A dict to keep track of NN config and
            data attributes.
        module_of_output (dict): It keeps track of name of layers given their
            output var.
        tensor_op_counter (int): Counter used to assign names of tensorops.
        in_class (bool): It tracks the processing of NN architecture class.
        unprocessed_nodes (list): It keeps track of unprocessed nodes to 
            retrieve other variables later, like 'image_size'. 

    """
    def __init__(self, input_nn_type: str, only_nn: bool):
        super().__init__()

        self.input_nn_type: str = input_nn_type
        self.only_nn: bool = only_nn
        self.buml_model: NN = NN(name="my_nn")  # the name will be updated later
        self.previous_assign: ast.AST | None = None

        self.data_config: dict = {"config": {}, "train_data": {},
                                  "test_data": {}}
        self.module_of_output: dict = {}
        self.tensor_op_counter: int = 1
        self.in_class: bool = False
        self.unprocessed_nodes: list = []
        # Accumulate warnings about unsupported features
        self.migration_warnings: list = []
        # Track variable usage counts in forward method for input_reused detection
        self.variable_usage_count: dict = {}
        self.forward_method_body: list = []


    def _analyze_variable_usage(self, forward_method):
        """
        Analyze variable usage in the forward method to determine which variables
        are used multiple times. This helps set input_reused correctly for TensorOps.
        """
        for stmt in forward_method.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    var_name = node.id
                    self.variable_usage_count[var_name] = self.variable_usage_count.get(var_name, 0) + 1

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        It visits a ClassDef node in the AST, representing a class 
        definition in the source code. This method is called for each 
        class definition encountered in the AST (in case of 'subclassing'
        architecture, the model is defined inside a class). It extracts 
        the name of the NN, and visits its init and forward methods
        to create the BUML model.

        Parameters:
            node (ast.ClassDef): The AST node representing a class definition.

        Returns:
            None, but the the buml model is created. 
        """

        # Retrieve the name of the model
        if not node.bases:
            # Class has no base classes, skip it
            return

        if isinstance(node.bases[0], ast.Attribute):
            base_name = node.bases[0].attr
        elif isinstance(node.bases[0], ast.Name):
            base_name = node.bases[0].id
        else:
            # Unknown base class type
            return

        if base_name == "Model" or base_name == "Module":
            self.buml_model.name = node.name
            self.in_class = True

        # First pass: Find and analyze forward method for variable usage
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == 'forward':
                self._analyze_variable_usage(child)
                self.forward_method_body = child.body
                break

        # Second pass: Process all methods normally
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self.visit(child)

        # Set 'input_reused' and 'name_module_input' layer parameters
        self.set_remaining_lyr_params()
        # Add permute before and after conv blocks to make TF and PyTorch equivalent
        self.add_permute_dim()
        # Add squeeze after global pooling layers (TF to PyTorch specific)
        if hasattr(self, 'add_squeeze_for_global_pooling'):
            self.add_squeeze_for_global_pooling()
        # Renumber functional layers (f_*) to start from 1
        self._renumber_functional_layers()
        # Set in_class var to False at the end of NN architecture processing
        self.in_class = False

    def _renumber_functional_layers(self):
        """Renumber functional layers: if only one, remove suffix; if multiple, renumber from 1."""
        # Group functional layers by base name (e.g., f_adaptive_max_pool1d)
        func_layer_groups = {}
        for module in self.buml_model.modules:
            if hasattr(module, 'name') and module.name.startswith('f_'):
                # Extract base name: f_adaptive_max_pool1d_5 -> f_adaptive_max_pool1d
                parts = module.name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_name = parts[0]
                    if base_name not in func_layer_groups:
                        func_layer_groups[base_name] = []
                    func_layer_groups[base_name].append(module)

        # Track name changes for updating TensorOp references
        name_mapping = {}

        # Renumber: if only one, remove suffix; if multiple, number from 1
        for base_name, modules_list in func_layer_groups.items():
            if len(modules_list) == 1:
                # Only one layer of this type - remove suffix
                module = modules_list[0]
                old_name = module.name
                new_name = base_name  # No suffix
                module.name = new_name
                name_mapping[old_name] = new_name
                # Update references
                for var, mod_name in list(self.module_of_output.items()):
                    if mod_name == old_name:
                        self.module_of_output[var] = new_name
            else:
                # Multiple layers - renumber from 1
                for idx, module in enumerate(modules_list, start=1):
                    old_name = module.name
                    new_name = f"{base_name}_{idx}"
                    module.name = new_name
                    name_mapping[old_name] = new_name
                    for var, mod_name in list(self.module_of_output.items()):
                        if mod_name == old_name:
                            self.module_of_output[var] = new_name

        # Update layers_of_tensors in TensorOps to reflect renamed layers
        for module in self.buml_model.modules:
            if hasattr(module, 'layers_of_tensors') and module.layers_of_tensors:
                updated_layers = []
                for layer_ref in module.layers_of_tensors:
                    if layer_ref in name_mapping:
                        updated_layers.append(name_mapping[layer_ref])
                    else:
                        updated_layers.append(layer_ref)
                module.layers_of_tensors = updated_layers


    def set_remaining_lyr_params(self):
        """
        It iterates through the layers and tensorops to set their 'input_reused'
        and 'name_module_input' parameters. Also resolves multiple return values.
        Also marks modules to prevent variable reuse for residual connections.
        """
        # First pass: identify modules whose outputs are used in binops
        binop_source_modules = set()
        for module in self.buml_model.modules:
            if hasattr(module, 'tns_type') and 'binop' in module.tns_type:
                if hasattr(module, 'layers_of_tensors'):
                    for source in module.layers_of_tensors:
                        if isinstance(source, str) and not source.startswith('op_'):
                            binop_source_modules.add(source)

        # Second pass: mark modules that use binop sources as input
        # We need to mark the module that IMMEDIATELY FOLLOWS the binop source
        # so it doesn't reuse the source's output variable
        for i, module in enumerate(self.buml_model.modules):
            # Check if this module takes input from a binop source module
            if hasattr(module, 'name_module_input') and module.name_module_input:
                if module.name_module_input in binop_source_modules:
                    module.input_reused = True

        # Third pass: set remaining params
        for i, module in enumerate(self.buml_model.modules):
            if isinstance(module, Layer) or hasattr(module, 'input_reused'):
                #                    self.module_of_output,
                #                    self.buml_model.modules, i)
                set_remaining_params(module, self.module_of_output,
                                   self.buml_model.modules, i)

        # Handle tuple returns
        if hasattr(self, 'pytorch_return_vars') and self.pytorch_return_vars:
            # Store actual return variable names
            return_vars = []
            for pytorch_var in self.pytorch_return_vars:
                if pytorch_var == '_':
                    continue
                return_vars.append(pytorch_var)

            if return_vars:
                self.buml_model.return_vars = ', '.join(return_vars)


    def add_permute_dim(self):
        """
        It permutes input and output of PyTorch cnn layers if needed
        to make PyTorch and Tensorflow equivalent.
        It is only applied to TensorFlow code and implemented in the
        ASTParserTF child class.
        """


    def visit_For(self, node: ast.For):
        """
        It visits a For node in the AST, representing a for loop in the 
        source code. It collects information and stores it in instance 
        attributes or other data structures as needed.

        Parameters:
            node(ast.For): The AST node representing a for loop.

        Returns:
            None, but collects some config and data attributes and stores 
                them in self.data_config dictionary.
        """
        if self.only_nn is False:
            if (isinstance(node.iter, ast.Name) and
                  isinstance(node.target, ast.Name) and
                  node.body):
                if isinstance(node.body[0], ast.Assign):
                    if (node.iter.id == "metrics" and
                        isinstance(node.body[0].value, ast.List)):
                        dt_conf = self.data_config["train_data"]
                        dt_conf["task_type"] = "multi_class"

                    elif (node.iter.id == "metrics" and
                          "classification" in self.data_config["config"]):
                        self.data_config["train_data"]["task_type"] = "binary"

            if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Name) and
                    node.iter.func.id == "range" and
                    node.iter.args):
                # Handle both constant values and variable references
                if isinstance(node.iter.args[0], ast.Constant):
                    self.data_config["config"]["epochs"] = node.iter.args[0].value
                elif isinstance(node.iter.args[0], ast.Name):
                    # Variable reference - skip setting epochs in config
                    pass

        else:
            self.generic_visit(node)



    def visit_Assign(self, node: ast.Assign):
        """
        It visits an Assign node in the AST, representing an assignment 
        statement in the source code. This method processes assignments
        by visiting nodes where values are assigned to variables.


        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model and collects attributes 
                for config and data in data_config dict.
        """
        if self.input_nn_type == "subclassing":
            if self.in_class:
                self.handle_subclassing_nn(node)

        elif self.input_nn_type == "sequential":
            self.handle_sequential_nn(node)

        if not self.only_nn:
            self.handle_outer_assignments(node)


    def visit_AnnAssign(self, node: ast.AnnAssign):
        """
        It visits annotated assigments. Checks if the model is annotated.

        Parameters:
            node (ast.AnnAssign): The AST node representing an annotated
                assignment statement.

        Returns:
            None, but it populates the buml model.
        """
        if self.input_nn_type == "sequential":
            self.handle_sequential_nn(node)


    def _create_synthetic_assign(self, node, var_name='_return_output'):
        """Create a synthetic assignment node for return expressions."""
        synthetic_assign = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=node.value
        )
        synthetic_assign.lineno = node.lineno
        synthetic_assign.col_offset = node.col_offset
        return synthetic_assign

    def _handle_tuple_return(self, node):
        """Handle tuple return statements (e.g., return a, b)."""
        pytorch_return_vars = [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
        if not hasattr(self, 'pytorch_return_vars'):
            self.pytorch_return_vars = []
        self.pytorch_return_vars = pytorch_return_vars

    def visit_Return(self, node: ast.Return):
        """
        It visits return statements. Converts complex return expressions into synthetic
        assignments to ensure they are fully processed.

        The variable name used for the synthetic assignment doesn't matter since it's the
        last operation in forward() and won't be used elsewhere. The key is ensuring the
        expression is fully processed and preserved.

        Parameters:
            node (ast.Return): The AST node representing a return statement.

        Returns:
            None, but processes the return value and tracks multiple return values.
        """
        if not self.in_class or self.input_nn_type != "subclassing":
            return

        if isinstance(node.value, (ast.Call, ast.BinOp, ast.Subscript, ast.UnaryOp, ast.IfExp)):
            synthetic_assign = self._create_synthetic_assign(node)
            self.visit_Assign(synthetic_assign)
        elif isinstance(node.value, (ast.Tuple, ast.List)):
            self._handle_tuple_return(node)
        elif isinstance(node.value, ast.Name):
            pass
        elif isinstance(node.value, (ast.Constant, ast.Num)) or node.value is None:
            pass
        else:
            synthetic_assign = self._create_synthetic_assign(node)
            self.visit_Assign(synthetic_assign)


    def handle_sequential_nn(self, node: ast.Assign):
        """
        It handles the sequential NN model architecture.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.

        """
        seq = False
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id == "Sequential":
                    seq = True
            elif isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "Sequential":
                    seq = True
        if seq:
            target = (
                node.targets[0] if hasattr(node, 'targets') else node.target
            )
            if isinstance(target, ast.Name):
                self.buml_model.name = target.id

            elif isinstance(target, ast.Attribute):
                self.buml_model.name = target.attr

            self.handle_sequential_layers(node, self.buml_model.name)

            self.add_permute_dim()
            # Add squeeze after global pooling layers (TF to PyTorch specific)
            if hasattr(self, 'add_squeeze_for_global_pooling'):
                self.add_squeeze_for_global_pooling()



    @abstractmethod
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



    def handle_subclassing_nn(self, node: ast.Assign):
        """
        It is used to visit assignment nodes inside the NN class in
        subclassing architecture.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """
        # Init method
        if isinstance(node.targets[0], ast.Attribute):
            self.handle_init(node)

        #Forward method, simple calls
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.Call)):
            self.handle_forward_simple_call(node)

        #Forward method, attribute access on call result (e.g., x.max(dim=1).values)
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.Attribute) and
              isinstance(node.value.value, ast.Call)):
            self.handle_forward_simple_call(node)

        #Forward RNN
        elif (isinstance(node.targets[0], ast.Tuple) and
              isinstance(node.value, ast.Call)):
            self.handle_forward_tuple_assignment(node)

        #Forward binary operations
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.BinOp)):
            self.handle_forward_binop(node)

        #Forward shape unpacking (e.g., b, t, _ = x.shape)
        elif (isinstance(node.targets[0], ast.Tuple) and
              isinstance(node.value, ast.Attribute)):
            self.handle_forward_shape_unpacking(node)

        #Forward RNN
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.Subscript)):
            self.handle_forward_slicing(node)

        #Forward method, binary operations (add, subtract, etc.)
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.BinOp)):
            self.handle_forward_binop(node)

        #Forward method, simple variable assignments (e.g., inp = x)
        elif (isinstance(node.targets[0], ast.Name) and
              isinstance(node.value, ast.Name)):
            self.handle_forward_variable_assignment(node)


    @abstractmethod
    def handle_init(self, node: ast.Assign):
        """
        It handles the statements in the NN init method.

        Parameters:
            node (ast.Assign): The AST node representing an assignment 
                statement.

        Returns:
            None, but populates the BUML model. 
        """

    @abstractmethod
    def handle_forward_simple_call(self, node: ast.Assign):
        """
        This method:
        - retrieves the input and output variables of modules
        and populates 'module_of_output' dictionary.
        - sets the activation function as attribute of its layer (for PyTorch).
        - adds permute_in attributes to cnn layers if they are preceeded by
        permute tensorop (the permute op is sometimes used before a cnn layer
        to make pytorch and tensorflow models equivalent as cnn in both
        frameworks receive data in a different order). Relevant for PyTorch.
        - sets the order of modules in buml model and processes tensorops.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """

    @abstractmethod
    def handle_forward_shape_unpacking(self, node: ast.Assign):
        """
        Handle tuple unpacking of shape attributes (e.g., b, t, _ = x.shape).

        Parameters:
            node (ast.Assign): The AST node with tuple target and attribute value.

        Returns:
            None, but should track shape variables for use in reshape operations.
        """

    def handle_forward_tuple_assignment(self, node: ast.Assign):
        """
        This method is relevant for PyTorch.
        It handles rnn tuple assignments such as 
        'x, _ = self.l4(x)'
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment 
                statement.

        Returns:
            None, but populates the BUML model. 
        """

    def handle_forward_slicing(self, node: ast.Assign):
        """
        This method is relevant for PyTorch.
        It handles rnn slicing calls such as 'x = x[:, -1, :]'
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment 
                statement.

        Returns:
            None, but populates the BUML model. 
        """


    def param_value(self, param: ast.Call):
        """
        Get the value of a parameter based on its type.
        
        Parameters:
            node (ast.Call): The AST node representing a call statement.

        Returns:
            The value of the parameter.
        """
        if isinstance(param, ast.Constant):
            return param.value
        elif isinstance(param, ast.Tuple):
            values = [self.param_value(el) for el in param.elts]
            return values
        elif isinstance(param, ast.List):  # List values
            values = [self.param_value(el) for el in param.elts]
            return values
        elif (isinstance(param, ast.UnaryOp) and
            isinstance(param.op, ast.USub)):  # Negative numbers
            # Handle UnaryOp with USub for negative numbers
            if isinstance(param.operand, ast.Constant):
                return -param.operand.value
        elif isinstance(param, ast.Name):
            return param.id
        elif isinstance(param, ast.Attribute):
            value = self.param_value(param.value)
            return f"{value}.{param.attr}"
        elif isinstance(param, ast.Call):
            func_name = self.param_value(param.func)
            args = ", ".join(str(self.param_value(arg)) for arg in param.args)
            return f"{func_name}({args})"
        elif isinstance(param, ast.BinOp):
            # Handle binary operations like a * b, a + b, etc.
            left = self.param_value(param.left)
            right = self.param_value(param.right)
            op_map = {
                ast.Mult: '*',
                ast.Add: '+',
                ast.Sub: '-',
                ast.Div: '/',
                ast.FloorDiv: '//',
                ast.Mod: '%',
            }
            op_symbol = op_map.get(type(param.op), '?')
            return f"{left} {op_symbol} {right}"

        self.migration_warnings.append(
            f"Unhandled parameter type '{type(param).__name__}'. Parameter value may not be correctly migrated."
        )
        return param

    def extract_layer_params(self, call_node: ast.Call):
        """
        This method extracts the layers attributes (keywords 
        and positional params) and returns them as a dictionary.

        Parameters:
            node (ast.Call): The AST node representing a call statement.

        Returns:
            A dictionary containing the parameters.
        """
        params = []
        keyword_params = {}
        for arg in call_node.args:
            value = self.param_value(arg)
            if value is not None:
                params.append(value)

        for keyword in call_node.keywords:
            value = self.param_value(keyword.value)
            if value is not None:
                keyword_params[keyword.arg] = value
        keyword_params["positional_params"] = params
        return keyword_params


    def extract_layer(self, call_node: ast.Call):
        """
        It extracts the layer type and its parameters.
        
        Parameters:
            node (ast.Call): The AST node representing a call statement.

        Returns:
            The layer type and its parameters.
        """
        if isinstance(call_node.func, ast.Name):
            layer_type = call_node.func.id
        else:  # isinstance(call_node.func, ast.Attribute):
            layer_type = call_node.func.attr

        params = self.extract_layer_params(call_node)
        return layer_type, params

    @abstractmethod
    def handle_forward_binop(self, node: ast.Assign):
        """
        It handles binary operations such as 'x = a + b' or 'x = a.squeeze(1) + b'

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the BUML model.
        """

    @abstractmethod
    def extract_tensorop(self, node: ast.Assign):
        """
        It extracts the tensorop name and its parameters.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but populates the buml model.
        """



    @abstractmethod
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


    def handle_outer_simple_assignment(self, node: ast.Assign):
        """
        It extracts information from simple assignment statements
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config and data in
                data_config dict.
        """
        # Validate node structure before accessing attributes
        if not (isinstance(node.value, ast.Call) and
                hasattr(node.value, 'func') and
                hasattr(node.value.func, 'id')):
            return

        if node.value.func.id == self.buml_model.name:
            self.buml_model.name = node.targets[0].id
        elif node.value.func.id == "classification_report":
            self.data_config["config"]["classification"] = True
        elif node.value.func.id == "mean_absolute_error":
            self.data_config["train_data"]["task_type"] = "regression"
            self.data_config["config"]["metrics"] = ["mae"]
        elif node.value.func.id == "compute_mean_std":
            self.get_images_attr(node)
        elif (isinstance(node.targets[0], ast.Name) and
              node.value.args and
              isinstance(node.value.args[0], ast.Constant) and
              isinstance(node.value.args[0].value, str)):
            path = node.value.args[0].value
            target_id = node.targets[0].id
            if "train" in target_id:
                self.data_config["train_data"]["path_data"] = path
                if path.endswith("csv"):
                    self.data_config["train_data"]["input_format"] = "csv"
            elif "test" in target_id:
                self.data_config["test_data"]["path_data"] = path


    def handle_outer_constant_assignment(self, node: ast.Assign):
        """
        It extracts information from constant assignment statements
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config and data in
                data_config dict.
        """
        # Validate node structure
        if not (node.targets and
                isinstance(node.targets[0], ast.Name) and
                isinstance(node.value, ast.Constant)):
            return

        if node.targets[0].id == "batch_size":
            self.data_config["config"]["batch_size"] = node.value.value
        elif node.targets[0].id == "train_path":
            path = node.value.value
            if path.endswith("csv"):
                self.data_config["train_data"]["input_format"] = "csv"
            self.data_config["train_data"]["path_data"] = path
        elif node.targets[0].id == "test_path":
            self.data_config["test_data"]["path_data"] = node.value.value
        elif node.targets[0].id == "epochs":
            self.data_config["config"]["epochs"] = node.value.value
        elif node.targets[0].id == "image_size":
            self.data_config["train_data"]["images_size"] = node.value.value


    def handle_outer_tuple_assignment(self, node: ast.Assign):
        """
        It extracts information from tuple assignment statements
        called outside the NN class.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects images_size attribute in
                data_config dict.
        """
        # Validate node structure
        if not (node.targets and
                isinstance(node.targets[0], ast.Name) and
                isinstance(node.value, (ast.Tuple, ast.List))):
            return

        if node.targets[0].id.lower() == "image_size":
            size = [i.value for i in node.value.elts]
            self.data_config["train_data"]["images_size"] = size


    @abstractmethod
    def get_images_attr(self, node: ast.Assign):
        """
        It extracts information related to images.
        
        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for data in data_config dict.
        """


    def get_params_from_optimizer(self, node: ast.Assign):
        """
        It extracts information related to the optimizer.

        Parameters:
            node (ast.Assign): The AST node representing an assignment
                statement.

        Returns:
            None, but collects attributes for config in data_config dict.
        """
        # Validate node structure
        if not (isinstance(node.value, ast.Call) and
                hasattr(node.value, 'func') and
                hasattr(node.value.func, 'attr')):
            return

        self.data_config["config"]["optimizer"] = node.value.func.attr.lower()
        keywords = node.value.keywords
        learning_rate = next(
            (k.value.value for k in keywords
             if k.arg in {'learning_rate', 'lr'}),
             None)
        momentum = next(
            (k.value.value for k in keywords if k.arg == 'momentum'),
            None)
        weight_decay = next(
            (k.value.value for k in keywords if k.arg == 'weight_decay'),
            None)
        self.data_config["config"]["learning_rate"] = learning_rate
        if momentum:
            self.data_config["config"]["momentum"] = momentum
        if weight_decay:
            self.data_config["config"]["weight_decay"] = weight_decay

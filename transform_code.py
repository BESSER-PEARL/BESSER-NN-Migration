"""
Helper functions to transform PyTorch and TensorFlow nn code to BUML model.
"""

import argparse
import ast
import sys
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
from typing import TYPE_CHECKING

from besser.BUML.metamodel.nn import Configuration, Dataset, Image
from besser.BUML.metamodel.nn import Layer, NN
from tf2torch.input_shape_retriever import update_model


if TYPE_CHECKING:
    from ast_parser_nn import ASTParser

def parse_arguments_transform():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Read a NN file and convert it to buml code."
    )
    parser.add_argument(
        "filename", type=str, help="Path to the file to read"
    )
    parser.add_argument(
        "typeinput", type=str, default="subclassing",
        help="Type of the input NN architecture."
    )
    parser.add_argument(
        "typeoutput", type=str, default="subclassing",
        help="Type of the output NN architecture."
    )
    parser.add_argument(
        "--onlynn", type=str2bool, help="Whether the file contains only \
            NN def or also the code for training and evaluation.",
            default=True, const=True, nargs='?'
    )
    parser.add_argument(
        "--datashape", type=parse_tuple, default=None,
        help=(
            "The shape of the input data (optional)."
            "It is needed when transforming tf code to pytorch code"
            "It is used to recover some layer attributes dynamically"
            "If the migrated script defines a dataset, it can be skipped"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/migrated_nn",
        help="The output directory where the migrated file will be saved"
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="The base name of the output file (e.g., 'my_model.py'). The generation type (subclassing/sequential) will be added as suffix (e.g., 'my_model_subclassing.py')"
    )

    return parser.parse_args()

def str2bool(v):
    """
    Parse bool from a string input

    Parameters:
    ----------
    v(str): The bool in a string format.

    Returns:
    -------
    The bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_tuple(value: str):
    """
    Parse a tuple from a string input
    
    Parameters:
    ----------
    value(str): The tuple in a string format.

    Returns:
    -------
    The tuple
    """
    return (1,) + tuple(map(int, value.strip("()").split(",")))

def _display_migration_warnings(extractor):
    """Display migration warnings if any exist."""
    if not (hasattr(extractor, 'migration_warnings') and extractor.migration_warnings):
        return

    print("\n" + "="*60)
    print("MIGRATION WARNINGS")
    print("="*60)
    print(f"\nFound {len(extractor.migration_warnings)} issue(s) during migration:\n")
    for i, warning in enumerate(extractor.migration_warnings, 1):
        print(f"{i}. {warning}")
    print("\n" + "="*60)
    print("The migration will continue, but please review these warnings.")
    print("Some PyTorch features may not be fully supported in the TensorFlow output.")
    print("="*60 + "\n")

def _restructure_sequential_model(buml_model):
    """Restructure sequential model from sub_nn to main model."""
    nn_obj = next((obj for obj in buml_model.sub_nns if obj.name == buml_model.name), None)
    if not nn_obj:
        return

    buml_model.modules.clear()
    buml_model.layers.clear()
    for module in nn_obj.modules:
        if isinstance(module, Layer):
            buml_model.add_layer(module)
        elif isinstance(module, NN):
            buml_model.add_sub_nn(module)
        else:
            buml_model.modules.append(module)
    buml_model.sub_nns.remove(nn_obj)

def _add_configuration(buml_model, config):
    """Add configuration to BUML model if exists."""
    if not config:
        return
    if "classification" in config:
        del config["classification"]
    cnf = Configuration(**config)
    buml_model.add_configuration(cnf)

def _add_datasets(buml_model, dt_tr, dt_ts):
    """Add training and test datasets to BUML model if they exist."""
    if not dt_tr:
        return

    train_data = Dataset(name="train_data", path_data=dt_tr["path_data"],
                         task_type=dt_tr["task_type"],
                         input_format=dt_tr["input_format"])
    test_data = Dataset(name="test_data", path_data=dt_ts["path_data"])

    if dt_tr["input_format"] == "images":
        if "normalize_images" not in dt_tr:
            dt_tr["normalize_images"] = False
        img = Image(shape=dt_tr["images_size"], normalize=dt_tr["normalize_images"])
        train_data.add_image(img)

    buml_model.add_train_data(train_data)
    buml_model.add_test_data(test_data)

def transform(args: argparse.Namespace, framework: str, ast_parser_class: 'ASTParser'):
    """
    It gets the AST and transforms it to a buml model.

    Parameters:
        args(argparse.Namespace): An object with arg attributes.
            framework (str): "TF" or "PyTorch".
        ast_parser_class ('ASTParser'): The class to use to parse the AST

    Returns:
        The buml model and its output architecture type (i.e., sequential
            or subclassing). It is specified in the config file.
    """
    with open(args.filename, "r", encoding="utf-8") as file:
        code = file.read()

    tree = ast.parse(code)
    extractor = ast_parser_class(args.typeinput, args.onlynn)
    extractor.visit(tree)
    buml_model = extractor.buml_model

    _display_migration_warnings(extractor)

    if args.typeinput == "sequential":
        # For sequential models, set_remaining_lyr_params was not called during visit
        # (it's only called in visit_ClassDef for subclassing), so call it here
        extractor.set_remaining_lyr_params()
        _restructure_sequential_model(buml_model)

    if framework == "TF":
        update_model(args.typeinput, buml_model, args.filename, args.datashape)

    _add_configuration(buml_model, extractor.data_config["config"])
    _add_datasets(buml_model, extractor.data_config["train_data"], extractor.data_config["test_data"])

    return buml_model, args.typeoutput


def param_to_list(lyr_type: str, lyr_params: dict, params_to_convert: list,
                  layers_of_params: list):
    """
    It converts int parameters to list format for buml model

    Parameters:
        lyr_type (str): The type of the layer.
        lyr_params (dict): A dictionary of all the layer parameters and their
            values.
        params_to_convert (list): The list of parameters to be converted.
        layers_of_params (list): The list of layers that need their params
            to be converted.

    Returns:
        None
    """

    if lyr_type in layers_of_params:
        for param in lyr_params:
            # Check for 2D/3D layers (case-insensitive: Conv2d, Conv2D, MaxPool2d, MaxPool2D, etc.)
            if lyr_type.lower().endswith("2d") or lyr_type.lower().endswith("3d"):
                # For 2D/3D layers, convert any param in params_to_convert to a list of appropriate dimension
                if (param in params_to_convert and
                    isinstance(lyr_params[param], int)):
                    dim = 2 if lyr_type.lower().endswith("2d") else 3
                    lyr_params[param] = [lyr_params[param]] * dim
            else:
                # For 1D layers, convert to single-element list
                if (param in params_to_convert and
                    isinstance(lyr_params[param], int)):
                    lyr_params[param] = [lyr_params[param]]


def process_positional_params(lyr_type: str, lyr_params: dict,
                              pos_params: dict):
    """
    It processes the positional parameters to convert them
    to keyword parameters.

    Parameters:
        lyr_type (str): The type of the layer.
        lyr_params (dict): A dictionary of all the layer parameters
            and their values.
        pos_params (dict): A dictionary storing the as keys layers 
            types and as values the names of their positional params.

    Returns:
        None
    """
    lyr_of_pos_parm = next(
        (e for e in pos_params if lyr_type.startswith(e)), None
    )
    if lyr_of_pos_parm and lyr_params["positional_params"]:
        pos_params_with_values = {}
        for i, pos_arg in enumerate(lyr_params["positional_params"]):
            par = pos_params[lyr_of_pos_parm][i]
            pos_params_with_values[par] = pos_arg
        lyr_params.update(pos_params_with_values)
    lyr_params.pop("positional_params")


def set_static_params(lyr_type: str, lyr_params: dict,
                      static_params: dict):
    """
    It handles the parameters that are retrieved from the layer type.
    Ex: For a MaxPool1D layer, 'pooling_type' ('max) and 'dimension ('1D')
    are retrieved.

    Parameters:
        lyr_type (str): The type of the layer.
        lyr_params (dict): A dictionary of all the layer parameters
            and their values.
        static_params (dict): A dictionary storing as keys the layers
            types and as values their fixed params.

    Returns:
        None
    """
    if lyr_type in static_params:
        lyr_params.update(static_params[lyr_type])


def set_remaining_params(lyr_obj: Layer, inputs_outputs: dict,
                         module_of_output: dict, modules_list: list,
                         current_index: int):
    """
    It sets the 'input_reused' and 'name_module_input' layer parameters.

    Parameters:
        lyr_obj (Layer): The buml layer object.
        inputs_outputs (dict): It stores input and output variables
            of layers.
        module_of_output (dict): Maps output variable names to the module
            that produces them.
        modules_list (list): The list of all modules in execution order.
        current_index (int): The index of the current layer in modules_list.

    Returns:
        None
    """

    layer_name = lyr_obj.name

    # Skip if module is not in inputs_outputs
    if layer_name not in inputs_outputs:
        return

    # Skip identity tensorops: they should preserve exact variable names
    if hasattr(lyr_obj, 'tns_type') and lyr_obj.tns_type == 'identity':
        return

    # Handle both single output and tuple output (list)
    input_var = inputs_outputs[layer_name][0]
    output_var = inputs_outputs[layer_name][1]

    # Check if input is different from output(s)
    # For tuple outputs (RNN): check if input is not in the output tuple
    # For single output: check if input != output
    if ((isinstance(output_var, list) and input_var not in output_var) or
        (not isinstance(output_var, list) and input_var != output_var)):
        lyr_obj.input_reused = True

        # Only set name_module_input if it wasn't already set during parsing
        # (to avoid overwriting correct values with stale module_of_output data)
        if not hasattr(lyr_obj, 'name_module_input') or lyr_obj.name_module_input is None:
            # Determine if input is the original network input or from another module
            if input_var in module_of_output:
                # Input variable is produced by some module: check if it comes before or after
                producing_module_name = module_of_output[input_var]
                producing_module = next((m for m in modules_list if m.name == producing_module_name), None)
                if producing_module:
                    producing_index = modules_list.index(producing_module)
                    if producing_index < current_index:
                        # Producing module comes BEFORE (input from that module)
                        lyr_obj.name_module_input = producing_module_name
                    else:
                        # Producing module comes AFTER (input is original network input)
                        lyr_obj.name_module_input = 'INPUT'
            else:
                # Input variable not produced by any module (original network input)
                lyr_obj.name_module_input = 'INPUT'

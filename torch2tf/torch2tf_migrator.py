"""
It converts PyTorch code to BUML code.

Argument:
    filename (str): Path to a PyTorch file containing
        the code to transform.
    configfile (str, optional): Path to the configuration file that
        has the values of 'input_nn_type', 'output_nn_type', and 'only_nn'.
    datashape (str, optional): The shape of the input data (optional).
"""

import sys
import os
sys.path.insert(0, r'C:\Users\daoudi\projects\BESSER')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from besser.generators.nn.tf.tf_code_generator import TFGenerator
from ast_parser_pytorch import (
    ASTParserTorch
)
from transform_code import (
    parse_arguments_transform, transform
)


def main():
    """It transforms PyTorch code to TensorFlow code"""
    args = parse_arguments_transform()

    buml_model, output_nn_type = transform(args, "PyTorch", ASTParserTorch)

    tf_model = TFGenerator(
        model=buml_model,
        output_dir=args.output_dir,
        generation_type=output_nn_type,
        strip_layer_counter_suffix=True  # Migration adds counter suffixes to layer names
    )
    tf_model.generate()

if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Daniel Gao
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "onnx",
# ]
# ///

import argparse
import onnx
import textwrap


def name_pair(arg):
    result = arg.split(',')
    if len(result) != 2:
        raise argparse.ArgumentError('Name change must be two strings separated by a ","')
    if len(result[0]) == 0:
        raise argparse.ArgumentError('Name change must be two strings separated by a ","')
    if len(result[1]) == 0:
        raise argparse.ArgumentError('Name change must be two strings separated by a ","')
    return result


def main(args) -> None:
    onnx_model = onnx.load(args.model)

    input_mapping = {}
    if args.inputs:
        print('Input Mapping:')
        for before, after in args.inputs:
            input_mapping[before] = after
            print(f'{before} -> {after}')
        print()


    output_mapping = {}
    if args.outputs:
        print('Output Mapping:')
        for before, after in args.outputs:
            output_mapping[before] = after
            print(f'{before} -> {after}')
        print()

    print('Processing inputs...')
    for node in onnx_model.graph.input:
        cur_name = node.name
        if cur_name in input_mapping:
            new_name = input_mapping[cur_name]
            print(f'*** {cur_name} -> {new_name}')
            node.name = new_name
        else:
            print(f'    {cur_name}')

    print()
    print('Processing outputs...')
    for node in onnx_model.graph.output:
        cur_name = node.name
        if cur_name in output_mapping:
            new_name = output_mapping[cur_name]
            print(f'*** {cur_name} -> {new_name}')
            node.name = new_name
        else:
            print(f'    {cur_name}')

    print()
    print('Processing node connections...')
    for node in onnx_model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] in input_mapping:
                node.input[i] = input_mapping[node.input[i]]

        for i in range(len(node.output)):
            if node.output[i] in output_mapping:
                node.output[i] = output_mapping[node.output[i]]

    if args.out_model:
        onnx.save(onnx_model, args.out_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Changes ONNX model input/output names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Example usage:

            uv run change_onnx_names.py --model in.onnx \\
                --inputs before,after a,b \\
                --outputs out_a,out_b \\
                --out_model out.onnx

            where "before,after" means the name "before" is changed to "after"
            '''))
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--inputs', type=name_pair, nargs='*')
    parser.add_argument('--outputs', type=name_pair, nargs='*')
    parser.add_argument('--out_model', type=str)

    args = parser.parse_args()

    if not args.inputs and not args.outputs:
        print('No names to change')
    else:
        main(args)

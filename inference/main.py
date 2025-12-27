"""
Main file for CANDI inference,
Imputation and denoising assays

Takes in a bam file of given assays.
Make a temporary directory to save intermediate results of all assays.
Save the requested assays and the lower and upper bounds.
"""

import os
import sys
import shutil
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import argparse
from inputs import process_input_data, load_candi_predictor
from run_model import run_through_model

def inf_arg_parser():
    """
    Argument parser of CANDI inference
    """

    parser = argparse.ArgumentParser(
        description="CANDI: Context-Aware Neural Data Imputation - Inference Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                TODO: add examples here
                    """)

    # === DATA CONFIGURATION ===
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_path', type=str, required=True,
                            help='Path to the input directory.\n' \
                            'Directory should be set as follows:\n' \
                            '<Given_Path>/<ASSAY>/<BAM>.bam')
    data_group.add_argument('--model_path', type=str, required=True,
                            help='Path to the model provided by CANDI.\n' \
                            'Either a model file or a directory')
    data_group.add_argument('--output_path', type=str, default="./",
                            help='Path to the output directory.')
    data_group.add_argument('--temp_path', type=str, default="./temp",
                            help='Path to the temporary folder to save intermediate work.' \
                            'Deleted afterwards')
    data_group.add_argument('--fasta_path', type=str, default="./temp/fasta.fa",
                            help='Path to the fasta file to be used. The standarad hg38.fa\n' \
                            'If it does not exist it will be downloaded and save in this directory')
    data_group.add_argument('--assays', type=str, nargs="+", default=[],
                            help='The assays to save. Default is all assays.\n'\
                                'For the list of available assays, look at README')
    data_group.add_argument('--chromosomes', type=str, nargs="+", default=["chr19"],
                            help='The chromosomes to process. possible choices: 1 to 22 and X.\n'\
                                'eg: "chr19 chr20" or "19 20"')

    # === SYSTEM CONFIGURATION ===
    # TODO: Wait on Mehdi to finish this
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified')
    system_group.add_argument('--check_gpus', action='store_true',
                             help='Check GPU availability and exit')

    # === ADVANCED OPTIONS ===
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--debug', '-D', action='store_true',
                               help='Enable debug mode with extra logging')
    advanced_group.add_argument('--save_extra', type=str, default="",
                               help='How much extra information needs to be saved.\n' \
                                'Default: save the mean.\n' \
                                '"s": save the confidence intervals as well.')
    advanced_group.add_argument('--confidence', type=int, default="95",
                               help='The confidence interval to save with')

    return parser.parse_args()

def main():

    args = inf_arg_parser()
    os.makedirs(args.temp_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    # setup
    chr_sizes = process_input_data(args.data_path, args.temp_path, args.chromosomes)
    model = load_candi_predictor(args.model_path)

    # process
    run_through_model(args, model, chr_sizes)

    #cleanup
    if not args.debug: shutil.rmtree(args.temp_path)

    return

# ------------------------------------------------------------------------
# Extra stuff
# ------------------------------------------------------------------------
# TODO: remove
def tester():
    # download_url = f"https://www.encodeproject.org/files/ENCFF282ZSZ/@@download/ENCFF282ZSZ.bam"
    # save_dir_name = "/home/azr/misc/test.bam"
    # import requests

    # with requests.get(download_url, stream=True) as response:
    #     response.raise_for_status()  # Check for request errors
    #     with open(save_dir_name, 'wb') as file:
    #         # Iterate over the response in chunks (e.g., 8KB each)
    #         for chunk in response.iter_content(chunk_size=int(1e3*1024)):
    #             # Write each chunk to the file immediately
    #             file.write(chunk)
    import torch
    import torch.nn.functional as F
    def split_tensor(window, tensor, factor=1):
        n, d = tensor.shape
        window = window * factor
        stride = window - (window//4) * factor  # 900
        tensor = tensor.unsqueeze(0).permute(0, 2, 1)  # (1, d, n)

        # Unfold into overlapping windows
        patches = F.unfold(tensor, kernel_size=(d, window), stride=(1, stride))  # (d*1200, L)
        b = patches.size(-1)
        windows = patches.view(d, window, b).permute(2, 1, 0)  # (b, 1200, d)

        return windows

    def reconstruct_tensor(window, tensor, true_len):

        b, window, d = tensor.shape # (b, 1200, d)

        windows = tensor.permute(2, 1, 0).reshape(1, d * window, b)

        stride = window - (window // 4)  # 900
        # Reconstruct using fold
        reconstructed = F.fold(
            windows, output_size=(d, true_len), kernel_size=(d, window), stride=(1, stride)
        ).squeeze(0,1).permute(1, 0)  # (n, d)

        # Normalize for overlap
        mask = F.fold(
            torch.ones_like(windows), output_size=(d, true_len), kernel_size=(d, window), stride=(1, stride)
        ).squeeze(0,1).permute(1, 0)
        reconstructed = reconstructed / mask

        return reconstructed

    window = 12
    true_len = 15
    assays = 2
    test_tensor = torch.rand((true_len, assays))

    patched = split_tensor(window, test_tensor, 1)
    repatched = reconstruct_tensor(window, patched, true_len)

    print(torch.allclose(test_tensor, repatched, atol=1e-3))

    print("ALL_DONE")

if __name__=="__main__":
    main()
    # tester()

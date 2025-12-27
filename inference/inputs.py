"""
File to prepare inputs for CANDI inference

Creates the npz files per choromosome as done in CANDI runs
"""

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(1, str(Project_DIR))

import json
import pysam
import shutil
import contextlib
import numpy as np

from pred import CANDIPredictor # external import

ASSAYS=[
    'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac',
    'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3',
    'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac',
    'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac',
    'chipseq-control'
    ]

SEQUENCING_PLATFORMS = [
    'Illumina Genome Analyzer IIx', 'Illumina Genome Analyzer',
    'Illumina Genome Analyzer IIe', 'Illumina HiSeq 2000',
    'Illumina Genome Analyzer II', 'Illumina HiSeq 4000',
    'Illumina HiSeq 2500', 'Illumina Genome Analyzer I',
    'Illumina NextSeq 500'
    ]

RUN_TYPES = ['single-ended', 'paired-ended']

CHR_SIZES_POSSIBLE = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
}

RESOLUTION = 25

# ------------------------------------------------------------------------
# Input processing functions
# ------------------------------------------------------------------------
class BAM_TO_SIGNAL(object):
    """
    Slightly modified BAM_TO_SIGNAL to process input metadata as well
    """

    def __init__(self, bam_file, chr_sizes, input_mdata, output_mdata, resolution=RESOLUTION):

        self.bam_file = bam_file
        self.output_dir = os.path.join("/",*self.bam_file.split('/')[:-1])
        self.chr_sizes = chr_sizes
        self.resolution = resolution
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')
        self.input_mdata = input_mdata
        self.output_mdata = output_mdata

    def initialize_empty_bins(self):
        return {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage_pysam(self):
        bins = self.initialize_empty_bins()

        total_mapped_reads = 0
        bins_with_reads = 0

        read_lens = []

        paired_read_counts = 0
        single_read_counts = 0

        for chr in self.chr_sizes:
            for read in self.bam.fetch(chr):
                if read.is_unmapped:
                    continue
                total_mapped_reads += 1
                read_lens.append(read.reference_length)
                if read.is_paired: paired_read_counts += 1
                else: single_read_counts += 1

                start_bin = read.reference_start // self.resolution
                end_bin = read.reference_end // self.resolution
                for i in range(start_bin, end_bin + 1):
                    if bins[chr][i] == 0:
                        bins_with_reads += 1
                    bins[chr][i] += 1

        # Calculate coverage as the percentage of bins with at least one read
        total_bins = sum(len(b) for b in bins.values())
        coverage = (bins_with_reads / total_bins) if total_bins > 0 else 0

        mean_read_len = np.mean(np.array(read_lens))
        bam_is_paired = paired_read_counts > single_read_counts

        return bins, total_mapped_reads, coverage, mean_read_len, bam_is_paired

    def save_signal_metadata(self, depth, mean_read_len, bam_is_paired):

        file_name = os.path.join(self.output_dir, "input_metadata.json")
        mdict = {
            "depth":depth,
            "sequencing_platform":self.input_mdata["sequencing_platform"],
            "mean_read_len":mean_read_len,
            "run_type":bam_is_paired,
            }

        with open(file_name, 'w') as file:
            json.dump(mdict, file, indent=4)

        file_name = os.path.join(self.output_dir, "output_metadata.json")
        mdict = {
            "depth":depth,
            "sequencing_platform":self.output_mdata["sequencing_platform"],
            "mean_read_len":mean_read_len,
            "run_type":self.output_mdata["run_type"],
            }

        mdict.update(self.output_mdata)

        with open(file_name, 'w') as file:
            json.dump(mdict, file, indent=4)

    def save_signal(self, bins):

        for chr, data in bins.items():
            file_name = os.path.join(self.output_dir, f"{chr}.npz")
            np.savez_compressed(file_name, np.array(data))

    def full_preprocess(self):

        data, depth, _, mean_read_len, bam_is_paired = self.calculate_coverage_pysam()
        self.save_signal(data)
        self.save_signal_metadata(np.log2(depth), mean_read_len, bam_is_paired)

# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
# TODO: metadata
def process_bam(bam_file, input_mdata, output_mdata, chr_sizes):

    bam_processor = BAM_TO_SIGNAL(
        bam_file=bam_file,
        chr_sizes=chr_sizes,
        input_mdata=input_mdata,
        output_mdata=output_mdata
    )
    bam_processor.full_preprocess()

    os.remove(bam_file)
    if os.path.exists(f"{bam_file}.bai"):
        os.remove(f"{bam_file}.bai")

def process_metadata(metadata):

    seq_platform = metadata["sequencing_platform"]
    try:
        index = SEQUENCING_PLATFORMS.index(seq_platform) + 1
    except:
        index = 0
    metadata["sequencing_platform"] = index

    run_type = metadata["run_type"]
    try:
        index = RUN_TYPES.index(run_type)
    except:
        index = -1
    metadata["run_type"] = index

def process_chr_sizes(chromosomes):
    """
    Prepares the sizes of chromosome that will be processes

    @args
        - chromosomes (str list): user given list of chromosomes
    @rets
        - chr_size (str:int dict): sizes of choromosome to be processed
    """

    chr_possible = CHR_SIZES_POSSIBLE.keys()
    chr_sizes = {}

    for chr in chromosomes:

        if not (chr in chr_possible or f"chr{chr}" in chr_possible):
            print(f"Given chromosome: {chr}, is out of scope. Skipping")
            continue

        # Turn int only into fullname
        chr = f"chr{chr}" if chr[:3] != "chr" else chr
        chr_sizes[chr] = CHR_SIZES_POSSIBLE[chr]

    assert len(chr_sizes) != 0, "Need atleast one correct chromosome to process"

    return chr_sizes

def process_input_data(bios_path, temp_path, chromosomes):
    """
    Iterates over the given bios folder and generates preprocessed data
    """

    chr_sizes = process_chr_sizes(chromosomes)
    given_exps = os.listdir(bios_path)

    processed_assays = 0

    for exp in given_exps:

        if exp not in ASSAYS:
            print(f"Unsupported assay {exp}. Skipping")
            continue

        exp_path = os.path.join(bios_path, exp)
        temp_exp_path = os.path.join(temp_path, exp)
        os.makedirs(temp_exp_path, exist_ok=True)

        input_mdata_path = os.path.join(exp_path, "input_metadata.json")
        try:
            with open(input_mdata_path) as f:
                input_mdata = json.load(f)
        except:
            input_mdata = {
                "sequencing_platform" : "N/A",
                "run_type" : "paired"
            }
        process_metadata(input_mdata)

        output_mdata_path = os.path.join(exp_path, "output_metadata.json")
        try:
            with open(output_mdata_path) as f:
                output_mdata = json.load(f)
        except:
            output_mdata = {
                "sequencing_platform" : "N/A",
                "run_type" : "N/A"
            }
        process_metadata(output_mdata)

        # Get the first bam file
        bam_file = next((f for f in os.listdir(exp_path) if f.endswith('.bam')), None)
        if bam_file is None:
            print(f".bam file not found in path: {exp_path}. Skipping")
            continue

        # copy it to temp path and then process
        file_path = os.path.join(exp_path,bam_file)
        temp_file_path = os.path.join(temp_exp_path,bam_file)
        shutil.copy2(file_path, temp_file_path)
        pysam.index(temp_file_path)

        process_bam(temp_file_path, input_mdata, output_mdata, chr_sizes)

        processed_assays += 1

    assert processed_assays > 0, f"Need at least one supported assay to work with."

    return chr_sizes

# ------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------

def load_candi_predictor(model_path):

    # suppress output
    with contextlib.redirect_stdout(None):
        model = CANDIPredictor(model_path)

    return model

# ------------------------------------------------------------------------
# Extra stuff
# ------------------------------------------------------------------------

def main():
    pass

if __name__=="__main__":
    main()

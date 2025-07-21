import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import multiprocessing as mp
import requests, os, itertools, ast, io, pysam, datetime, pyBigWig, time, gzip, pickle, json, subprocess, random, glob, shutil, psutil

from torch.utils.data import Dataset
import torch, sys, math
from intervaltree import IntervalTree
import pybedtools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
# from mpi4py import MPI
import gc
import concurrent.futures
from multiprocessing import Pool
import tracemalloc
from bs4 import BeautifulSoup

import numpy as np
# from scipy.stats import pearsonr, spearmanr
# from prettytable import PrettyTable
# import pyBigWig

def get_binned_vals(bigwig_file, chr, resolution=25):
    with pyBigWig.open(bigwig_file) as bw:
        if chr not in bw.chroms():
            raise ValueError(f"{chr} not found in the BigWig file.")
        chr_length = bw.chroms()[chr]
        start, end = 0, chr_length
        vals = np.array(bw.values(chr, start, end, numpy=True))
        vals = np.nan_to_num(vals, nan=0.0)
        vals = vals[:end - (end % resolution)]
        vals = vals.reshape(-1, resolution)
        bin_means = np.mean(vals, axis=1)
        return bin_means
    
def get_DNA_sequence(chrom, start, end, fasta_file="/project/compbio-lab/encode_data/hg38.fa"):
    """
    Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

    :param fasta_file: Path to the fasta file.
    :param chrom: Chromosome name (e.g., 'chr1').
    :param start: Start position (0-based).
    :param end: End position (1-based, exclusive).
    :return: Sequence string.
    """
    try:
        # Open the fasta file
        fasta = pysam.FastaFile(fasta_file)
        
        # Ensure coordinates are within the valid range
        if start < 0 or end <= start:
            raise ValueError("Invalid start or end position")
        
        # Retrieve the sequence
        sequence = fasta.fetch(chrom, start, end)
        
        return sequence
    except Exception as e:
        print(f"Error retrieving sequence: {e}")
        return None

def dna_to_onehot(sequence):
    # Create a mapping from nucleotide to index
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}
    
    # Convert the sequence to indices
    indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)
    
    # Create one-hot encoding
    one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

    # Remove the fifth column which corresponds to 'N'
    one_hot = one_hot[:, :4]
    
    return one_hot

def download_save(url, save_dir_name):
    try:
        # print(f"downloading {url}")
        # Stream the download; this loads the file piece by piece
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check for request errors
            with open(save_dir_name, 'wb') as file:
                # Iterate over the response in chunks (e.g., 8KB each)
                for chunk in response.iter_content(chunk_size=int(1e3*1024)):
                    # Write each chunk to the file immediately
                    file.write(chunk)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def get_bin_value_dict(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]

    # t1 = datetime.datetime.now()
    vals = bw.values(chr, start, end, numpy=True)
    vals = vals[:end - (end % resolution)]

    vals = vals.reshape(-1, resolution)

    # Compute the mean, but handle empty slices
    with np.errstate(invalid='ignore'):  # suppress warnings for invalid operations
        bin_means = np.nanmean(vals, axis=1)

    bin_means = np.nan_to_num(bin_means, nan=0.0)

    # t2 = datetime.datetime.now()
    # print(f"binning took {t2 - t1} for {chr} of length {end}")

    input_dict["signals"] = bin_means

    if input_dict["bw_obj"] == False:
        bw.close()
        del input_dict["bw"]
        
    return input_dict

def get_binned_values(bigwig_file, bin_size=25, chr_sizes_file="data/hg38.chrom.sizes"):
    main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
    chr_sizes = {}

    with open(chr_sizes_file, 'r') as f:
        for line in f:
            chr_name, chr_size = line.strip().split('\t')
            if chr_name in main_chrs:
                chr_sizes[chr_name] = int(chr_size)

    inputs = []
    for chr, size in chr_sizes.items():
        inputs.append({"bw": bigwig_file, "chr": chr, "start": 0, "end": bin_size * (size // bin_size), "resolution": bin_size, "bw_obj": False})

    res = {}
    for i in inputs:
        res[i["chr"]] = get_bin_value_dict(i)["signals"]

    return res

def extract_donor_information(json_data):
    # Check if 'donor' key exists in the JSON data
    # Initialize an empty dictionary to store donor information
    donor_info = json_data.get('donor', {})
    extracted_info = {}
    
    # Extract relevant donor information
    extracted_info['Status'] = donor_info.get('status')
    extracted_info['Accession'] = donor_info.get('accession')
    extracted_info['Aliases'] = donor_info.get('aliases')
    extracted_info['Species'] = donor_info.get('organism', {}).get('scientific_name')
    extracted_info['Life Stage'] = donor_info.get('life_stage')
    extracted_info['Age'] = donor_info.get('age')
    extracted_info['Sex'] = donor_info.get('sex')
    extracted_info['Ethnicity'] = donor_info.get('ethnicity')
    
    return extracted_info

def visualize_encode_data(df):
    # Remove all rows for which num_nonexp_available < 3
    df_filtered = df[df['num_nonexp_available'] >= 3]

    # Sort biosamples based on num_nonexp_available
    df_sorted = df_filtered.sort_values('num_nonexp_available', ascending=False)

    # Prepare the DataFrame for the heatmap (experiments as rows, biosamples as columns)
    heatmap_df = df_sorted.set_index('Accession').drop(['num_available', 'num_nonexp_available'], axis=1).T

    # Convert experiments to numerical values: 1 for available data and NaN for missing
    heatmap_numeric = heatmap_df.notna().astype(int)

    # Calculate the sum of non-NaN entries for each row (experiment) and sort the DataFrame
    heatmap_numeric['non_nan_count'] = heatmap_numeric.sum(axis=1)
    heatmap_numeric_sorted = heatmap_numeric.sort_values('non_nan_count', ascending=False).drop('non_nan_count', axis=1)

    # Create a custom colormap
    cmap = ListedColormap(['white', 'blue'])

    # Plot the heatmap with a larger figure size
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 15))  # Increase figure size

    # Create the heatmap
    sns.heatmap(heatmap_numeric_sorted, cmap=cmap, cbar=False, linewidths=0.0)

    # Remove x-axis labels
    ax.set_xticklabels([])

    # Decrease font-size for y-axis labels
    plt.setp(ax.get_yticklabels(), fontsize=9)
    plt.savefig(f"data/dataset.png", dpi=200)

def visualize_availability(
    sorted_data_hist_uniq_exp, sorted_data_tf_uniq_exp, 
    sorted_data_rest_uniq_exp, encode_imputation_challenge_assays): # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Histone Modifications
        axs[0].bar([item[0] for item in sorted_data_hist_uniq_exp], [item[1] for item in sorted_data_hist_uniq_exp], color='green', alpha=0.8)
        axs[0].set_title('Histone Modifications')
        axs[0].tick_params(axis='x', rotation=90, labelsize=7)
        assays = [item[0] for item in sorted_data_hist_uniq_exp]
        values = [item[1] for item in sorted_data_hist_uniq_exp]
        # for assay, value in sorted_data_hist_uniq_exp:
        #     if assay in encode_imputation_challenge_assays:
        #         axs[0].text(assays.index(assay), value, '*', color='red', fontsize=14, ha='center')

        # TF Binding
        axs[1].bar([item[0] for item in sorted_data_tf_uniq_exp if item[1]>15], [item[1] for item in sorted_data_tf_uniq_exp if item[1]>15], color='orange', alpha=0.8)
        axs[1].set_title('TF Binding')
        axs[1].tick_params(axis='x', rotation=90, labelsize=7)
        assays = [item[0] for item in sorted_data_tf_uniq_exp]
        values = [item[1] for item in sorted_data_tf_uniq_exp]
        # for assay, value in sorted_data_tf_uniq_exp:
        #     if assay in encode_imputation_challenge_assays:
        #         axs[1].text(assays.index(assay), value, '*', color='red', fontsize=14, ha='center')

        # Other Assays
        axs[2].bar([item[0] for item in sorted_data_rest_uniq_exp], [item[1] for item in sorted_data_rest_uniq_exp], color='blue', alpha=0.8)
        axs[2].set_title('Other Assays')
        axs[2].tick_params(axis='x', rotation=90, labelsize=10)
        assays = [item[0] for item in sorted_data_rest_uniq_exp]
        values = [item[1] for item in sorted_data_rest_uniq_exp]
        # for assay, value in sorted_data_rest_uniq_exp:
        #     if assay in encode_imputation_challenge_assays:
        #         axs[2].text(assays.index(assay), value, '*', color='red', fontsize=14, ha='center')

        plt.tight_layout()
        plt.savefig(f"data/availability.png", dpi=200)

def single_download(dl_dict):
    num_attempts = 10

    url, save_dir_name, exp, bios = dl_dict["url"], dl_dict["save_dir_name"], dl_dict["exp"], dl_dict["bios"]

    if os.path.exists(save_dir_name) ==  False:
        print(f"downloading assay: {exp} | biosample: {bios}")
        attempt = 0
        is_done = False
        while is_done == False and attempt < num_attempts:
            if attempt > 0:
                time.sleep(10)

            print(f"attemp number {attempt}")
            is_done = download_save(url, save_dir_name)
            attempt += 1
                
        if "bam" in save_dir_name:
            os.system(f"samtools index {save_dir_name}")
            print(f"processing BAM to Signal | assay: {exp} | biosample: {bios}")

            bam_to_signal = BAM_TO_SIGNAL(
                bam_file=save_dir_name, 
                chr_sizes_file="data/hg38.chrom.sizes")

            bam_to_signal.full_preprocess()
            os.system(f"rm {save_dir_name}")

    else:
        print(f"assay: {exp} | biosample: {bios} already exists!")

def get_encode_chromatin_state_annotation_metadata(
    url="https://www.encodeproject.org/report.tsv?type=Annotation&searchTerm=annotation&annotation_type=chromatin+state&organism.scientific_name=Homo+sapiens&software_used.software.name=chromhmm&assembly=GRCh38",
    metadata_file_path="data/"):
    base_url = "https://www.encodeproject.org"
    # Download the TSV file
    try:
        # Read TSV directly from URL, skipping first row which contains field descriptions
        df = pd.read_csv(url, sep='\t', skiprows=1)
    except Exception as e:
        print(f"Error downloading/parsing TSV file: {e}")
        return None
    
    for i in range(len(df)):
        df.loc[i, "url"] = base_url + df.loc[i, "ID"]

        bios_data = requests.get(df.loc[i, "url"], headers={'accept': 'application/json'})
        bios_data = bios_data.json()

        # Find bigBed files for GRCh38
        bed_files = []
        
        # Check files in the annotation
        if 'files' in bios_data:
            for file_obj in bios_data['files']:
                # Check if it's a file object (has required attributes)
                if isinstance(file_obj, dict):
                    is_valid = (
                        file_obj.get('file_format') == 'bigBed' and
                        file_obj.get('assembly') == 'GRCh38' and
                        file_obj.get('status') == 'released'
                    )
                    
                    if is_valid:
                        file_info = {
                            'accession': file_obj.get('accession'),
                            'download_url': file_obj.get('href'),
                            'cloud_metadata': file_obj.get('cloud_metadata', {}).get('url')
                        }
                        bed_files.append(file_info)
                
                # If it's a file reference, fetch the file details
                elif isinstance(file_obj, str) and file_obj.startswith('/files/ENCFF'):
                    file_url = base_url + file_obj
                    file_response = requests.get(file_url, headers={'accept': 'application/json'})
                    file_data = file_response.json()
                    
                    is_valid = (
                        file_data.get('file_format') == 'bigBed' and
                        file_data.get('assembly') == 'GRCh38' and
                        file_data.get('status') == 'released'
                    )
                    
                    if is_valid:
                        file_info = {
                            'accession': file_data.get('accession'),
                            'download_url': file_data.get('href')                        
                            }
                        bed_files.append(file_info)

        # Store the bigBed files information in the dataframe
        for j, file_info in enumerate(bed_files):
            for key, value in file_info.items():
                if key == "download_url":
                    value = base_url + value
                df.loc[i, f"bed_file_{key}"] = value

    df.to_csv(metadata_file_path + "chromatin_state_annotation_metadata.csv")
    return df

def get_chromatin_state_annotation_data(metadata_file_path="data/", parse_bigBed=True):
    metadata = pd.read_csv(metadata_file_path + "chromatin_state_annotation_metadata.csv")

    if not os.path.exists(f"{metadata_file_path}/chromatin_state_annotations/"):
        os.mkdir(f"{metadata_file_path}/chromatin_state_annotations/")
    
    for index, row in metadata.iterrows():
        try:
            biosample_term_name = row['Biosample term name']
            biosample_term_name = biosample_term_name.replace(" ", "_")
            biosample_term_name = biosample_term_name.replace("/", "_")
            biosample_term_name = biosample_term_name.replace("'", "")
            print(f"Downloading {biosample_term_name}'s chromatin state annotation: {index}/{len(metadata)}")
            bed_file_download_url = row['bed_file_download_url']
            accession = row['Accession']
            save_dir_name = f"{metadata_file_path}/chromatin_state_annotations/{biosample_term_name}"
            if not os.path.exists(save_dir_name):
                os.mkdir(save_dir_name)
                
            if bed_file_download_url.endswith('.bed.gz'):
                download_save(bed_file_download_url, f"{save_dir_name}/temp.bed.gz")
                with gzip.open(f"{save_dir_name}/temp.bed.gz", 'rb') as f_in, open(f"{save_dir_name}/{accession}.bed", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(f"{save_dir_name}/temp.bed.gz")

            elif bed_file_download_url.endswith('.bigBed'):
                if not os.path.exists(f"{save_dir_name}/{accession}.bigBed"):
                    download_save(bed_file_download_url, f"{save_dir_name}/{accession}.bigBed")

                print(f"Parsing {accession}'s chromatin state annotation")
                os.mkdir(f"{save_dir_name}/parsed200_{accession}/")
                if parse_bigBed:
                    # binned_bw = get_binned_bigBed_annotation(
                    #     bigBed_file=f"{save_dir_name}/{accession}.bigBed", 
                    #     resolution=25, 
                    #     chr_sizes_file=f"{metadata_file_path}/hg38.chrom.sizes")

                    binned_bw = get_binned_bigBed_annotation(
                        bigBed_file=f"{save_dir_name}/{accession}.bigBed", 
                        resolution=200, 
                        chr_sizes_file=f"{metadata_file_path}/hg38.chrom.sizes")

                    for chr, data in binned_bw.items():
                        np.savez_compressed(
                                f"{save_dir_name}/parsed200_{accession}/{chr}.npz", 
                                np.array(data))

        except:
            print(f"Error downloading {biosample_term_name}'s chromatin state annotation: {index}/{len(metadata)}")

def get_binned_bigBed_annotation(bigBed_file, resolution=25, chr_sizes_file="data/hg38.chrom.sizes"):
    main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
    chr_sizes = {}
    res = {}

    with open(chr_sizes_file, 'r') as f:
        for line in f:
            chr_name, chr_size = line.strip().split('\t')
            if chr_name in main_chrs:
                chr_sizes[chr_name] = int(chr_size)

    bb = pyBigWig.open(bigBed_file)

    for chr, size in chr_sizes.items():
        start = 0
        end = resolution * (size // resolution)
        
        num_bins = int((end - start + resolution - 1) // resolution)
        labels = [None] * num_bins  # Initialize list of labels
        intervals = bb.entries(chr, start, end)
        if intervals is None:
            res[chr] = labels
            continue

        for interval_start, interval_end, interval_string in intervals:
            label = interval_string.split('\t')[0]
            peak_start_adj = max(interval_start, start)
            peak_end_adj = min(interval_end, end)
            start_bin = int((peak_start_adj - start) // resolution)
            end_bin = int((peak_end_adj - start - 1) // resolution)

            for bin_idx in range(start_bin, end_bin + 1):
                labels[bin_idx] = label
        
        res[chr] = np.array(labels)
    
    bb.close()
    return res

def get_binned_bigBed_peaks(bigBed_file, resolution=25, chr_sizes_file="data/hg38.chrom.sizes"):
    main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
    chr_sizes = {}
    res = {}

    with open(chr_sizes_file, 'r') as f:
        for line in f:
            chr_name, chr_size = line.strip().split('\t')
            if chr_name in main_chrs:
                chr_sizes[chr_name] = int(chr_size)

    bb = pyBigWig.open(bigBed_file)

    for chr, size in chr_sizes.items():
        start = 0
        end = resolution * (size // resolution)
        
        num_bins = int((end - start + resolution - 1) // resolution)
        vector = np.zeros(num_bins, dtype=int)
        intervals = bb.entries(chr, start, end, withString=False)
        if intervals is None:
            bb.close()
            return vector
        for peak_start, peak_end in intervals:
            peak_start_adj = max(peak_start, start)
            peak_end_adj = min(peak_end, end)
            start_bin = int((peak_start_adj - start) // resolution)
            end_bin = int((peak_end_adj - start - 1) // resolution)
            vector[start_bin:end_bin + 1] = 1
        
        res[chr] = vector

    bb.close()
    return res

def load_region_chromatin_states(parsed_path, chrom):
    """
    Load chromatin state labels for a specific genomic region from parsed npz files.
    
    Args:
        parsed_path (str): Path to directory containing parsed npz files (e.g. "data/chromatin_state_annotations/cell_type/parsed_ENCSR123ABC/")
        chrom (str): Chromosome name (e.g. "chr1")
        start (int): Start position (0-based)
        end (int): End position (exclusive)
        resolution (int): Resolution of binned data in base pairs
        
    Returns:
        numpy.ndarray: Array of chromatin state labels for the specified region
    """
    # Load the chromosome's data
    npz_file = os.path.join(parsed_path, f"{chrom}.npz")
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"No data file found for chromosome {chrom}")
        
    # Load the data with allow_pickle=True
    with np.load(npz_file, allow_pickle=True) as data:
        # The npz file contains a single array
        chr_data = data['arr_0']

    return chr_data
        
def download_activity_data(metadata_file_path="data/"):

    def download_github_directory(
        directory_url="https://github.com/ernstlab/ChromActivity/tree/main/data/labels", 
        download_path="/project/compbio-lab/encode_data/activity_data/"):
        # Create a session
        session = requests.Session()

        # GitHub Raw URL Base
        raw_base_url = "https://raw.githubusercontent.com/"

        # Extract repo information
        repo_info = directory_url.split("github.com/")[1].split("/tree/")
        repo_base = repo_info[0]
        branch_path = repo_info[1].split("/")
        branch_name = branch_path[0]
        directory_path = "/".join(branch_path[1:])

        # Generate API URL
        api_url = f"https://api.github.com/repos/{repo_base}/contents/{directory_path}?ref={branch_name}"

        # Request directory contents
        response = session.get(api_url)
        if response.status_code != 200:
            print(f"Failed to fetch directory contents: {response.status_code}")
            return

        contents = response.json()

        # Ensure the download path exists
        os.makedirs(download_path, exist_ok=True)

        # Download files
        for item in contents:
            if item["type"] == "file":
                file_url = item["download_url"]
                file_name = os.path.join(download_path, item["name"])
                print(f"Downloading {item['name']}...")

                # Download the file
                file_response = session.get(file_url)
                if file_response.status_code == 200:
                    with open(file_name, "wb") as file:
                        file.write(file_response.content)
                else:
                    print(f"Failed to download {item['name']}: {file_response.status_code}")

            elif item["type"] == "dir":
                # Recursive call for subdirectories
                sub_dir_path = os.path.join(download_path, item["name"])
                download_github_directory(item["html_url"], sub_dir_path)

    download_github_directory()
    
    pass

################################################################################

class GET_DATA(object):
    def __init__(self):
        self.encode_imputation_challenge_assays = ["DNase-seq", "H3K4me3", "H3K36me3", "H3K27ac", "H3K9me3",
                "H3K27me3", "H3K4me1", "H3K9ac", "H3K4me2", "H2AFZ", "H3K79me2", "ATAC-seq",
                "H3K18ac", "H4K20me1", "H3K4ac", "H4K8ac", "H3K79me1", "H3K14ac", "H2BK120ac", 
                "H2BK12ac", "H2BK5ac",  "H4K91ac", "H2BK15ac", "H3K23ac",  "H4K5ac",
                "H3K5bac", "H3K23me2", "H2BK20ac", "H3K9me1", "H3F3A", "H4K12ac",  "H3T11ph", "HAk9ac", "H3K9me2"]

        self.select_assays = ["DNase-seq", "H3K4me3", "H3K36me3", "H3K27ac", "H3K9me3", "H3K27me3", "H3K4me1", "ATAC-seq", "CTCF"]

        self.expression_data = ["RNA-seq", "CAGE"]
        
        self.target_file_format = {
            "ChIP-seq": "bam", 
            "CAGE": "bam",
            "ChIA-PET": "bam",
            "ATAC-seq": "bam", 
            "RNA-seq": "bam",
            "DNase-seq": "bam"}

        self.headers = {'accept': 'application/json'}
        # report_url= https://www.encodeproject.org/report/?type=Experiment&control_type!=*&perturbed=false&assay_title=TF+ChIP-seq&assay_title=Histone+ChIP-seq&assay_title=DNase-seq&assay_title=ATAC-seq&assay_title=ChIA-PET&assay_title=CAGE&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=total+RNA-seq&status=released
        self.exp_tsv_url = """https://www.encodeproject.org/report.tsv?type=Experiment&control_type!=*&perturbed=false&assay_title=TF+ChIP-seq&assay_title=Histone+ChIP-seq&assay_title=DNase-seq&assay_title=ATAC-seq&assay_title=ChIA-PET&assay_title=CAGE&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=total+RNA-seq&status=released"""
        self.biosample_tsv_url = """https://www.encodeproject.org/report.tsv?type=Biosample&organism.scientific_name=Homo+sapiens"""

        self.experiment_url = """https://www.encodeproject.org/experiments/"""
        self.biosample_url = """https://www.encodeproject.org/biosamples/"""

    def search_ENCODE(self, metadata_file_path="data/"):
        """
        DF1:
            rows: biosamples
            columns: experiments (from self.experiments)
        
        DF2:
            rows: biosamples
            columns: metadata fields (including replicate biosample if any)
        """

        exp_search_report = pd.read_csv(self.exp_tsv_url, sep='\t', skiprows=1)
        """
        exp search report columns:
        ['ID', 'Accession', 'Assay name', 'Assay title', 'Biosample classification', 'Target', 'Target of assay',
       'Target gene symbol', 'Biosample summary', 'Biosample term name', 'Dbxrefs', 'Description', 'Lab', 'Project', 'Status', 'Files',
       'Related series', 'Biosample accession', 'Biological replicate', 'Technical replicate', 'Linked antibody', 'Organism', 'Life stage',
       'Biosample age', 'Biosample treatment', 'Biosample treatment ontology ID', 'Biosample treatment amount',
       'Biosample treatment amount units', 'Biosample treatment duration', 'Biosample treatment duration units', 'Synchronization',
       'Post-synchronization time', 'Post-synchronization time units', 'Biosample modification site target organism',
       'Biosample modification site introduced gene organism', 'Replicates', 'Mixed biosamples', 'Cellular component',
       'Library construction platform', 'Library construction method']
        """

        bios_search_report = pd.read_csv(self.biosample_tsv_url, sep='\t', skiprows=1)
        """
        biosample search report columns:
        ['ID', 'Accession', 'Description', 'Age', 'Age units', 'Biosample age', 'Biosample term name', 'Biosample classification', 'Synchronization',
       'Post-synchronization time', 'Post-synchronization time units', 'Organism', 'Child biosamples', 'Modification site',
       'Modification site target organism', 'Modification site introduced gene organism', 'Modification purpose',
       'Modification method', 'Modification category', 'Source', 'Lab', 'Life stage', 'Status', 'Biosample treatment',
       'Biosample treatment type details', 'Biosample treatment purpose', 'Culture harvest date', 'Date obtained', 'Summary']
        """
        bios_search_report.set_index('Accession', inplace=True)
        bios_search_report = bios_search_report.to_dict('index')

        self.DF1 = {}
        self.DF2 = {}

        hist_uniq_exp = {}
        tf_uniq_exp = {}
        rest_uniq_exp = {}

        # for each experiment, look up the biosample and connect the experiment and biosample data
        for i in range(len(exp_search_report)):
            if i%1000 == 0:
                print(f"{i}/{len(exp_search_report)}")

            exp = exp_search_report["Accession"][i]

            if exp_search_report["Assay name"][i] == "DNase-seq":
                assay = "DNase-seq"
            elif exp_search_report["Assay name"][i] == "RNA-seq":
                assay = "RNA-seq"
            elif exp_search_report["Assay name"][i] == "ATAC-seq":
                assay = "ATAC-seq"
            elif exp_search_report["Assay name"][i] == "CAGE":
                assay = "CAGE"
            elif exp_search_report["Assay name"][i] == "ChIA-PET":
                assay = "ChIA-PET"
            elif exp_search_report["Assay name"][i] == "ChIP-seq":
                assay = exp_search_report["Target of assay"][i]

            
            ########################################################################

            if exp_search_report["Assay title"][i] == "Histone ChIP-seq":
                if assay not in hist_uniq_exp.keys():
                    hist_uniq_exp[assay] = 1
                else:
                    hist_uniq_exp[assay] += 1

            elif exp_search_report["Assay title"][i] == "TF ChIP-seq":
                if assay not in tf_uniq_exp.keys():
                    tf_uniq_exp[assay] = 1
                else:
                    tf_uniq_exp[assay] += 1
            else:
                if assay not in rest_uniq_exp.keys():
                    rest_uniq_exp[assay] = 1
                else:
                    rest_uniq_exp[assay] += 1

        ########################################################################

        sorted_data_hist_uniq_exp = sorted(hist_uniq_exp.items(), key=lambda x: x[1], reverse=True)
        sorted_data_tf_uniq_exp = sorted(tf_uniq_exp.items(), key=lambda x: x[1], reverse=True)
        sorted_data_rest_uniq_exp = sorted(rest_uniq_exp.items(), key=lambda x: x[1], reverse=True)

        ########################################################################

        visualize_availability(
            sorted_data_hist_uniq_exp, sorted_data_tf_uniq_exp, 
            sorted_data_rest_uniq_exp, self.encode_imputation_challenge_assays)
        
        ########################################################################
        
        self.sorted_data_hist_uniq_exp = dict(sorted_data_hist_uniq_exp)
        self.sorted_data_rest_uniq_exp = dict(sorted_data_rest_uniq_exp)
        self.sorted_data_tf_uniq_exp = dict(sorted_data_tf_uniq_exp)

        # for each experiment, look up the biosample and connect the experiment and biosample data
        for i in range(len(exp_search_report)):
            if i%1000 == 0:
                print(f"{i}/{len(exp_search_report)}")

            exp = exp_search_report["Accession"][i]

            if exp_search_report["Assay name"][i] == "DNase-seq":
                assay = "DNase-seq"
            elif exp_search_report["Assay name"][i] == "RNA-seq":
                assay = "RNA-seq"
            elif exp_search_report["Assay name"][i] == "ATAC-seq":
                assay = "ATAC-seq"
            elif exp_search_report["Assay name"][i] == "CAGE":
                assay = "CAGE"
            elif exp_search_report["Assay name"][i] == "ChIA-PET":
                assay = "ChIA-PET"
            elif exp_search_report["Assay name"][i] == "ChIP-seq":
                assay = exp_search_report["Target of assay"][i]

            ########################################################################
            """
            what to assays to include in search:
                - all histone mods
                - TF avail > 15
                - DNase, ATAC, CAGE, RNA-seq, ChIA-PET
            """

            statement1 = bool(exp_search_report["Assay title"][i] == "Histone ChIP-seq")
            statement2 = bool(exp_search_report["Assay title"][i] == "TF ChIP-seq") and bool(self.sorted_data_tf_uniq_exp[assay] > 15)
            statement3 = bool(assay in ["DNase-seq", "RNA-seq", "ATAC-seq", "CAGE", "ChIA-PET"])

            if statement1 or statement2 or statement3:
                biosample_accessions = exp_search_report["Biosample accession"][i].split(",")

                for biosample_accession in biosample_accessions:
                    if biosample_accession not in bios_search_report.keys():
                        continue
                    
                    if biosample_accession in self.DF1.keys():
                        self.DF1[biosample_accession][assay] = exp

                    else:
                        self.DF1[biosample_accession] = {}
                        self.DF1[biosample_accession][assay] = exp

                    if biosample_accession not in self.DF2.keys():
                        try:
                            self.DF2[biosample_accession] = bios_search_report[biosample_accession]

                            if len(biosample_accessions) == 1:
                                self.DF2[biosample_accession]["isogenic_replicates"] = None
                            else:
                                self.DF2[biosample_accession]["isogenic_replicates"] = ",".join(
                                    [x for x in biosample_accessions if x != biosample_accession])

                        except:
                            pass

        self.DF1 = pd.DataFrame.from_dict(self.DF1, orient='index').sort_index(axis=1)
        self.DF2 = pd.DataFrame.from_dict(self.DF2, orient='index').sort_index(axis=1)
        """
        save DF1 and DF2 from search_ENCODE
        """
        self.DF1.to_csv(metadata_file_path + "DF1.csv")
        self.DF2.to_csv(metadata_file_path + "DF2.csv")

    def filter_biosamples(self, metadata_file_path="data/"):

        """
        read DF1 and DF2 metadata files and run download_search_results on them
        """
        self.DF1 = pd.read_csv(metadata_file_path + "DF1.csv")
        self.DF2 = pd.read_csv(metadata_file_path + "DF2.csv")

        self.DF1.rename(columns={'Unnamed: 0': 'Accession'}, inplace=True)
        self.DF2.rename(columns={'Unnamed: 0': 'Accession'}, inplace=True)
        
        self.DF1['num_available'] = self.DF1.count(axis=1) - 1
        self.DF1['num_nonexp_available'] = self.DF1.drop(
            columns=['Accession', 'RNA-seq', 'CAGE', "num_available"]).count(axis=1)
            
        self.DF1 = self.DF1.sort_values(by='Accession').reset_index(drop=True)
        self.DF2 = self.DF2.sort_values(by='Accession').reset_index(drop=True)

        ########################################################################

        """
        what biosamples to download
            - any histone mod available
            - DNase or ATAC available
            - > 3 TF available
        """

        statement1 = self.DF1[list(self.sorted_data_hist_uniq_exp.keys())].count(axis=1) > 0
        statement2 = self.DF1[["DNase-seq", "ATAC-seq"]].count(axis=1) > 0
        statement3 = self.DF1[[
            tf for tf in self.sorted_data_tf_uniq_exp.keys() if self.sorted_data_tf_uniq_exp[tf] > 15]].count(axis=1) > 3

        combined_statement = (statement1 | statement2 | statement3)

        self.DF1 = self.DF1[combined_statement].reset_index(drop=True)
        self.DF2 = self.DF2[combined_statement].reset_index(drop=True)
    
        ########################################################################

        visualize_encode_data(self.DF1)

        ########################################################################
                
        self.DF1 = self.DF1.drop(["num_available", "num_nonexp_available"], axis=1)
    
        """
        save DF1 and DF2 from search_ENCODE
        """
        self.DF1.to_csv(metadata_file_path + "DF1.csv")
        self.DF2.to_csv(metadata_file_path + "DF2.csv")

    def load_metadata(self, metadata_file_path="data/"):
        self.DF1 = pd.read_csv(metadata_file_path + "DF1.csv").drop(["Unnamed: 0"], axis=1)
        self.DF2 = pd.read_csv(metadata_file_path + "DF2.csv").drop(["Unnamed: 0"], axis=1)

    def get_experiment(self, dl_dict, process_bam=True):
        num_attempts = 10

        def download_save(url, save_dir_name):
            try:
                download_response = requests.get(url, allow_redirects=True)
                open(save_dir_name, 'wb').write(download_response.content)
                return True

            except:
                return False

        url, save_dir_name, exp, bios = dl_dict["url"], dl_dict["save_dir_name"], dl_dict["exp"], dl_dict["bios"]

        if os.path.exists(save_dir_name) ==  False and os.path.exists(
            f"{'/'.join(save_dir_name.split('/')[:-1])}/signal_DSF1_res25/") == False:

            print(f"downloading assay: {exp} | biosample: {bios}")
            attempt = 0
            is_done = False
            while is_done == False and attempt < num_attempts:
                if attempt > 0:
                    time.sleep(20)

                print(f"    attemp number {attempt}")
                is_done = download_save(url, save_dir_name)
                attempt += 1
            
            if is_done == False:
                open(save_dir_name.replace(".bam",".failed"), 'w').write("failed to download")
                print("failed to download", save_dir_name)
                return

            else:  
                if "bam" in save_dir_name:
                    try:
                        os.system(f"samtools index {save_dir_name}")

                        if process_bam:
                            print(f"processing BAM to Signal | assay: {exp} | biosample: {bios}")

                            bam_to_signal = BAM_TO_SIGNAL(
                                bam_file=save_dir_name, 
                                chr_sizes_file="data/hg38.chrom.sizes")

                            bam_to_signal.full_preprocess()
                            
                            os.system(f"rm {save_dir_name}")

                    except:
                        print("failed to process", save_dir_name)

        else:
            print(f"assay: {exp} | biosample: {bios} already exists!")

    def get_biosample(self, bios, df1_ind, metadata_file_path, assembly):
        # i = df1_ind
        to_download_bios = []

        i = self.DF1[self.DF1['Accession'] == bios].index[0]

        bios_data =  requests.get(f"""https://www.encodeproject.org/biosamples/{bios}""", headers=self.headers)
        bios_data = bios_data.json()
        donor_info = extract_donor_information(bios_data)

        if os.path.exists(metadata_file_path + "/" + bios + "/") == False:
            os.mkdir(metadata_file_path + "/" + bios + "/")
        
        with open(metadata_file_path + "/" + bios + '/donor.json', 'w') as file:
            json.dump(donor_info, file, indent=4)

        for exp in self.DF1.columns:
            if exp not in ["Accession", "num_nonexp_available", "num_available"]:
                try:
                    if pd.notnull(self.DF1[exp][i]):
                        # print(bios, exp, self.DF1[exp][i])
                        experiment_accession = self.DF1[exp][i]
                        if os.path.exists(metadata_file_path + "/" + bios + "/" + exp) == False:
                            os.mkdir(metadata_file_path + "/" + bios + "/" + exp)

                    else:
                        continue
                    
                    exp_url = self.experiment_url + experiment_accession
                    
                    exp_respond = requests.get(exp_url, headers=self.headers)
                    exp_results = exp_respond.json()
                    
                    e_fileslist = list(exp_results['original_files'])
                    e_files_navigation = []

                    for ef in e_fileslist:
                        efile_respond = requests.get("https://www.encodeproject.org{}".format(ef), headers=self.headers)
                        efile_results = efile_respond.json()

                        if efile_results['file_format'] == "bam" or efile_results['file_format'] == "tsv":
                            # try: #ignore files without sufficient info or metadata

                            if efile_results['status'] == "released": 
                                #ignore old and depricated versions

                                if "origin_batches" in efile_results.keys():
                                    if ',' not in str(efile_results['origin_batches']):
                                        e_file_biosample = str(efile_results['origin_batches'])
                                        e_file_biosample = e_file_biosample.replace('/', '')
                                        e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                                    else:
                                        repnumber = int(efile_results['biological_replicates'][0]) - 1
                                        e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                                else:
                                    repnumber = int(efile_results['biological_replicates'][0]) - 1
                                    e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]

                                # ignore files that contain both replicates 
                                if e_file_biosample == bios:
                                    parsed = [exp, efile_results['accession'], e_file_biosample,
                                        efile_results['file_format'], efile_results['output_type'], 
                                        efile_results['dataset'], efile_results['biological_replicates'], 
                                        efile_results['file_size'], efile_results['assembly'], 
                                        "https://www.encodeproject.org{}".format(efile_results['href']), 
                                        efile_results['date_created'], efile_results['status']]

                                    if "read_length" in efile_results:
                                        read_length = efile_results["read_length"]
                                        run_type = efile_results["run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    elif "mapped_read_length" in efile_results:
                                        read_length = efile_results["mapped_read_length"]
                                        run_type = efile_results["mapped_run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    else:
                                        parsed.append(None)
                                        parsed.append(None)

                                    e_files_navigation.append(parsed)
                        # except:
                        #     pass

                    if len(e_files_navigation) == 0:
                        for ef in e_fileslist:
                            efile_respond = requests.get("https://www.encodeproject.org{}".format(ef), headers=self.headers)
                            efile_results = efile_respond.json()
                            if efile_results['file_format'] == "bam" or efile_results['file_format'] == "tsv":
                                if efile_results['status'] == "released": 
                                    if "origin_batches" in efile_results.keys():
                                        if ',' not in str(efile_results['origin_batches']):
                                            e_file_biosample = str(efile_results['origin_batches'])
                                            e_file_biosample = e_file_biosample.replace('/', '')
                                            e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                                        else:
                                            repnumber = int(efile_results['biological_replicates'][0]) - 1
                                            e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                                    else:
                                        repnumber = int(efile_results['biological_replicates'][0]) - 1
                                        e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                                    
                                    parsed = [exp, efile_results['accession'], e_file_biosample,
                                        efile_results['file_format'], efile_results['output_type'], 
                                        efile_results['dataset'], efile_results['biological_replicates'], 
                                        efile_results['file_size'], efile_results['assembly'], 
                                        "https://www.encodeproject.org{}".format(efile_results['href']), 
                                        efile_results['date_created'], efile_results['status']]

                                    if "read_length" in efile_results:
                                        read_length = efile_results["read_length"]
                                        run_type = efile_results["run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    elif "mapped_read_length" in efile_results:
                                        read_length = efile_results["mapped_read_length"]
                                        run_type = efile_results["mapped_run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    else:
                                        parsed.append(None)
                                        parsed.append(None)

                                    e_files_navigation.append(parsed)

                    e_files_navigation = pd.DataFrame(e_files_navigation, columns=[
                        'assay', 'accession', 'biosample', 'file_format', 
                        'output_type', 'experiment', 'bio_replicate_number', 
                        'file_size', 'assembly', 'download_url', 'date_created', 
                        'status', 'read_length', "run_type"])

                    # select one file from e_files_navigation to download
                    e_files_navigation.to_csv(metadata_file_path + "/" + bios + "/" + exp + "/all_files.csv")
                    
                    # just keep target assembly
                    e_files_navigation = e_files_navigation[e_files_navigation['assembly'] == assembly]

                    # Convert 'date_created' to datetime
                    e_files_navigation['date_created'] = pd.to_datetime(e_files_navigation['date_created'])
                    
                    if exp == "RNA-seq":
                        # Filter rows where 'output_type' is 'gene quantification'
                        filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'gene quantifications']
                    else:
                        # Filter rows where 'output_type' is 'alignments'
                        if "alignments" in e_files_navigation['output_type'].unique():
                            filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'alignments']

                        elif "redacted alignments" in e_files_navigation['output_type'].unique():
                            filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'redacted alignments']

                    # Find the row with the newest 'date_created'
                    newest_row = filtered_df[filtered_df['date_created'] == filtered_df['date_created'].max()]

                    # Print the newest row
                    # print(newest_row)
                    # print(newest_row.to_json(indent=4))

                    with open(metadata_file_path + "/" + bios + "/" + exp + "/file_metadata.json", "w") as f:
                        f.write(newest_row.to_json(indent=4))
                    
                    if exp == "RNA-seq":
                        save_dir_name = metadata_file_path + "/" + bios + "/" + exp + "/" + newest_row["accession"].values[0] + ".tsv"
                    else:
                        save_dir_name = metadata_file_path + "/" + bios + "/" + exp + "/" + newest_row["accession"].values[0] + ".bam"
                    
                    url = newest_row["download_url"].values[0]
                    file_size = newest_row["file_size"].values[0]

                    to_download_bios.append({"url":url, "save_dir_name":save_dir_name, "exp":exp, "bios":bios})

                except Exception as e:
                    with open(metadata_file_path + "/" + bios  + f"/failed_{exp}", "w") as f:
                        f.write(f"failed to download {bios}_{exp}\n {e}")

        # NUM_BIOS_DOWNLOADED += 1
        # if NUM_BIOS_DOWNLOADED % 30 == 0:
        #     print(NUM_BIOS_DOWNLOADED)
        return to_download_bios

    def get_biosample_wrapper(self, *args):
        # Wrapper method that can be called in a multiprocessing context
        return self.get_biosample(*args)

    def get_all(self, metadata_file_path="data/", mode="parallel", n_p=25, assembly="GRCh38"):
        to_download = []
        if os.path.exists(metadata_file_path + "DF3.csv"):
            # parse to_download from DF3
            df = pd.read_csv(metadata_file_path + "DF3.csv").drop("Unnamed: 0", axis=1)
            for i in range(len(df)):
                to_download.append(
                    {
                        "url":df["url"][i], "save_dir_name":df["save_dir_name"][i], 
                        "exp":df["exp"][i], "bios":df["bios"][i]
                        }
                )
                if os.path.exists(metadata_file_path + "/" + df["bios"][i]) == False:
                    os.mkdir(metadata_file_path + "/" + df["bios"][i])
                
                if os.path.exists(metadata_file_path + "/" + df["bios"][i] + "/" + df["exp"][i]) == False:
                    os.mkdir(metadata_file_path + "/" + df["bios"][i] + "/" + df["exp"][i])

        else:

            if mode == "parallel":
                def pool_get_biosample(args):
                    return self.get_biosample(*args)

                args_list = [(self.DF1["Accession"][i], i, metadata_file_path, assembly) for i in range(len(self.DF1))]
                with mp.Pool(n_p) as pool:
                    # Map the get_biosample function to the arguments
                    results = pool.starmap(self.get_biosample_wrapper, args_list)

                for bios_dl in results:
                    to_download.extend(bios_dl)

            else:
                for i in range(len(self.DF1)):
                    bios = self.DF1["Accession"][i]
                    bios_dl = self.get_biosample(bios, i, metadata_file_path, assembly)
                    for dl in bios_dl:
                        to_download.append(dl)
                
            df3 = pd.DataFrame(to_download, columns=["url", "save_dir_name", "exp", "bios"])
            df3.to_csv(metadata_file_path + "/DF3.csv")

        if mode == "parallel":
            with mp.Pool(n_p) as pool:
                pool.map(self.get_experiment, to_download)

        else:
            for d in to_download:
                self.get_experiment(d)

class BAM_TO_SIGNAL(object):
    def __init__(self, bam_file, chr_sizes_file, resolution=25):
        self.bam_file = bam_file
        self.chr_sizes_file = chr_sizes_file
        self.resolution = resolution
        self.read_chr_sizes()
        self.load_bam()

    def read_chr_sizes(self):
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        self.chr_sizes = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def load_bam(self):
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')

    def initialize_empty_bins(self):
        return {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage_pysam(self, downsampling_factor=1.0):
        bins = self.initialize_empty_bins()

        total_mapped_reads = 0 
        bins_with_reads = 0  

        for chr in self.chr_sizes:
            for read in self.bam.fetch(chr):
                if random.random() < 1.0 / downsampling_factor:
                    if read.is_unmapped:
                        continue
                    total_mapped_reads += 1  

                    start_bin = read.reference_start // self.resolution
                    end_bin = read.reference_end // self.resolution
                    for i in range(start_bin, end_bin + 1):
                        if bins[chr][i] == 0:  
                            bins_with_reads += 1  
                        bins[chr][i] += 1
        
        # Calculate coverage as the percentage of bins with at least one read
        total_bins = sum(len(b) for b in bins.values())  
        coverage = (bins_with_reads / total_bins) if total_bins > 0 else 0

        return bins, total_mapped_reads, coverage

    def save_signal_metadata(self, depth, coverage, downsampling_factor):
        if os.path.exists(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/") == False:
            os.mkdir(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/")
        
        filename = f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/metadata.json"
        mdict = {
            "coverage":coverage,
            "depth":depth,
            "dsf":downsampling_factor}

        with open(filename, 'w') as file:
            json.dump(mdict, file, indent=4)
    
    def save_signal(self, bins, downsampling_factor=1):
        if os.path.exists(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/") == False:
            os.mkdir(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/")

        for chr, data in bins.items():
            np.savez_compressed(
                f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/{chr}.npz", 
                np.array(data))
            # data_tensor = torch.tensor(data)
            # torch.save(data_tensor, f"{'/'.join(self.bam_file.split('/')[:-1])}/tensors_DSF{downsampling_factor}_res{self.resolution}/{chr}.pt")
    
    def full_preprocess(self, dsf_list=[1,2,4,8]):
        t0 = datetime.datetime.now()

        for dsf in dsf_list:
            data, depth, coverage = self.calculate_coverage_pysam(downsampling_factor=dsf)
            self.save_signal(data, downsampling_factor=dsf)
            self.save_signal_metadata(depth, coverage, downsampling_factor=dsf)

        t1 = datetime.datetime.now()
        print(f"took {t1-t0} to get signals for {self.bam_file} at resolution: {self.resolution}bp")

class ENCODE_IMPUTATION_DATASET(object):
    def __init__(self, path):
        """
        each pkl.gz file is for one biosample and is a dictionary:
        d = {
            "assay1":[list of several pairs of ([chr, start, end], [signal_array]) ],
            "assay2":[list of several pairs of ([chr, start, end], [signal_array]) ],
            "assay3":[list of several pairs of ([chr, start, end], [signal_array]) ],
        }

        let's say we have A assays, and M sample ( len(d["assay1"])=M ).
        signal_arrays are of the same length and for all assays, signal_array[i] corresponds to the same genomic position. 
        if we have M pairs of ([chr, start, end], [signal_array]) for each assay, we will have M samples of size: (len(signal_array), number_of_assays)
        """

        self.path = path
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.all_ct = ['C{:02d}'.format(i) for i in range(1, 52)]

        availability = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in availability.keys():
                    availability[f[:3]] = 0
                availability[f[:3]] += 1
                

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".pkl.gz" in f: 
                self.biosamples[f[:3]] = f"{self.path}/{f}"
        
        # Sort the keys in availability in descending order
        sorted_keys = sorted(availability, key=availability.get, reverse=True)

        # Create a new dictionary with sorted keys
        self.biosamples = {key: self.biosamples[key] for key in sorted_keys if key in self.biosamples}

        self.preprocessed_datasets = []
        for f in os.listdir(self.path):
            if ".pt" in f and "mixed_dataset" in f: 
                self.preprocessed_datasets.append(f"{self.path}/{f}")
        
    def get_biosample_pkl(self, pkl_path):
        with gzip.open(pkl_path, 'rb') as f:
            loaded_file = pickle.load(f)

        bios_assays = loaded_file.keys()
        assay_availability = {ass: (True if ass in bios_assays else False) for ass in self.all_assays}

        M = len(loaded_file[list(loaded_file.keys())[0]])
        L = len(loaded_file[list(loaded_file.keys())[0]][0][1])
        D = len(self.all_assays)

        missing_f_i = []
        # Initialize an empty list to hold all samples
        all_samples = []
        
        # Iterate over all assays
        for i, assay in enumerate(self.all_assays):
            if assay_availability[assay]:
                # If assay is available, append its signal arrays to all_samples
                assay_samples = []
                for j in range(len(loaded_file[assay])):
                    assay_samples.append(loaded_file[assay][j][1])

            else:
                missing_f_i.append(i)
                # If assay is not available, append -1 of appropriate shape
                assay_samples = []
                for j in range(M):
                    assay_samples.append([-1 for _ in range(L)])
            
            all_samples.append(assay_samples)

        # Convert all_samples to a numpy array and transpose to get shape (M, L, D)
        all_samples_tensor = np.array(all_samples, dtype=np.float32).transpose(1, 2, 0)

        # Convert numpy array to PyTorch tensor
        all_samples_tensor = torch.from_numpy(all_samples_tensor)
        all_samples_tensor = all_samples_tensor.float() 

        # Create a mask tensor
        mask = (all_samples_tensor == -1)

        return all_samples_tensor, mask, missing_f_i
    
    def get_dataset_pt(self, pt_path):
        ds = torch.load(pt_path)
        mask = (ds == -1)
        mask_2 = (ds.sum(dim=1) < 0) # missing assay pattern per sample

        indices = [torch.nonzero(mask_2[i, :], as_tuple=True)[0].tolist() for i in range(mask_2.shape[0])]

        unique_indices = [list(x) for x in set(tuple(x) for x in indices)]
        pattern_dict = {tuple(pattern): [] for pattern in unique_indices}

        for i, pattern in enumerate(indices):
            pattern_dict[tuple(pattern)].append(i)

        return ds, mask, pattern_dict
       
class ExtendedEncodeDataHandler:
    """
        set alias for bios(E) and experiments(M) -> save in json
        navigate all bios-exps -> save in json
        [OPTIONAL: merge bios by donor ID?]
            - write in updated navigation json + alias json
        [train-val-test-split] -- EIC + whole_bios -> save in json
        [OPTIONAL: npz-to-npy?]
            - convert all npz files to npy format
        generate_genome_loci(context_length, ccre=False, frac_genome=0.1)
            if not ccre:
                generate_random_loci(frac_genome, context_length)
            else:
                generate_ccre_loci(context_length)
        load_exp(bios_name, exp_name, locus)
        load_bios(bios_name, locus) 
            - for all available exp for each bios, load all
        
        make tensor(loci)
    """
    def __init__(self, base_path, resolution=25):
        self.base_path = base_path
        self.headers = {'accept': 'application/json'}
        self.chr_sizes_file = os.path.join(self.base_path, "hg38.chrom.sizes")
        self.alias_path = os.path.join(self.base_path, "aliases.json")
        self.navigation_path = os.path.join(self.base_path, "navigation.json")
        self.merged_navigation_path = os.path.join(self.base_path, "merged_navigation.json")
        self.split_path = os.path.join(self.base_path, "train_va_test_split.json")
        self.merged_split_path = os.path.join(self.base_path, "merged_train_va_test_split.json")

        self.blacklist_file = os.path.join(self.base_path, "hg38_blacklist_v2.bed") 
        self.blacklist = self.load_blacklist(self.blacklist_file)

        self.resolution = resolution
        self.df1_path = os.path.join(self.base_path, "DF1.csv")
        self.df1 = pd.read_csv(self.df1_path)

        self.df2_path = os.path.join(self.base_path, "DF2.csv")
        self.df2 = pd.read_csv(self.df2_path).drop("Unnamed: 0", axis=1)

        self.df3_path = os.path.join(self.base_path, "DF3.csv")
        self.df3 = pd.read_csv(self.df3_path).drop("Unnamed: 0", axis=1)

        self.eicdf_path = os.path.join(self.base_path, "EIC_experiments.csv")
        self.eic_df = pd.read_csv(self.eicdf_path)
        self.expstats = pd.read_csv(os.path.join(self.base_path, "ExpStats.csv")).drop("Unnamed: 0", axis=1)


        self.RawExpMetaData = pd.read_csv(os.path.join(self.base_path, "RawExpMetaData.csv"))
        self.RawExpMetaData.index = list(self.RawExpMetaData["Unnamed: 0"])
        self.RawExpMetaData = self.RawExpMetaData.drop("Unnamed: 0", axis=1)

        for c in self.RawExpMetaData.columns:
            for md in self.RawExpMetaData.index:
                print(type(self.RawExpMetaData.loc[md, c]))
        exit()

        # mapping from metadata row → conversion function
        cast_fn = {
            "depth":        lambda s: list(map(int,   s.strip("[]").split(","))),
            "coverage":     lambda s: list(map(float, s.strip("[]").split(","))),
            "read_length":  lambda s: list(map(int,   s.strip("[]").split(","))),
            "run_type":     lambda s: list(map(str,   s.strip("[]").split(","))),
        }

        for row, fn in cast_fn.items():
            if row in self.RawExpMetaData.index:      # skip rows that don’t exist
                self.RawExpMetaData.loc[row] = (
                    self.RawExpMetaData.loc[row]
                    .apply(fn)                        # apply to every column
                )

        # optional: inspect the converted rows
        print(self.RawExpMetaData.loc[list(cast_fn)])


        # print(self.expstats)
        # print(self.RawExpMetaData)
        exit()

    def load_blacklist(self, blacklist_file):
        """Load blacklist regions from a BED file into IntervalTrees."""
        blacklist = {}
        with open(blacklist_file, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                if chrom not in blacklist:
                    blacklist[chrom] = IntervalTree()
                blacklist[chrom].addi(start, end)
        return blacklist
    
    def is_region_allowed(self, chrom, start, end):
        """Check if a region overlaps with blacklist regions using IntervalTree."""
        if chrom not in self.blacklist:
            return True
        tree = self.blacklist[chrom]
        overlapping = tree.overlap(start, end)
        return len(overlapping) == 0

    def report(self):
        """
        Generates a formatted text report of the dataset.
        """
        print("Extended Encode Data Report")
        print("----------------------------")
        print(f"Total number of complete biosamples: {sum(self.DS_checkup())}")
        
        # Count biosamples with more than n assays available
        assay_count = self.df1.notna().sum(axis=1)
        max_assays = assay_count.max()
        print("\nNumber of biosamples with more than N assays available:")
        for n in range(max_assays, 0, -1):
            count = (assay_count >= n).sum()
            print(f"  - More than {n} assays: {count}")
        
        # Count of biosamples where each assay is available
        print("\nAssays availability in biosamples:")
        for assay in self.df1.columns[1:]:  # Skip 'Accession'
            available_count = self.df1[assay].notna().sum()
            print(f"  - {assay}: {available_count} biosamples")
        
        # Training and testing biosamples
        print("\nTraining and testing biosamples:")
        print(f"  - Number of training biosamples: {len([b for b, s in self.split_dict.items() if s == 'train'])}")
        print(f"  - Number of testing biosamples: {len([b for b, s in self.split_dict.items() if s == 'test'])}")
        
        # Count of isogenic replicates
        isogenic_train_test = self.df2[self.df2['isogenic_replicates'].notnull()]
        isogenic_count = 0
        for index, row in isogenic_train_test.iterrows():
            if row['Accession'] in self.split_dict and self.split_dict[row['Accession']] == 'train' \
            and row['isogenic_replicates'] in self.split_dict and self.split_dict[row['isogenic_replicates']] == 'test':
                isogenic_count += 1
        print(f"  - Number of isogenic replicates, one in train and one in test: {isogenic_count}\n")

    def coords(self, mode="train"):
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        if mode == "train":
            main_chrs.remove("chr21") # reserved for validation
        self.chr_sizes = {}

        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)    
        
        # self.chr_sizes = {"chr1":self.chr_sizes["chr1"]}

        self.genomesize = sum(list(self.chr_sizes.values()))

    def is_exp_complete(self, bios_name, exp, check_pval=True):
        required_dsfs = ['DSF1', 'DSF2', 'DSF4', 'DSF8']
        
        bios_path = os.path.join(self.base_path, bios_name)
        exp_path = os.path.join(bios_path, exp)
        exp_listdir = os.listdir(exp_path)

        exp_full = True # assume the experiment is complete until proven otherwise

        if exp == "RNA-seq":
            if  "file_metadata.json" in exp_listdir:
                tsv_files = [f for f in exp_listdir if f.endswith('.tsv')]
                if len(tsv_files) == 0:
                    return False
        else:
            for dsf in required_dsfs:
                if exp_full == True:
                    if  "file_metadata.json" in exp_listdir:
                        if f'signal_{dsf}_res25' in exp_listdir:
                            md1_path = os.path.join(exp_path, f'signal_{dsf}_res25', "metadata.json")
                            exp_full = os.path.exists(md1_path)
                        else:
                            exp_full = False
                    else:
                        exp_full = False
            if exp != "CAGE" and check_pval:
                if exp_full:
                    if not self.is_bigwig_complete(bios_name, exp):
                        exp_full = False

        return exp_full
        
    def is_bios_complete(self, bios_name):
        
        if self.merge_ct and self.eic==False:
            available_exps = list(self.navigation[bios_name].keys())
        else:
            try:
                available_exps = self.df1.loc[self.df1['Accession'] == bios_name].dropna(axis=1).columns.tolist()[1:]
                available_exps.remove("Accession")
            except Exception as e:
                return f"Error reading DF1.csv: {e}"

        missing_exp = []
        for exp in available_exps:
            if self.merge_ct and self.eic==False:
                exp_full = self.is_exp_complete(self.navigation[bios_name][exp][0].split("/")[-3], exp)
            else:
                exp_full = self.is_exp_complete(bios_name, exp)
                        
            if not exp_full:
                missing_exp.append(exp)

        return missing_exp

    def fix_bios(self, bios_name):
        missing_exp = self.is_bios_complete(bios_name)
        missingrows = []
        if len(missing_exp) > 0:
            print(f"Biosample: {bios_name}, Missing Experiments: {missing_exp}")
            print(f"fixing {bios_name}!")
            for exp in missing_exp:
                rows = self.df3.loc[(self.df3["bios"] == bios_name)&(self.df3["exp"] == exp), :]
                missingrows.append(rows)

            missingrows = pd.concat(missingrows, axis=0).reset_index(drop=True)
            # print(missingrows)
            for i in range(len(missingrows)):
                dl_dict = {}
                dl_dict["url"] = missingrows.loc[i, "url"]
                dl_dict["save_dir_name"] = missingrows.loc[i, "save_dir_name"]
                dl_dict["exp"] = missingrows.loc[i, "exp"]
                dl_dict["bios"] = missingrows.loc[i, "bios"]
                single_download(dl_dict)
    
    def is_bigwig_complete(self, bios_name, exp):
        chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        full = 1
        if "signal_BW_res25" in os.listdir(os.path.join(self.base_path, bios_name, exp)):
            for c in chrs:
                if c+".npz" not in os.listdir(os.path.join(self.base_path, bios_name, exp, "signal_BW_res25")):
                    full = 0
        else:
            full = 0
        return full

    def filter_nav_complete_exps(self):
        for bios in list(self.navigation.keys()):
            for exp in list(self.navigation[bios].keys()):
                if os.path.exists(os.path.join(self.base_path, bios, exp)):
                    if not self.is_exp_complete(bios, exp):
                        del self.navigation[bios][exp]
                else:
                    del self.navigation[bios][exp]

    def get_signal_pval_bigwig(self, bios_name, exp, assembly="GRCh38", attempt=0):
        def select_preferred_row(df):
            if df.empty:
                raise ValueError("The DataFrame is empty. Cannot select a preferred row.")
            
            # Define preferences
            preferences = [
                ('derived_from_bam', True),
                ('bio_replicate_number', lambda x: len(x) == 1),
                ('same_bios', True),
                ('default', True)
            ]

            for column, condition in preferences:
                if len(df) > 1:
                    if callable(condition):
                        df = df[df[column].apply(condition)]
                    else:
                        df = df[df[column] == condition]
                if len(df) == 1:
                    return df.iloc[0]
            
            # Sort by date_created if still multiple rows
            if len(df) > 1:
                df['date_created'] = pd.to_datetime(df['date_created'])
                df = df.sort_values(by='date_created', ascending=False)

            # Return the top row of the filtered DataFrame
            return df.iloc[0]

        bios_path = os.path.join(self.base_path, bios_name)
        exp_path = os.path.join(bios_path, exp)
        
        if not os.path.exists(os.path.join(exp_path, 'signal_pval_res25')):
            try:
                with open(os.path.join(exp_path, 'file_metadata.json'), 'r') as file:
                    exp_md = json.load(file)
                
                bam_accession = exp_md["accession"][list(exp_md["accession"].keys())[0]]
                
                exp_url = "https://www.encodeproject.org{}".format(exp_md["experiment"][list(exp_md["experiment"].keys())[0]])
                exp_respond = requests.get(exp_url, headers=self.headers)
                exp_results = exp_respond.json()
                
                e_fileslist = list(exp_results['original_files'])
                e_files_navigation = []

                for ef in e_fileslist:
                    efile_respond = requests.get("https://www.encodeproject.org{}".format(ef), headers=self.headers)
                    efile_results = efile_respond.json()

                    filter_statement = bool(
                        efile_results['file_format'] == "bigWig" and 
                        efile_results['output_type'] in ['signal p-value', "read-depth normalized signal"] and 
                        efile_results['assembly']==assembly and 
                        efile_results['status'] == "released"
                    )

                    if filter_statement:

                        if "origin_batches" in efile_results.keys():
                            if ',' not in str(efile_results['origin_batches']):
                                e_file_biosample = str(efile_results['origin_batches'])
                                e_file_biosample = e_file_biosample.replace('/', '')
                                e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                            else:
                                repnumber = int(efile_results['biological_replicates'][0]) - 1
                                e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                        else:
                            repnumber = int(efile_results['biological_replicates'][0]) - 1
                            e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                        
                        
                        parsed = [exp, efile_results['accession'], bios_name,
                            efile_results['file_format'], efile_results['output_type'], 
                            efile_results['dataset'], efile_results['biological_replicates'], 
                            efile_results['file_size'], efile_results['assembly'], 
                            "https://www.encodeproject.org{}".format(efile_results['href']), 
                            efile_results['date_created'], efile_results['status']]
                        
                        if "preferred_default" in efile_results.keys():
                            parsed.append(efile_results["preferred_default"])
                        else:
                            parsed.append(None)
                        
                        if bam_accession in "|".join(efile_results["derived_from"]):
                            parsed.append(True)
                        else:
                            parsed.append(False)

                        if e_file_biosample == bios_name:
                            parsed.append(True)
                        else:
                            parsed.append(False)

                        e_files_navigation.append(parsed)
                
                e_files_navigation = pd.DataFrame(e_files_navigation, columns=[
                        'assay', 'accession', 'biosample', 'file_format', 
                        'output_type', 'experiment', 'bio_replicate_number', 
                        'file_size', 'assembly', 'download_url', 'date_created', 
                        'status', "default", "derived_from_bam", "same_bios"])
                
                # e_files_navigation['date_created'] = pd.to_datetime(e_files_navigation['date_created'])
                # e_files_navigation = e_files_navigation[e_files_navigation['date_created'] == e_files_navigation['date_created'].max()]

                best_file = select_preferred_row(e_files_navigation)
                
                # if len(e_files_navigation) > 0:
                #     print(e_files_navigation, "\n")
                # else:
                #     print(bios_name, exp, exp_md["experiment"][list(exp_md["experiment"].keys())[0]])

                # url = "https://www.encodeproject.org{}".format(efile_results['href'])
                save_dir_name = os.path.join(exp_path, best_file['accession']+".bigWig")
                
                download_prompt = {"url":best_file["download_url"], "save_dir_name":save_dir_name, "exp":exp, "bios":bios_name}

                try:
                    if not self.is_bigwig_complete(bios_name, exp):
                        if os.path.exists(f"{exp_path}/signal_BW_res25/"):
                            shutil.rmtree(f"{exp_path}/signal_BW_res25/")
                            print(f"cleaned up old files...")

                        t0 = datetime.datetime.now()
                        single_download(download_prompt)
                        t1 = datetime.datetime.now()
                        print(f"download took {t1-t0}")
                        binned_bw = get_binned_values(save_dir_name)
                        t2 = datetime.datetime.now()
                        print(f"binning took {t2-t1}")

                        os.mkdir(f"{exp_path}/signal_BW_res25")

                        for chr, data in binned_bw.items():
                            np.savez_compressed(
                                f"{exp_path}/signal_BW_res25/{chr}.npz", 
                                np.array(data))
                        
                        os.system(f"rm {save_dir_name}")
                    else:
                        print(f"{exp_path}/signal_BW_res25/ already exists!")

                except:
                    print(f"failed at downloading/processing {bios_name}-{exp}, attempt={attempt}")
                    if os.path.exists(save_dir_name):
                        os.system(f"rm {save_dir_name}")

                    attempt +=1
                    if attempt<10:
                        print("retrying...")
                        self.get_signal_pval_bigwig(bios_name, exp, assembly=assembly, attempt=attempt)
                
            except:
                print(f"skipped {bios_name}-{exp}")

    def get_peaks_bigbed(self, bios_name, exp, assembly="GRCh38", attempt=0):
        def select_preferred_row(df):
            if df.empty:
                raise ValueError("The DataFrame is empty. Cannot select a preferred row.")
            
            # Define preferences
            preferences = [
                ('derived_from_bam', True),
                ('bio_replicate_number', lambda x: len(x) == 1),
                ('same_bios', True),
                ('default', True)
            ]

            for column, condition in preferences:
                if len(df) > 1:
                    if callable(condition):
                        df = df[df[column].apply(condition)]
                    else:
                        df = df[df[column] == condition]
                if len(df) == 1:
                    return df.iloc[0]
            
            # Sort by date_created if still multiple rows
            if len(df) > 1:
                df['date_created'] = pd.to_datetime(df['date_created'])
                df = df.sort_values(by='date_created', ascending=False)

            # Return the top row of the filtered DataFrame
            return df.iloc[0]

        bios_path = os.path.join(self.base_path, bios_name)
        exp_path = os.path.join(bios_path, exp)
        
        if True:#not os.path.exists(os.path.join(exp_path, 'signal_pval_res25')):
            try:
                with open(os.path.join(exp_path, 'file_metadata.json'), 'r') as file:
                    exp_md = json.load(file)
                
                bam_accession = exp_md["accession"][list(exp_md["accession"].keys())[0]]
                
                exp_url = "https://www.encodeproject.org{}".format(exp_md["experiment"][list(exp_md["experiment"].keys())[0]])
                exp_respond = requests.get(exp_url, headers=self.headers)
                exp_results = exp_respond.json()
                
                e_fileslist = list(exp_results['original_files'])
                e_files_navigation = []

                for ef in e_fileslist:
                    efile_respond = requests.get("https://www.encodeproject.org{}".format(ef), headers=self.headers)
                    efile_results = efile_respond.json()

                    filter_statement = bool(
                        efile_results['file_format'] == "bigBed" and 
                        "peaks" in efile_results['output_type'] and 
                        efile_results['assembly']==assembly and 
                        efile_results['status'] == "released"
                    )

                    if filter_statement:

                        if "origin_batches" in efile_results.keys():
                            if ',' not in str(efile_results['origin_batches']):
                                e_file_biosample = str(efile_results['origin_batches'])
                                e_file_biosample = e_file_biosample.replace('/', '')
                                e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                            else:
                                repnumber = int(efile_results['biological_replicates'][0]) - 1
                                e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                        else:
                            repnumber = int(efile_results['biological_replicates'][0]) - 1
                            e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                        
                        
                        parsed = [exp, efile_results['accession'], bios_name,
                            efile_results['file_format'], efile_results['output_type'], 
                            efile_results['dataset'], efile_results['biological_replicates'], 
                            efile_results['file_size'], efile_results['assembly'], 
                            "https://www.encodeproject.org{}".format(efile_results['href']), 
                            efile_results['date_created'], efile_results['status']]
                        
                        if "preferred_default" in efile_results.keys():
                            parsed.append(efile_results["preferred_default"])
                        else:
                            parsed.append(None)
                        
                        if bam_accession in "|".join(efile_results["derived_from"]):
                            parsed.append(True)
                        else:
                            parsed.append(False)

                        if e_file_biosample == bios_name:
                            parsed.append(True)
                        else:
                            parsed.append(False)

                        e_files_navigation.append(parsed)
                
                e_files_navigation = pd.DataFrame(e_files_navigation, columns=[
                        'assay', 'accession', 'biosample', 'file_format', 
                        'output_type', 'experiment', 'bio_replicate_number', 
                        'file_size', 'assembly', 'download_url', 'date_created', 
                        'status', "default", "derived_from_bam", "same_bios"])
                
                e_files_navigation['date_created'] = pd.to_datetime(e_files_navigation['date_created'])
                e_files_navigation = e_files_navigation[e_files_navigation['date_created'] == e_files_navigation['date_created'].max()]

                best_file = select_preferred_row(e_files_navigation)

                url = "https://www.encodeproject.org{}".format(efile_results['href'])
                save_dir_name = os.path.join(exp_path, best_file['accession']+".bigBed")
                
                download_prompt = {"url":best_file["download_url"], "save_dir_name":save_dir_name, "exp":exp, "bios":bios_name}

                try:
                    if os.path.exists(f"{exp_path}/peaks_res25/"):
                        shutil.rmtree(f"{exp_path}/peaks_res25/")
                        print(f"cleaned up old files...")

                    t0 = datetime.datetime.now()
                    single_download(download_prompt)
                    t1 = datetime.datetime.now()
                    print(f"download took {t1-t0}")
                    binned_bw = get_binned_bigBed_peaks(save_dir_name)
                    t2 = datetime.datetime.now()
                    print(f"binning took {t2-t1}")

                    os.mkdir(f"{exp_path}/peaks_res25")

                    for chr, data in binned_bw.items():
                        np.savez_compressed(
                            f"{exp_path}/peaks_res25/{chr}.npz", 
                            np.array(data))
                    
                    os.system(f"rm {save_dir_name}")

                except:
                    print(f"failed at downloading/processing {bios_name}-{exp}, attempt={attempt}")
                    if os.path.exists(save_dir_name):
                        os.system(f"rm {save_dir_name}")

                    attempt +=1
                    if attempt<10:
                        print("retrying...")
                        self.get_signal_pval_bigwig(bios_name, exp, assembly=assembly, attempt=attempt)
                
            except:
                print(f"skipped {bios_name}-{exp}")

    def mp_fix_DS(self, n_p=2):
        bios_list = self.df1.Accession.to_list()
        random.shuffle(bios_list)
        with mp.Pool(n_p) as p:
            p.map(self.fix_bios, bios_list)

    def DS_checkup(self):
        bios_list = self.df1.Accession.to_list()
        is_comp = []
        for bs in bios_list:
            missing = self.is_bios_complete(bs)
            if len(missing) > 0:
                is_comp.append(0)
            else:
                is_comp.append(1)
        
        return sum(is_comp) / len(is_comp)

    def set_alias(self, excludes=["ChIA-PET", "CAGE", "RNA-seq"]):
        if os.path.exists(self.alias_path):
            with open(self.alias_path, 'r') as file:
                self.aliases = json.load(file)
            return

        """Set aliases for biosamples, experiments, and donors based on data availability."""
        self.df1.set_index('Accession', inplace=True)
        self.df1 = self.df1.drop("Unnamed: 0", axis=1)

        # Alias for biosamples
        biosample_counts = self.df1.count(axis=1).sort_values(ascending=False)
        num_biosamples = len(biosample_counts)
        biosample_alias = {biosample: f"E{str(index+1).zfill(len(str(num_biosamples)))}" for index, biosample in enumerate(biosample_counts.index)}

        # Alias for experiments
        experiment_counts = self.df1.count().sort_values(ascending=False)
        experiment_counts = experiment_counts.drop(excludes)
        num_experiments = len(experiment_counts)
        experiment_alias = {
            experiment: f"M{str(index+1).zfill(len(str(num_experiments)))}" for index, experiment in enumerate(
                experiment_counts.index)}

        self.aliases = {
            "biosample_aliases": biosample_alias,
            "experiment_aliases": experiment_alias}

        with open(self.alias_path, 'w') as file:
            json.dump(self.aliases, file, indent=4)

    def navigate_bios_exps(self):
        """Navigate all biosample-experiment pairs and save in JSON."""
        navigation = {}
        for bios in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, bios)):
                navigation[bios] = {}
                for exp in os.listdir(os.path.join(self.base_path, bios)):
                    exp_path = os.path.join(self.base_path, bios, exp)
                    if os.path.isdir(exp_path):
                        navigation[bios][exp] = os.listdir(exp_path)
        
        with open(self.navigation_path, 'w') as file:
            json.dump(navigation, file, indent=4)

    def navigate_merge_celltypes(self, min_exp=3):
        celltypes = {ct:[] for ct in self.df2["Biosample term name"].unique()}
        for i in range(len(self.df2)):
            celltypes[self.df2["Biosample term name"][i]].append(self.df2["Accession"][i])

        # Create list to store data for DataFrame
        df_data = []
        
        for ct in celltypes.keys():
            for sub_bios in celltypes[ct]:
                # Get experiments for this biosample
                if os.path.exists(os.path.join(self.base_path, sub_bios)):
                    experiments = [exp for exp in os.listdir(os.path.join(self.base_path, sub_bios)) if os.path.isdir(os.path.join(self.base_path, sub_bios, exp))]
                else:
                    continue
                
                # Get donor info
                donor_info = {}
                donor_info_path = os.path.join(self.base_path, sub_bios, "donor.json")
                if os.path.exists(donor_info_path):
                    with open(donor_info_path, 'r') as file:
                        donor_info = json.load(file)
                
                # Create one row per experiment
                for exp in experiments:
                    row = {
                        'biosample_term_name': ct,
                        'accession': sub_bios,
                        'experiment': exp,
                        'donor_status': donor_info.get('Status'),
                        'donor_accession': donor_info.get('Accession'),
                        'donor_age': donor_info.get('Age'),
                        'donor_life_stage': donor_info.get('Life stage'),
                        'donor_sex': donor_info.get('Sex'),
                        'donor_organism': donor_info.get('Organism'),
                        'isogenic_replicates': self.df2.loc[self.df2['Accession'] == sub_bios, 'isogenic_replicates'].iloc[0],
                        'source': self.df2.loc[self.df2['Accession'] == sub_bios, 'Source'].iloc[0]
                    }
                    df_data.append(row)
        
        # Create DataFrame
        celltype_df = pd.DataFrame(df_data)
        
        # Sort by celltype and experiment
        celltype_df = celltype_df.sort_values(['biosample_term_name', 'accession', "experiment"]).reset_index(drop=True)

        merged_data = {}
        for cell_type, group_df in celltype_df.groupby('biosample_term_name'):
            

            # Find replicate pairs
            replicate_list = group_df['isogenic_replicates'].unique()
            unique_replicates = [rep for rep in replicate_list if pd.notna(rep)]
            all_replicates = [rep.split(',') for rep in unique_replicates]
            unique_replicates = list(set(item for sublist in all_replicates for item in sublist))

            rep_map = []
            for i, row in group_df.iterrows():
                if pd.notna(row['isogenic_replicates']):
                    replicates = row['isogenic_replicates'].split(',')
                    rep_map.append(tuple(sorted(replicates + [row['accession']])))

            rep_map = list(set(rep_map))

            replicates = []

            # Extract replicates with similar experiments
            exp_counts = {}
            for rep in unique_replicates:
                if rep in group_df['accession'].values:
                    exp_counts[rep] = set(group_df[group_df['accession'] == rep]['experiment'].values)

            for rep_gp in rep_map:
                shared_exps = set(group_df[group_df['accession'] == rep_gp[0]]['experiment'].values)
                for rep in rep_gp[1:]:
                    exps = set(group_df[group_df['accession'] == rep]['experiment'].values)
                    shared_exps = shared_exps.intersection(exps)
                
                if len(shared_exps) > min_exp:
                    replicates.append([rep_gp, shared_exps])
            
            for rep_gp, shared_exps in replicates:
                for rep in rep_gp:
                    for exp in shared_exps:
                        # Remove the row corresponding to rep-exp from group_df
                        group_df = group_df[~((group_df['accession'] == rep) & (group_df['experiment'] == exp))]
            

            non_replicate = []
            # Handle remaining experiments in group_df
            if not group_df.empty:
                # Get unique experiments
                unique_exps = group_df['experiment'].unique()
                
                for exp in unique_exps:
                    exp_df = group_df[group_df['experiment'] == exp]
                    
                    if len(exp_df) > 1:
                        # Score each option based on multiple criteria
                        scores = {}
                        
                        # Count prevalence of each accession across all experiments
                        accession_counts = group_df['accession'].value_counts()
                        
                        # Count prevalence of each donor across all experiments
                        donor_counts = group_df['donor_accession'].value_counts()
                        
                        # Count prevalence of each source across all experiments
                        source_counts = group_df['source'].value_counts()
                        
                        for _, row in exp_df.iterrows():
                            score = (
                                accession_counts[row['accession']] * 20 +  # Weight accession highest
                                donor_counts[row['donor_accession']] * 10 +  # Weight donor second
                                source_counts[row['source']]  * 5                 # Weight source third
                            )
                            scores[row.name] = score
                        
                        # Select the row with highest score
                        best_row_idx = max(scores.items(), key=lambda x: x[1])[0]
                        # group_df = group_df[group_df.index != best_row_idx].copy()
                        non_replicate.append(group_df[group_df.index == best_row_idx].iloc[0])
            
            # create merged data for replicate groups and non-replicate experiments
            for i, (rep_gp, shared_exps) in enumerate(replicates):
                for j, rep in enumerate(rep_gp):
                    name = f"{cell_type.replace(' ', '_').replace('-', '_')}_grp{i+1}_rep{j+1}"
                    
                    merged_data[name] = {}
                    for exp in shared_exps:
                        merged_data[name][exp] = []
                        exp_path = os.path.join(self.base_path, rep, exp)
                        exp_files = os.listdir(exp_path)
                        for f in exp_files:
                            merged_data[name][exp].append(os.path.join(exp_path, f))

            non_replicate = pd.DataFrame(non_replicate).reset_index(drop=True)
            if len(non_replicate) > min_exp:
                name = f"{cell_type.replace(' ', '_').replace('-', '_')}_nonrep"
                merged_data[name] = {}
                for i in range(len(non_replicate)):
                    exp = non_replicate["experiment"][i]
                    
                    exp_path = os.path.join(self.base_path, non_replicate["accession"][i], exp)
                    exp_files = os.listdir(exp_path)
                    if len(exp_files) == 0:
                        continue
                    merged_data[name][exp] = []
                    for f in exp_files:
                        merged_data[name][exp].append(os.path.join(exp_path, f))

        with open(self.merged_navigation_path, 'w') as file:
            json.dump(merged_data, file, indent=4)

    def init_eic(self, target_split="train"):
        eic_nav_path = os.path.join(self.base_path, "navigation_eic.json")
        eic_aliases_path = os.path.join(self.base_path, "aliases_eic.json")
        eic_split_path = os.path.join(self.base_path, "train_va_test_split_eic.json")

        if os.path.exists(eic_nav_path) and os.path.exists(eic_aliases_path) and os.path.exists(eic_split_path):
            with open(eic_nav_path, 'r') as file:
                self.navigation  = json.load(file)

            with open(eic_split_path, 'r') as file:
                self.split_dict  = json.load(file)

            with open(eic_aliases_path, 'r') as file:
                self.aliases  = json.load(file)

            for bios in list(self.navigation.keys()):
                if self.split_dict[bios] != target_split:
                    del self.navigation[bios]
            
            return

        celltypes = {ct:[] for ct in self.df2["Biosample term name"].unique()}
        for i in range(len(self.df2)):
            celltypes[self.df2["Biosample term name"][i]].append(self.df2["Accession"][i])

        split = {} # keys are bios accessions | values are "train"/"test"/"val"
        aliases = {
            "biosample_aliases": {}, # keys are bios accessions
            "experiment_aliases": {} # keys are exp names
        }

        so_far = {}
        missed = []
        
        to_move = {
            "training_data":{}, # keys are cell_types | values are file paths (list)
            "validation_data":{},  # keys are cell_types | values are file paths (list)
            "blind_data":{}  # keys are cell_types | values are file paths (list)
        }

        for i in range(self.eic_df.shape[0]):
            exp_accession = self.eic_df["experiment"][i] 
            exp_type = self.eic_df["mark/assay"][i]
            data_type = self.eic_df["data_type"][i]

            ct = self.eic_df["cell_type"][i].replace("_", " ")
            if ct == "H1-hESC":
                ct = "H1"
            elif ct == "skin fibroblast":
                ct = "fibroblast of skin of back"

            if ct not in aliases["biosample_aliases"].keys():
                aliases["biosample_aliases"][ct] = self.eic_df["cell_type_id"][i]

            if exp_type not in aliases["experiment_aliases"].keys():
                aliases["experiment_aliases"][exp_type] = self.eic_df["mark_id"][i]
            
            # find corresponding bios in df1
            if exp_accession in self.df1[exp_type].values:
                bios_accession = self.df1.loc[self.df1[exp_type] == exp_accession, "Accession"].values[0]

                if self.is_exp_complete(bios_accession, exp_type):
                    if ct not in to_move[data_type].keys():
                        to_move[data_type][ct] = []

                    to_move[data_type][ct].append(os.path.join(self.base_path, bios_accession, exp_type))
                    
                    if ct not in so_far.keys():
                        so_far[ct] = []
                    so_far[ct].append(bios_accession)

                else:
                    missed.append([exp_type, exp_accession, data_type, ct])
                    # print("missing files for ", [exp_type, exp_accession, data_type, ct], bios_accession)

            else:
                missed.append([exp_type, exp_accession, data_type, ct])

        for i in range(len(missed)):
            found = False
            for j in celltypes[missed[i][-1]]:
                if missed[i][0] in self.navigation[j].keys():
                    if len(self.is_bios_complete(j))==0:
                        bios_accession = j
                        ct = missed[i][-1]
                        data_type = missed[i][-2]
                        exp_type = missed[i][0]
                        if ct not in to_move[data_type].keys():
                            to_move[data_type][ct] = []
                        to_move[data_type][ct].append(os.path.join(self.base_path, bios_accession, exp_type))

                        found == True
                        break

        for ct, files in to_move["training_data"].items():
            for f in files:
                dst = os.path.join(self.base_path, f"T_{ct.replace(' ', '_')}", f.split("/")[-1])
                if not os.path.exists(dst):
                    shutil.copytree(f, dst)
                    print(f"copying {dst}")

                elif not self.is_exp_complete(f"T_{ct.replace(' ', '_')}", f.split("/")[-1]):
                    shutil.rmtree(dst)
                    shutil.copytree(f, dst)
                    print(f"copying {dst}")

                split[f"T_{ct.replace(' ', '_')}"] = "train"

        for ct, files in to_move["validation_data"].items():
            for f in files:
                dst = os.path.join(self.base_path, f"V_{ct.replace(' ', '_')}", f.split("/")[-1])
                if not os.path.exists(dst):
                    shutil.copytree(f, dst)
                    print(f"copying {dst}")

                elif not self.is_exp_complete(f"V_{ct.replace(' ', '_')}", f.split("/")[-1]):
                    shutil.rmtree(dst)
                    shutil.copytree(f, dst)
                    print(f"copying {dst}")
                    
                split[f"V_{ct.replace(' ', '_')}"] = "val"
        
        for ct, files in to_move["blind_data"].items():
            for f in files:
                dst = os.path.join(self.base_path, f"B_{ct.replace(' ', '_')}", f.split("/")[-1])

                if not os.path.exists(dst):
                    shutil.copytree(f, dst)
                    print(f"copying {dst}")
                
                elif not self.is_exp_complete(f"B_{ct.replace(' ', '_')}", f.split("/")[-1]):
                    shutil.rmtree(dst)
                    shutil.copytree(f, dst)
                    print(f"copying {dst}")

                split[f"B_{ct.replace(' ', '_')}"] = "test"

        navigation = {} # keys are bios accessions | values are "train"/"test"/"val"
        for bios in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, bios)):
                if bios[0] in ["V", "T", "B"]:
                    navigation[bios] = {}
                    for exp in os.listdir(os.path.join(self.base_path, bios)):
                        exp_path = os.path.join(self.base_path, bios, exp)
                        if os.path.isdir(exp_path):
                            navigation[bios][exp] = os.listdir(exp_path)
        
        with open(eic_nav_path, 'w') as file:
            json.dump(navigation, file, indent=4)

        with open(eic_split_path, 'w') as file:
            json.dump(split, file, indent=4)

        with open(eic_aliases_path, 'w') as file:
            json.dump(aliases, file, indent=4)
        
        self.navigation = navigation
        self.split_dict = split
        self.aliases = aliases

        for bios in list(self.navigation.keys()):
            if self.split_dict[bios] != target_split:
                del self.navigation[bios]

    def filter_navigation(self, include=[], exclude=[]):
        """
        filter based on a list of assays to include
        """
        for bios in list(self.navigation.keys()):
            bios_exps = list(self.navigation[bios].keys())
            if bios[0] in ["T", "V", "B"]:
                del self.navigation[bios]

            elif self.must_have_chr_access: 
                if ("ATAC-seq" not in bios_exps) and ("DNase-seq" not in bios_exps):
                    del self.navigation[bios]

        if len(include) == 0 and len(exclude) == 0:
            return

        elif len(exclude) == 0 and len(include) != 0:
            for bios in list(self.navigation.keys()):
                for exp in list(self.navigation[bios].keys()):
                    if exp not in include:
                        del self.navigation[bios][exp]

            for exp in list(self.aliases["experiment_aliases"].keys()):
                if exp not in include:
                    del self.aliases["experiment_aliases"][exp]

        elif len(include) == 0 and len(exclude) != 0:
            for bios in list(self.navigation.keys()):
                for exp in list(self.navigation[bios].keys()):
                    if exp in exclude:
                        del self.navigation[bios][exp]
                        
            for exp in exclude:
                if exp in self.aliases["experiment_aliases"].keys():
                    del self.aliases["experiment_aliases"][exp]

        else:
            return
                        
    def merged_train_val_test_split(self, random_seed=42, splits=(0.8, 0.10, 0.10)):
        """
        Split cell types according to specific rules and proportions:
        1. Split cell types into train (70%), validation (15%), and test (15%) sets
        2. For any of the self.navigation keys, if self.has_rnaseq(key) is true, assign it to test
        """
        if os.path.exists(self.merged_split_path):
            with open(self.merged_split_path, 'r') as file:
                self.split_dict = json.load(file)
            return

        if sum(splits) != 1:
            raise ValueError("Split proportions must sum to 1")

        random.seed(random_seed)
        self.split_dict = {}
        
        # Group keys by cell type
        cell_types = {}
        rna_seq_keys = []
        
        # First pass: organize keys and handle RNA-seq cases
        for key in self.navigation.keys():
            if self.has_rnaseq(key):
                rna_seq_keys.append(key)
                continue
                
            cell_name = key.split('_grp')[0] if '_grp' in key else key.split('_nonrep')[0]
            if cell_name not in cell_types:
                cell_types[cell_name] = []
            cell_types[cell_name].append(key)

        # Calculate split sizes
        train_size, val_size, test_size = splits
        num_cell_types = len(cell_types)
        train_ct = int(num_cell_types * train_size)
        val_ct = int(num_cell_types * val_size)
        # test_ct will be the remainder

        # Randomly shuffle cell types
        cell_type_names = list(cell_types.keys())
        random.shuffle(cell_type_names)

        # Split cell types into train/val/test
        train_types = cell_type_names[:train_ct]
        val_types = cell_type_names[train_ct:train_ct + val_ct]
        test_types = cell_type_names[train_ct + val_ct:]

        # Assign splits based on cell type
        for cell_type in train_types:
            for key in cell_types[cell_type]:
                self.split_dict[key] = 'train'

        for cell_type in val_types:
            for key in cell_types[cell_type]:
                self.split_dict[key] = 'val'

        for cell_type in test_types:
            for key in cell_types[cell_type]:
                self.split_dict[key] = 'test'

        # Handle RNA-seq keys (always go to test)
        for key in rna_seq_keys:
            self.split_dict[key] = 'test'

        for bios in self.navigation.keys():
            if "RNA-seq" in self.navigation[bios].keys():
                self.split_dict[bios] = "test"

        # # Save split dictionary
        with open(self.merged_split_path, 'w') as file:
            json.dump(self.split_dict, file, indent=4)

        # Print statistics
        train_count = sum(1 for v in self.split_dict.values() if v == 'train')
        val_count = sum(1 for v in self.split_dict.values() if v == 'val')
        test_count = sum(1 for v in self.split_dict.values() if v == 'test')
        total_count = len(self.split_dict)

        print("\nSplit Statistics:")
        print(f"Total cell types: {num_cell_types}")
        print(f"Train cell types: {len(train_types)} ({len(train_types)/num_cell_types:.1%})")
        print(f"Val cell types: {len(val_types)} ({len(val_types)/num_cell_types:.1%})")
        print(f"Test cell types: {len(test_types)} ({len(test_types)/num_cell_types:.1%})")
        print(f"\nTotal samples: {total_count}")
        print(f"Train samples: {train_count} ({train_count/total_count:.1%})")
        print(f"Val samples: {val_count} ({val_count/total_count:.1%})")
        print(f"Test samples: {test_count} ({test_count/total_count:.1%})")
        print(f"RNA-seq samples in test: {len(rna_seq_keys)}")

    def train_val_test_split(self, splits=(0.7, 0.15, 0.15), random_seed=42):
        if os.path.exists(self.split_path):
            with open(self.split_path, 'r') as file:
                self.split_dict = json.load(file)
            return

        if sum(splits) != 1:
            raise ValueError("Sum of splits tuple must be 1.")

        train_size, val_size, test_size = splits
        train_data, temp_data = train_test_split(self.df1, test_size=(1 - train_size), random_state=random_seed)
        relative_val_size = val_size / (val_size + test_size)  # Relative size of validation in the temp data
        val_data, test_data = train_test_split(temp_data, test_size=(1 - relative_val_size), random_state=random_seed)
        
        self.split_dict = {}
        
        for idx in train_data.Accession:
            if self.has_rnaseq(idx):
                self.split_dict[idx] = 'test'
            else:
                self.split_dict[idx] = 'train'
        
        for idx in val_data.Accession:
            if self.has_rnaseq(idx):
                self.split_dict[idx] = 'test'
            else:
                self.split_dict[idx] = 'val'
        
        for idx in test_data.Accession:
            self.split_dict[idx] = 'test'
        
        with open(self.split_path, 'w') as file:
            json.dump(self.split_dict, file, indent=4)

    def convert_npz_to_npy(self):
        """Convert all NPZ files to NPY format."""
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.npz'):
                    data = np.load(os.path.join(root, file))
                    for arr_name in data.files:
                        np.save(os.path.join(root, f"{file.replace('.npz', '')}.npy"), data[arr_name])

    def generate_random_loci(self, m, context_length, exclude_chr=['chr21']):
        """Generate random genomic loci, excluding specified chromosomes."""
        self.m_regions = []
        used_regions = {chr: [] for chr in self.chr_sizes.keys() if chr not in exclude_chr}
        for chr in used_regions.keys():
            size = self.chr_sizes[chr]
            m_c = int(m * (size / self.genomesize)) + 1  # Calculate the proportional count of regions to generate
            mii = 0
            while mii < m_c:
                # Generate a random start position that is divisible by self.resolution
                rand_start = random.randint(0, (size - context_length) // self.resolution) * self.resolution
                rand_end = rand_start + context_length

                # Check if the region overlaps with any existing region in the same chromosome
                if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                    if self.is_region_allowed(chr, rand_start, rand_end):
                        self.m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1
        
        while sum([len(v) for v in used_regions.values()]) < m:
            # Generate a random start position that is divisible by self.resolution
            rand_start = random.randint(0, (size - context_length) // self.resolution) * self.resolution
            rand_end = rand_start + context_length

            # Check if the region overlaps with any existing region in the same chromosome
            if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                if self.is_region_allowed(chr, rand_start, rand_end):
                    self.m_regions.append([chr, rand_start, rand_end])
                    used_regions[chr].append((rand_start, rand_end))
                    mii += 1

    def generate_ccre_loci(self, m, context_length, ccre_filename="GRCh38-cCREs.bed", exclude_chr=['chr21']):
        """Generate loci based on CCRE data."""

        # Implement based on available CCRE data.
        self.ccres = pd.read_csv(os.path.join(self.base_path, ccre_filename), sep="\t", header=None)
        self.ccres.columns = ["chrom", "start", "end", "id1", "id2", "desc"]

        self.ccres = self.ccres[self.ccres["chrom"].isin(self.chr_sizes.keys())]
        self.ccres = self.ccres[~self.ccres["chrom"].isin(exclude_chr)]

        self.m_regions = []
        used_regions = {chr: [] for chr in self.ccres['chrom'].unique()}

        # Sort the DataFrame by chromosome and start position
        self.ccres = self.ccres.sort_values(['chrom', 'start'])

        # Select m/2 regions from the DataFrame
        while len(self.m_regions) < (m):
            while True:
                # Select a random row from the DataFrame
                row = self.ccres.sample(1).iloc[0]

                # Generate a start position that is divisible by self.resolution and within the region
                rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                rand_end = rand_start + context_length

                # Check if the region overlaps with any existing region in the same chromosome
                if rand_start >= 0 and rand_end <= self.chr_sizes[row['chrom']]:
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                        if self.is_region_allowed(row['chrom'], rand_start, rand_end):
                            self.m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
    def generate_full_chr_loci(self, context_length, chrs=["chr19"]):
        self.m_regions = []
        if chrs == "gw":
            chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
            chrs.remove("chr21")
            
        for chr in chrs:
            size = (self.chr_sizes[chr] // context_length) * context_length
            for i in range(0, size, context_length):
                if self.is_region_allowed(chr, i, i+context_length):
                    self.m_regions.append([chr, i, i+context_length])
        
    def load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as data:
        # with np.load(file_name, allow_pickle=True, mmap_mode='r') as data:
            return {file_name.split("/")[-3]: data[data.files[0]]}
    
    def load_bios_BW(self, bios_name, locus, DSF, f_format="npz", arcsinh=True):
        if self.eic and bios_name not in self.navigation.keys():
            exps = []
            if os.path.isdir(os.path.join(self.base_path, bios_name)):
                for exp in os.listdir(os.path.join(self.base_path, bios_name)):
                    exp_path = os.path.join(self.base_path, bios_name, exp)
                    if os.path.isdir(exp_path):
                        exps.append(exp)
            
            # print(bios_name, exps)
        else:
            exps = list(self.navigation[bios_name].keys())

        if "RNA-seq" in exps:
            exps.remove("RNA-seq")

        loaded_data = {}
        npz_files = []
        for e in exps:
            if self.merge_ct and self.eic==False:
                l = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"signal_BW_res{self.resolution}", f"{locus[0]}.{f_format}")
            else:
                l = os.path.join(self.base_path, bios_name, e, f"signal_BW_res{self.resolution}", f"{locus[0]}.{f_format}")
            npz_files.append(l)

        # Load files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded = list(executor.map(self.load_npz, npz_files))
        
        if len(locus) == 1:
            for l in loaded:
                for exp, data in l.items():
                    # print(data.dtype)
                    if arcsinh:
                        loaded_data[exp] = np.arcsinh(data).astype(np.float16)
                    else:
                        loaded_data[exp] = data.astype(np.float16)

            return loaded_data

        else:
            start_bin = int(locus[1]) // self.resolution
            end_bin = int(locus[2]) // self.resolution
            for l in loaded:
                for exp, data in l.items():
                    if arcsinh:
                        loaded_data[exp] = np.arcsinh(data[start_bin:end_bin])
                    else:
                        loaded_data[exp] = data[start_bin:end_bin]
            
            return loaded_data
    
    def make_bios_tensor_BW(self, loaded_data, missing_value=-1):
        dtensor = []
        availability = []

        L = len(loaded_data[list(loaded_data.keys())[0]])
        i = 0
        for assay, alias in self.aliases["experiment_aliases"].items():
            
            # assert i+1 == int(alias.replace("M",""))
            if assay in loaded_data.keys():
                dtensor.append(loaded_data[assay])
                availability.append(1)

            else:
                dtensor.append([missing_value for _ in range(L)])
                availability.append(0)

            i += 1
        
        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))
        return dtensor, availability

    def make_region_tensor_BW(self, loaded_data):
        data, availability = [], []
        for i in range(len(loaded_data)):
            d, avl = self.make_bios_tensor_BW(loaded_data[i])
            data.append(d)
            availability.append(avl)
        
        data, availability = torch.stack(data), torch.stack(availability)
        return data, availability

    def load_bios(self, bios_name, locus, DSF, f_format="npz"):
        """Load all available experiments for a given biosample and locus."""

        if self.eic and bios_name not in self.navigation.keys():
            # print("bios_name not in navigation! ", bios_name)
            exps = []
            if os.path.isdir(os.path.join(self.base_path, bios_name)):
                for exp in os.listdir(os.path.join(self.base_path, bios_name)):
                    exp_path = os.path.join(self.base_path, bios_name, exp)
                    if os.path.isdir(exp_path):
                        # print(exp_path, os.listdir(exp_path))
                        exps.append(exp)
            # print(bios_name, exps)
            
        else:
            exps = list(self.navigation[bios_name].keys())

        if "RNA-seq" in exps:
            exps.remove("RNA-seq")

        loaded_data = {}
        loaded_metadata = {}

        npz_files = []
        for e in exps:
            if self.merge_ct and self.eic==False:
                l =    os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
                jsn1 = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
                jsn2 = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), "file_metadata.json")

            else:
                l = os.path.join(self.base_path, bios_name, e, f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
                jsn1 = os.path.join(self.base_path, bios_name, e, f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
                jsn2 = os.path.join(self.base_path, bios_name, e, "file_metadata.json")

            npz_files.append(l)
            with open(jsn1, 'r') as jsnfile:
                md1 = json.load(jsnfile)

            with open(jsn2, 'r') as jsnfile:
                md2 = json.load(jsnfile)

            md = {
                "depth":md1["depth"], "coverage":md1["coverage"], 
                "read_length":md2["read_length"], "run_type":md2["run_type"] 
            }
            loaded_metadata[e] = md

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(self.load_npz, npz_files):
                if result is not None:
                    for exp, data in result.items():
                        if len(locus) == 1:
                            loaded_data[exp] = data.astype(np.int16)
                                
                        else:
                            start_bin = int(locus[1]) // self.resolution
                            end_bin = int(locus[2]) // self.resolution
                            loaded_data[exp] = data[start_bin:end_bin]
            
        return loaded_data, loaded_metadata

    def select_region_from_loaded_data(self, loaded_data, locus):
        region = {}
        start_bin = int(locus[1]) // self.resolution
        end_bin = int(locus[2]) // self.resolution
        for exp, data in loaded_data.items():
            region[exp] = data[start_bin:end_bin]
        
        return region

    def fill_in_y_prompt(self, md, missing_value=-1, stat_type="Median", sample=False):
        if sample:     
            def typed_choice_retry(seq, want_type, max_tries=1000):
                for _ in itertools.repeat(None, max_tries):
                    pick = random.choice(seq)
                    print(pick, type(pick))
                    if isinstance(pick, want_type):
                        return pick
                raise LookupError(f"Found no {want_type.__name__} after {max_tries} draws")

            if len(md.shape) == 2:
                for i, (assay, alias) in enumerate(self.aliases["experiment_aliases"].items()):
                    # print(self.RawExpMetaData.loc["depth", assay])
                    # print(self.RawExpMetaData.loc["coverage", assay])
                    # print(self.RawExpMetaData.loc["read_length", assay])
                    # print(self.RawExpMetaData.loc["run_type", assay])
                    # return
                    
                    if torch.all(md[:, i] == missing_value):
                        md[0, i] = float(np.log2(random.choice(self.RawExpMetaData.loc["depth", assay])))
                        md[1, i] = float(random.choice(self.RawExpMetaData.loc["coverage", assay]))
                        md[2, i] = float(random.choice(self.RawExpMetaData.loc["read_length", assay]))
                        md[3, i] =  float(bool("pair" in random.choice(self.RawExpMetaData.loc["run_type", assay])))
                        print(float(bool("pair" in random.choice(self.RawExpMetaData.loc["run_type", assay]))))

            else:
                for i, (assay, alias) in enumerate(self.aliases["experiment_aliases"].items()):
                    # print(self.RawExpMetaData.loc["depth", assay])
                    # print(self.RawExpMetaData.loc["coverage", assay])
                    # print(self.RawExpMetaData.loc["read_length", assay])
                    # print(self.RawExpMetaData.loc["run_type", assay])
                    # return

                    for b in range(md.shape[0]):
                        if torch.all(md[b, :, i] == missing_value):
                            # md[b, 0, i] = float(np.log2(random.choice(self.RawExpMetaData.loc["depth", assay])))
                            md[b, 0, i] = float(np.log2(typed_choice_retry(self.RawExpMetaData.loc["depth", assay], int)))
                            md[b, 1, i] = float(random.choice(self.RawExpMetaData.loc["coverage", assay]))
                            md[b, 2, i] = float(random.choice(self.RawExpMetaData.loc["read_length", assay]))
                            md[b, 3, i] = float(bool("pair" in random.choice(self.RawExpMetaData.loc["run_type", assay])))
                            print(float(bool("pair" in random.choice(self.RawExpMetaData.loc["run_type", assay]))))
            
            return md

        else:
            if stat_type not in ["Min", "Max", "Median"]:
                raise

            # Create lookup dictionary once outside the loops
            stat_lookup = {}
            for assay in self.aliases["experiment_aliases"]:
                stat_lookup[assay] = {
                    "depth": self.expstats.loc[(self.expstats["Experiment"]==assay) & (self.expstats["Metric"]=="depth"), stat_type].values[0],
                    "coverage": self.expstats.loc[(self.expstats["Experiment"]==assay) & (self.expstats["Metric"]=="coverage"), stat_type].values[0],
                    "read_length": self.expstats.loc[(self.expstats["Experiment"]==assay) & (self.expstats["Metric"]=="read_length"), stat_type].values[0]
                }

            if len(md.shape) == 2:
                for i, (assay, alias) in enumerate(self.aliases["experiment_aliases"].items()):
                    # assert i+1 == int(alias.replace("M",""))
                    
                    if torch.all(md[:, i] == missing_value):
                        md[0, i] = stat_lookup[assay]["depth"]
                        md[1, i] = stat_lookup[assay]["coverage"] 
                        md[2, i] = stat_lookup[assay]["read_length"]
                        md[3, i] = 1

            else:
                for i, (assay, alias) in enumerate(self.aliases["experiment_aliases"].items()):
                    # assert i+1 == int(alias.replace("M",""))
                    
                    for b in range(md.shape[0]):
                        if torch.all(md[b, :, i] == missing_value):
                            md[b, 0, i] = stat_lookup[assay]["depth"]
                            md[b, 1, i] = stat_lookup[assay]["coverage"]
                            md[b, 2, i] = stat_lookup[assay]["read_length"]
                            md[b, 3, i] = 1

            return md
         
    def make_bios_tensor(self, loaded_data, loaded_metadata, missing_value=-1):
        dtensor = []
        mdtensor = []
        availability = []

        L = len(loaded_data[list(loaded_data.keys())[0]])
        i = 0
        for assay, alias in self.aliases["experiment_aliases"].items():
            
            if assay in loaded_data.keys():
                # print(i, assay, alias, loaded_data.keys())
                dtensor.append(loaded_data[assay])
                availability.append(1)

                if "single" in loaded_metadata[assay]['run_type'][list(loaded_metadata[assay]['run_type'].keys())[0]]:
                    runt = 0
                elif "pair" in loaded_metadata[assay]['run_type'][list(loaded_metadata[assay]['run_type'].keys())[0]]:
                    runt = 1

                readl = loaded_metadata[assay]['read_length'][list(loaded_metadata[assay]['read_length'].keys())[0]]

                mdtensor.append([
                    np.log2(loaded_metadata[assay]['depth']), loaded_metadata[assay]['coverage'],
                    readl, runt])

            else:
                dtensor.append([missing_value for _ in range(L)])
                availability.append(0)
                mdtensor.append([missing_value, missing_value, missing_value, missing_value])

            i += 1
        
        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        mdtensor = torch.tensor(np.array(mdtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))
        return dtensor, mdtensor, availability

    def make_region_tensor(self, loaded_data, loaded_metadata):
        data, metadata, availability = [], [], []
        for i in range(len(loaded_data)):
            d, md, avl = self.make_bios_tensor(loaded_data[i], loaded_metadata[i])
            data.append(d)
            metadata.append(md)
            availability.append(avl)
        
        data, metadata, availability = torch.stack(data), torch.stack(metadata), torch.stack(availability)
        return data, metadata, availability

    def initialize_EED(self,
        m, context_length, bios_batchsize, loci_batchsize, loci_gen="chr19", 
        bios_min_exp_avail_threshold=3, check_completeness=True, shuffle_bios=True, 
        includes=[
            'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac', 
            'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3', 
            'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac', 
            'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac'], 
        excludes=[], 
        merge_ct=True, must_have_chr_access=False,
        eic=False, DSF_list=[1, 2, 4]):

        self.eic = eic
        self.merge_ct = merge_ct
        self.must_have_chr_access = must_have_chr_access
        self.set_alias()
        
        self.coords(mode="train")

        if loci_gen == "ccre":
            print("generating cCRE loci")
            self.generate_ccre_loci(m, context_length)
        
        elif loci_gen == "random":
            print("generating random loci")
            self.generate_random_loci(m, context_length)
        
        elif loci_gen == "debug":
            self.generate_full_chr_loci(context_length, chrs=["chr19"])
            self.m_regions = self.m_regions[(len(self.m_regions) - m) // 2 : (len(self.m_regions) + m) // 2]
        
        elif loci_gen == "gw":
            self.generate_full_chr_loci(context_length, chrs="gw")
        else:
            self.generate_full_chr_loci(context_length, chrs="loci_gen")

        print(f"num loci: {len(self.m_regions)}")
        
        if os.path.exists(self.navigation_path) == False:
            print("generating navigation file")
            self.navigate_bios_exps()
            
        with open(self.navigation_path, 'r') as navfile:
            self.navigation  = json.load(navfile)

        if self.merge_ct and eic==False:
            if os.path.exists(self.merged_navigation_path) == False:
                print("generating merged celltypes navigation file")
                self.navigate_merge_celltypes()

            with open(self.merged_navigation_path, 'r') as navfile:
                self.navigation  = json.load(navfile)

        if self.merge_ct:
            self.merged_train_val_test_split()
        else:
            self.train_val_test_split()

        if eic:
            self.init_eic(target_split="train")
        else:
            self.filter_navigation(exclude=excludes, include=includes)

        # filter biosamples
        for bios in list(self.navigation.keys()):
            if eic==False and len(self.navigation[bios]) < bios_min_exp_avail_threshold:
                del self.navigation[bios]

            elif self.split_dict[bios] != "train":
                del self.navigation[bios]

            elif check_completeness and eic==False: 
                if len(self.is_bios_complete(bios))>0:
                    del self.navigation[bios]
        
        mean_available_exps = []
        for bios in list(self.navigation.keys()):
            mean_available_exps.append(len(self.navigation[bios]))

        print(f"mean available exps: {np.mean(mean_available_exps)}")

        if shuffle_bios:
            keys = list(self.navigation.keys())
            random.shuffle(keys)
            self.navigation = {key: self.navigation[key] for key in keys}

        # print num_bios per assay
        unique_exp = {exp:0 for exp in self.df1.columns if exp not in ["Unnamed: 0", "Accession"]}
        for bios in self.navigation.keys():
            for exp in self.navigation[bios].keys():
                unique_exp[exp] += 1
        
        unique_exp = {k: v for k, v in sorted(unique_exp.items(), key=lambda item: item[1], reverse=True)}
        for exp, count in unique_exp.items():
            print(f"{exp} in present in {count} biosamples")

        self.signal_dim = len(self.aliases["experiment_aliases"].keys()) # sum(1 for value in unique_exp.values() if value > 0)
        # print(len(self.aliases["experiment_aliases"]))

        for k in list(self.aliases["experiment_aliases"].keys()):
            if k not in includes or k in excludes:
                del self.aliases["experiment_aliases"][k]

        print(f"signal_dim: {self.signal_dim}")
        self.num_regions = len(self.m_regions)
        self.num_bios = len(self.navigation)
        print(f"num_bios: {self.num_bios}")

        self.bios_batchsize = bios_batchsize
        self.loci_batchsize = loci_batchsize

        self.num_batches = math.ceil(self.num_bios / self.bios_batchsize)

        self.loci = {}
        for i in range(len(self.m_regions)):
            if self.m_regions[i][0] not in self.loci.keys():
                self.loci[self.m_regions[i][0]] = []

            self.loci[self.m_regions[i][0]].append(self.m_regions[i])
        self.dsf_list = DSF_list

    def new_epoch(self, shuffle_chr=True):
        self.chr_pointer = 0 
        self.bios_pointer = 0
        self.dsf_pointer = 0
        self.chr_loci_pointer = 0

        if shuffle_chr:
            keys = list(self.loci.keys())
            random.shuffle(keys)
            shuffled_loci = {key: self.loci[key] for key in keys}
            self.loci = shuffled_loci

        # tracemalloc.start()  # Start tracking memory allocations
        batch_bios_list = list(self.navigation.keys())[self.bios_pointer : self.bios_pointer+self.bios_batchsize]
        self.loaded_data = []
        self.loaded_metadata = []

        for bios in batch_bios_list:
            # print(f"loading {bios}")
        
            d, md = self.load_bios(bios, [list(self.loci.keys())[self.chr_pointer]], self.dsf_list[self.dsf_pointer])
            self.loaded_data.append(d)
            self.loaded_metadata.append(md)

        self.Y_loaded_data, self.Y_loaded_metadata = [], []
        self.Y_loaded_data = self.loaded_data.copy()
        self.Y_loaded_metadata = self.loaded_metadata.copy()

        self.Y_loaded_pval = []
        for bios in batch_bios_list:
            pval_d = self.load_bios_BW(bios, [list(self.loci.keys())[self.chr_pointer]], self.dsf_list[self.dsf_pointer])
            self.Y_loaded_pval.append(pval_d)

    def update_batch_pointers(self):
        if self.chr_loci_pointer + self.loci_batchsize >= len(self.loci[list(self.loci.keys())[self.chr_pointer]]):
            self.chr_loci_pointer = 0

            if self.dsf_pointer + 1 >= len(self.dsf_list):
                self.dsf_pointer = 0

                if self.bios_pointer + self.bios_batchsize >= self.num_bios:
                    self.bios_pointer = 0 

                    if self.chr_pointer + 1 >= len(self.loci.keys()):
                        self.chr_pointer = 0
                        return True

                    else:
                        self.chr_pointer += 1
                        
                else:
                    self.bios_pointer += self.bios_batchsize
                    
            else: 
                self.dsf_pointer += 1
            
            # print("loading new count data")
            batch_bios_list = list(self.navigation.keys())[self.bios_pointer : self.bios_pointer+self.bios_batchsize]

            self.loaded_data = []
            self.loaded_metadata = []
            
            for bios in batch_bios_list:
                d, md = self.load_bios(bios, [list(self.loci.keys())[self.chr_pointer]], self.dsf_list[self.dsf_pointer])
                self.loaded_data.append(d)
                self.loaded_metadata.append(md)

            if self.dsf_pointer == 0:
                self.Y_loaded_data, self.Y_loaded_metadata = [], []
                self.Y_loaded_data = self.loaded_data.copy()
                self.Y_loaded_metadata = self.loaded_metadata.copy()


                self.Y_loaded_pval = []
                for bios in batch_bios_list:
                    self.Y_loaded_pval.append(
                        self.load_bios_BW(bios, [list(self.loci.keys())[self.chr_pointer]], self.dsf_list[self.dsf_pointer]))

        else:
            self.chr_loci_pointer += self.loci_batchsize
        
        return False

    def get_batch(self, side="x", y_prompt=False, pval=False, dna_seq=False):
        """
        select subset of loci in working chr
        chr_loci = [locus for locus in self.loci if locus[0] == working_chr]
        
        for chr in loci.chrs:
            for batch in biosamples:
                for dsf in dsf_list:
                    load all bios_chr_dsf

                    for locus in chr_loci:
                        return bios_chr_dsf[locus]
        """
        
        current_chr = list(self.loci.keys())[self.chr_pointer]
        batch_loci_list = self.loci[current_chr][self.chr_loci_pointer : self.chr_loci_pointer+self.loci_batchsize]

        batch_data = []
        batch_metadata = []
        batch_availability = []
        
        if pval and side == "y":
            batch_pval = []

        for locus in batch_loci_list:
            if dna_seq:
                one_hot_sequence = dna_to_onehot(get_DNA_sequence(locus[0], locus[1], locus[2]))

            loc_d = []

            if side == "x":
                for data in self.loaded_data:
                    loc_d.append(self.select_region_from_loaded_data(data, locus))
                d, md, avl = self.make_region_tensor(loc_d, self.loaded_metadata)
                del loc_d

            elif side == "y":
                for data in self.Y_loaded_data:
                    loc_d.append(self.select_region_from_loaded_data(data, locus))
                d, md, avl = self.make_region_tensor(loc_d, self.Y_loaded_metadata)
                del loc_d

                if y_prompt:
                    # print(f"filling in y prompt for {locus}")
                    md = self.fill_in_y_prompt(md, sample=True)

                if pval:
                    loc_p = []
                    for pp in self.Y_loaded_pval:
                        loc_p.append(self.select_region_from_loaded_data(pp, locus))
                    p, avl_p = self.make_region_tensor_BW(loc_p)
                    del loc_p
                     
                    assert (avl_p == avl).all(), "avl_p and avl do not match"
                    batch_pval.append(p)

            batch_data.append(d)
            batch_metadata.append(md)
            batch_availability.append(avl)
        
        if dna_seq:
            if pval and side == "y":
                batch_data, batch_metadata = torch.concat(batch_data), torch.concat(batch_metadata) 
                batch_availability, batch_pval = torch.concat(batch_availability), torch.concat(batch_pval)
                return batch_data, batch_metadata, batch_availability, batch_pval, one_hot_sequence

            else:
                batch_data, batch_metadata, batch_availability = torch.concat(batch_data), torch.concat(batch_metadata), torch.concat(batch_availability)
                return batch_data, batch_metadata, batch_availability, one_hot_sequence

        else:
            if pval and side == "y":
                batch_data, batch_metadata = torch.concat(batch_data), torch.concat(batch_metadata) 
                batch_availability, batch_pval = torch.concat(batch_availability), torch.concat(batch_pval)
                return batch_data, batch_metadata, batch_availability, batch_pval

            else:
                batch_data, batch_metadata, batch_availability = torch.concat(batch_data), torch.concat(batch_metadata), torch.concat(batch_availability)
                return batch_data, batch_metadata, batch_availability
        
    def init_eval(
        self, context_length, bios_min_exp_avail_threshold=1, 
        check_completeness=False, split="test",
        includes=[
            'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac', 
            'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3', 
            'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac', 
            'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac', 
            'RNA-seq'], 
        excludes=[], 
        eic=False, 
        merge_ct=True, 
        must_have_chr_access=False):

        self.set_alias()
        self.merge_ct = merge_ct
        self.must_have_chr_access = must_have_chr_access
        self.eic = eic
        self.coords(mode="eval")
        
        if os.path.exists(self.navigation_path) == False:
            self.navigate_bios_exps()
            
        with open(self.navigation_path, 'r') as navfile:
            self.navigation  = json.load(navfile)

        if self.merge_ct and eic==False:
            if os.path.exists(self.merged_navigation_path) == False:
                print("generating merged celltypes navigation file")
                self.navigate_merge_celltypes()

            with open(self.merged_navigation_path, 'r') as navfile:
                self.navigation  = json.load(navfile)
            
            self.merged_train_val_test_split()

        else:
            self.train_val_test_split()
        
        if eic:
            self.init_eic(target_split=split)
        else:
            print("filtering eval navigation")
            self.filter_navigation(exclude=excludes, include=includes)

        # unique_exp = {exp:0 for exp in self.df1.columns if exp not in ["Unnamed: 0", "Accession"]}
        # for bios in self.navigation.keys():
        #     for exp in self.navigation[bios].keys():
        #         unique_exp[exp] += 1

        for k in list(self.aliases["experiment_aliases"].keys()):
            if k not in includes or k in excludes:
                del self.aliases["experiment_aliases"][k]
        
        # print(len(self.aliases["experiment_aliases"]))

        # exit()

        # filter biosamples
        for bios in list(self.navigation.keys()):
            if split == "test" and self.has_rnaseq(bios):
                continue

            elif eic==False and len(self.navigation[bios]) < bios_min_exp_avail_threshold:
                del self.navigation[bios]

            elif split != "all" and self.split_dict[bios] != split:
                del self.navigation[bios]

            elif eic==False and check_completeness:
                if len(self.is_bios_complete(bios))>0:
                    del self.navigation[bios]
        
        self.num_bios = len(self.navigation)
        self.signal_dim = len(self.aliases["experiment_aliases"].keys())
        print(f"eval signal_dim: {self.signal_dim}")
        print(f"num eval bios: {self.num_bios}")

        self.test_bios = []
        for b, s in self.split_dict.items():
            if s == split:
                if b in list(self.navigation.keys()):
                    self.test_bios.append(b)
    
    def has_rnaseq(self, bios_name):
        if self.merge_ct:
            if "RNA-seq" in self.navigation[bios_name].keys():
                return True
            else:
                return False

        else:
            if os.path.exists(os.path.join(self.base_path, bios_name, "RNA-seq")):
                return True
            else:
                return False

    def load_rna_seq_data(self, bios_name, gene_coord):
        if self.merge_ct:
            directory = os.path.dirname(self.navigation[bios_name]["RNA-seq"][0])
        else:
            directory = os.path.join(self.base_path, bios_name, "RNA-seq/")
            
        tsv_files = glob.glob(os.path.join(directory, '*.tsv'))

        file = os.path.join(directory, tsv_files[0])
        trn_data = pd.read_csv(file, sep="\t")

        for j in range(len(trn_data)):
            trn_data.at[j, "gene_id"] = trn_data["gene_id"][j].split(".")[0]
        
        for i in range(len(gene_coord)):
            gene_coord.at[i, "gene_id"] = gene_coord["gene_id"][i].split(".")[0]

        mapped_trn_data = []
        for i in range(len(gene_coord)):
            geneID = gene_coord["gene_id"][i]
            subset = trn_data.loc[trn_data["gene_id"] == geneID, :].reset_index(drop=True)

            if len(subset) > 0:
                mapped_trn_data.append([
                    gene_coord["chr"][i], gene_coord["start"][i], gene_coord["end"][i], gene_coord["strand"][i], geneID, subset["length"][0], subset["TPM"][0], subset["FPKM"][0]
                ])

        mapped_trn_data = pd.DataFrame(mapped_trn_data, columns=["chr", "start", "end", "strand", "geneID", "length", "TPM", "FPKM"])
        return mapped_trn_data

if __name__ == "__main__": 
    solar_data_path = "/project/compbio-lab/encode_data/"
    if sys.argv[1] == "check":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        print(eed.is_bios_complete(sys.argv[2]))

    elif sys.argv[1] == "get_refseq":
        if not os.path.exists(solar_data_path + "/hg38.fa"):
            refseq_url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
            savedir = solar_data_path + "/hg38.fa.gz"
            download_save(refseq_url, savedir)

            with gzip.open(savedir, 'rb') as f_in:
                with open(savedir.replace(".gz", ""), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        t0 = datetime.datetime.now()
        seq = get_DNA_sequence("chr1", (248387328//2)-80000, (248387328//2))
        seq = dna_to_onehot(seq)
        print(seq)
        print(seq.sum().sum())
        t1 = datetime.datetime.now()
        print(f"retrieval took {t1-t0}")

    elif sys.argv[1] == "fix":
        # Initialize data handlers
        d = GET_DATA()
        d.load_metadata(metadata_file_path=solar_data_path)
        dataset = ExtendedEncodeDataHandler(solar_data_path)

        # Check if each biosample from DF1 exists in solar_data_path
        missing_biosamples = []
        for biosample in d.DF1['Accession']:
            biosample_path = os.path.join(solar_data_path, biosample)
            if not os.path.exists(biosample_path):
                missing_biosamples.append(biosample)
                print(f"Missing biosample: {biosample}")

        print(f"\nTotal missing biosamples: {len(missing_biosamples)}")
        print(f"Total biosamples in DF1: {len(d.DF1['Accession'])}")
        print(f"Percentage complete: {100 * (1 - len(missing_biosamples)/len(d.DF1['Accession'])):.2f}%")

        dataset.mp_fix_DS(n_p=2)
    
    elif sys.argv[1] == "checkup":
        d = GET_DATA()
        d.load_metadata(metadata_file_path=solar_data_path)
        dataset = ExtendedEncodeDataHandler(solar_data_path)

        # Check if each biosample from DF1 exists in solar_data_path
        missing_biosamples = []
        for biosample in d.DF1['Accession']:
            biosample_path = os.path.join(solar_data_path, biosample)
            if not os.path.exists(biosample_path):
                missing_biosamples.append(biosample)
                print(f"Missing biosample: {biosample}")

        print(f"\nTotal missing biosamples: {len(missing_biosamples)}")
        print(f"Total biosamples in DF1: {len(d.DF1['Accession'])}")
        print(f"Percentage complete: {100 * (1 - len(missing_biosamples)/len(d.DF1['Accession'])):.2f}%")

        # Check for incomplete experiments in existing biosamples
        print("\nChecking for incomplete experiments in existing biosamples...")
        incomplete_count = 0
        for biosample in d.DF1['Accession']:
            biosample_path = os.path.join(solar_data_path, biosample)
            if os.path.exists(biosample_path):
                # Get expected experiments for this biosample from DF1
                expected_exps = d.DF1.loc[d.DF1['Accession'] == biosample].dropna(axis=1).columns.tolist()
                expected_exps.remove('Accession')  # Remove the Accession column
                
                # Check each experiment using is_exp_complete
                incomplete_exps = []
                for exp in expected_exps:
                    if not dataset.is_exp_complete(biosample, exp):
                        incomplete_exps.append(exp)
                
                if incomplete_exps:
                    incomplete_count += 1
                    print(f"\nBiosample {biosample} has incomplete experiments:")
                    for exp in incomplete_exps:
                        print(f"  - {exp}")

        print(f"\nTotal biosamples with incomplete experiments: {incomplete_count}")
        print(f"Percentage of existing biosamples with all experiments complete: {100 * ((len(d.DF1['Accession'])-len(missing_biosamples))-incomplete_count)/(len(d.DF1['Accession'])-len(missing_biosamples)):.2f}%")

    elif sys.argv[1] == "test":
        dataset = ExtendedEncodeDataHandler(solar_data_path)
        dataset.initialize_EED(
            m=100, context_length=1600*25, 
            bios_batchsize=1, loci_batchsize=1, loci_gen="random",
            bios_min_exp_avail_threshold=10, check_completeness=True, eic=False)

        exit()
        from scipy.stats import spearmanr
        dataset = ExtendedEncodeDataHandler(solar_data_path)
        dataset.initialize_EED(
            m=10, context_length=1600*25, 
            bios_batchsize=1, loci_batchsize=1, loci_gen="random",
            bios_min_exp_avail_threshold=10, check_completeness=True, eic=False)

        for epoch in range(10):
            dataset.new_epoch()
            print("new epoch")
            next_epoch = False

            while (next_epoch==False):

                _X_batch, _mX_batch, _avX_batch, _dnaseq_batch = dataset.get_batch(side="x", dna_seq=True)
                _Y_batch, _mY_batch, _avY_batch, _pval_batch = dataset.get_batch(side="y", pval=True)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                else:
                    print(_X_batch.shape, _mX_batch.shape, _avX_batch.shape, _dnaseq_batch.shape)
                    print(_Y_batch.shape, _mY_batch.shape, _avY_batch.shape, _pval_batch.shape)
                    # for e in range(len(_avY_batch[0])):
                    #     if _avY_batch[0, e] != 0:
                    #         correlation, p_value = spearmanr(_X_batch[0,:,e], _pval_batch[0,:,e],)
                    #         print(f"Spearman correlation: {correlation}")

                    # print(_Y_batch.float().mean(axis=1), _pval_batch.float().mean(axis=1))
                    print("\n\n")
                
                del _X_batch, _mX_batch, _avX_batch, _Y_batch, _mY_batch, _avY_batch
                gc.collect()
                torch.cuda.empty_cache()
                    
                next_epoch = dataset.update_batch_pointers()

    elif sys.argv[1] == "test_solar":
        dataset = ExtendedEncodeDataHandler(solar_data_path)
        dataset.initialize_EED(
            m=10, context_length=200*25, 
            bios_batchsize=50, loci_batchsize=1, ccre=False, 
            bios_min_exp_avail_threshold=4, check_completeness=True)
        
        avail = {}
        for k, v in dataset.navigation.items():
            avail[k] = len(v)
        
        print(avail)
        print(len(avail))
    
    elif sys.argv[1] == "synthetic":
        # Initialize the SyntheticData class with updated parameters
        synthetic_data = SyntheticData(n=0.1485, p=0.0203, num_features=47, sequence_length=1600)

        # Generate and visualize the base sequence
        base_sequence = synthetic_data.generate_base_sequence()

        plt.figure(figsize=(12, 4))
        plt.plot(base_sequence, label='Base Sequence')
        plt.title('Base Sequence')
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        # Apply transformations to derive 47 distinct sequences
        transformed_sequences = synthetic_data.apply_transformations()

        # Apply smoothing to introduce sequence dependence
        smoothed_sequences = synthetic_data.apply_smoothing(transformed_sequences)
        smoothed_sequences = np.array(smoothed_sequences)

        syn_metadata = synthetic_data.synth_metadata(transformed_sequences)
        syn_metadata = np.array(syn_metadata)

        num_labels = synthetic_data.num_features
        n_cols = math.floor(math.sqrt(num_labels))
        n_rows = math.ceil(num_labels / n_cols)

        # Visualize the smoothed sequences
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        fig.suptitle('Smoothed Sequences', fontsize=16)
        for i in range(num_labels):
            row, col = divmod(i, n_cols)
            ax = axs[row, col]
            seq = smoothed_sequences[i]
            md = syn_metadata[i]
            ax.plot(seq, label=f'F{i+1}: {md[0]:.1f}-{md[1]:.1f}-{md[2]:.1f}-{md[3]:.1f}')
            # ax.legend(fontsize=5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    elif sys.argv[1] == "eic":
        dataset = ExtendedEncodeDataHandler(solar_data_path)
        context_length=3200
        resolution = 25

        dataset.initialize_EED(
            m=100, context_length=context_length*resolution, 
            bios_batchsize=50, loci_batchsize=1, loci_gen="random",
            bios_min_exp_avail_threshold=3, check_completeness=True, eic=True)

    elif sys.argv[1] == "prompt":
        bioses = [b for b in os.listdir(solar_data_path) if os.path.isdir(os.path.join(solar_data_path, b)) and b not in ["chromatin_state_annotations", "activity_data"]]
        exps = {}
        for bios_name in bioses:
            for exp in os.listdir(os.path.join(solar_data_path, bios_name)):
                exp_path = os.path.join(solar_data_path, bios_name, exp)
                if os.path.isdir(exp_path):
                    
                    if exp not in exps.keys():
                        exps[exp] = []

                    try:
                        jsn1 = os.path.join(solar_data_path, bios_name, exp, "signal_DSF1_res25", "metadata.json")
                        with open(jsn1, 'r') as jsnfile:
                            md1 = json.load(jsnfile)

                        jsn2 = os.path.join(solar_data_path, bios_name, exp, "file_metadata.json")
                        with open(jsn2, 'r') as jsnfile:
                            md2 = json.load(jsnfile)

                        md = {
                            "depth":md1["depth"], "coverage":md1["coverage"], 
                            "read_length":list(md2["read_length"].values())[0], 
                            "run_type":list(md2["run_type"].values())[0] 
                        }

                        exps[exp].append(md)

                    except:
                        pass

        exps2 = {}
        for exp in exps.keys():
            if exp not in exps2.keys():
                exps2[exp] = {}
                print(exp)

            for i in range(len(exps[exp])):
                for md in exps[exp][i].keys():
                    if md not in exps2[exp].keys():
                        exps2[exp][md] = []

                    exps2[exp][md].append(exps[exp][i][md])

        raw_exp_md = pd.DataFrame(exps2)

        print(list(raw_exp_md.columns))
        raw_exp_md.to_csv(f"{solar_data_path}/RawExpMetaData.csv")
        print(raw_exp_md)
        exit()

        # Calculate basic statistics
        statistics = {}
        for exp, metrics in exps2.items():
            statistics[exp] = {}
            for metric, values in metrics.items():
                if metric == "run_type":
                    run_type_counts = pd.Series(values).value_counts().to_dict()
                    statistics[exp][metric] = run_type_counts

                else:
                    if metric == "depth":
                        values = np.log2(np.array(values, dtype=np.float64))
                    else:
                        values = np.array(values, dtype=np.float64)
                    statistics[exp][metric] = {
                        "mean": np.nanmean(values),
                        "median": np.nanmedian(values),
                        "std_dev": np.nanstd(values),
                        "min": np.nanmin(values),
                        "max": np.nanmax(values)
                    }

        # Create summary report
        summary_rows = []
        for exp, metrics in statistics.items():
            for metric, stats in metrics.items():
                if metric == "run_type":
                    for run_type, count in stats.items():
                        summary_rows.append([exp, metric, run_type, count, np.nan, np.nan, np.nan, np.nan])
                else:
                    summary_rows.append([exp, metric, np.nan, np.nan, stats["mean"], stats["median"], stats["std_dev"], stats["min"], stats["max"]])

        summary_report = pd.DataFrame(summary_rows, columns=['Experiment', 'Metric', 'Run Type', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'])
        summary_report.to_csv(f"{solar_data_path}/ExpStats.csv")

    elif sys.argv[1] == "check_pval":
        # from scipy.stats import spearmanr
        # dataset = ExtendedEncodeDataHandler(solar_data_path)
        # for bs in os.listdir(solar_data_path):
        #     if os.path.isdir(os.path.join(solar_data_path, bs)):
        #         exps = [x for x in os.listdir(os.path.join(solar_data_path, bs)) if os.path.isdir(os.path.join(solar_data_path, bs, x))]
        #         for exp in exps:
        #             if "signal_BW_res25" in os.listdir(os.path.join(solar_data_path, bs, exp)):
        #                 if "signal_DSF1_res25" in os.listdir(os.path.join(solar_data_path, bs, exp)):
        #                     count_data = dataset.load_npz(os.path.join(solar_data_path, bs, exp, "signal_DSF1_res25", "chr21.npz"))
        #                     pval =  dataset.load_npz(os.path.join(solar_data_path, bs, exp, "signal_BW_res25", "chr21.npz"))
                        
        #                     count_data = count_data[list(count_data.keys())[0]]
        #                     pval = pval[list(pval.keys())[0]]

        #                     correlation, p_value = spearmanr(count_data[:len(pval)], pval[:len(pval)])

        #                     print(f"{bs}-{exp} Spearman correlation: {correlation}")

        # exit()
        proc = []
        chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        for bs in os.listdir(solar_data_path):
            if os.path.isdir(os.path.join(solar_data_path, bs)):
                exps = [x for x in os.listdir(os.path.join(solar_data_path, bs)) if os.path.isdir(os.path.join(solar_data_path, bs, x))]
                for exp in exps:
                    full = 1
                    if "signal_BW_res25" in os.listdir(os.path.join(solar_data_path, bs, exp)):
                        for c in chrs:
                            if c+".npz" not in os.listdir(os.path.join(solar_data_path, bs, exp, "signal_BW_res25")):
                                full = 0
                    else:
                        full = 0
                    proc.append(full)
        
        print(f"frac exps with bigwig = {float(sum(proc))/len(proc)}")

    elif sys.argv[1] == "get_pval":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        def process_pair(pair):
            bios_name, exp = pair
            eed.get_signal_pval_bigwig(bios_name, exp)
        
        todo = []
        for bs in os.listdir(solar_data_path):
            # if bs[0] not in ["B", "V", "T"]:
            #     continue
            if os.path.isdir(os.path.join(solar_data_path, bs)):

                exps = [x for x in os.listdir(os.path.join(solar_data_path, bs)) if os.path.isdir(os.path.join(solar_data_path, bs, x))]
                for exp in exps:
                    todo.append([bs, exp])

        random.shuffle(todo)
        # multiprocess all bios_name, exp pairs in todo for function eed.get_signal_pval_bigwig(bios_name, exp)
        with mp.Pool(processes=2) as pool:
            pool.map(process_pair, todo)

    elif sys.argv[1] == "get_peaks":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        def process_pair(pair):
            bios_name, exp = pair
            eed.get_peaks_bigbed(bios_name, exp)
        
        todo = []
        for bs in os.listdir(solar_data_path):
            # if bs[0] not in ["B", "V", "T"]:
            #     continue
            if os.path.isdir(os.path.join(solar_data_path, bs)):

                exps = [x for x in os.listdir(os.path.join(solar_data_path, bs)) if os.path.isdir(os.path.join(solar_data_path, bs, x))]
                for exp in exps:
                    todo.append([bs, exp])

        random.shuffle(todo)
        # multiprocess all bios_name, exp pairs in todo for function eed.get_signal_pval_bigwig(bios_name, exp)
        with mp.Pool(processes=5) as pool:
            pool.map(process_pair, todo)

    elif sys.argv[1] == "download_bios":
        d = GET_DATA()
        d.load_metadata(metadata_file_path=solar_data_path)
        print(f"downloading biosample {sys.argv[2]}")
        exps =d.get_biosample(
            bios=sys.argv[2],
            df1_ind=0,
            metadata_file_path=solar_data_path,
            assembly="GRCh38"
        )
        for exp in exps:
            print(f"downloading {sys.argv[2]}-{exp}")
            single_download(exp)

    elif sys.argv[1] == "CS_annotations":
        metadata = get_encode_chromatin_state_annotation_metadata(metadata_file_path=solar_data_path)
        get_chromatin_state_annotation_data(metadata_file_path=solar_data_path)
    
    elif sys.argv[1] == "load_CS_annotations":
        # Find the parsed directory in chromatin state annotations
        cs_dir = os.path.join(solar_data_path, "chromatin_state_annotations", sys.argv[2])
        parsed_dirs = [d for d in os.listdir(cs_dir) if d.startswith('parsed200_')]
        print(f"Found parsed directories: {parsed_dirs}")
        parsed_dir = parsed_dirs[0]  # Use first one by default
        # parsed_dir = next(d for d in os.listdir(cs_dir) if d.startswith('parsed_'))
        # print(f"Found parsed directory: {parsed_dir}")
        parsed_path = os.path.join(cs_dir, parsed_dir)
        cs = load_region_chromatin_states(parsed_path, "chr1", 1000000//8, (1000000+30000)//8, resolution=200)
        print(cs)

    elif sys.argv[1] == "download_activity_data":
        download_activity_data(metadata_file_path=solar_data_path)

    elif sys.argv[1] == "update_eic_bw":
        """
        for eic data, rm */signal_bw_res25
        look up the corresponding ct-assay from the OG EIC bigwigs
        bin the values and save them in npz format at */signal_bw_res25
        """

        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        csv_file_path = "/project/compbio-lab/encode_data/EIC_experiments.csv"
        experiments_df = pd.read_csv(csv_file_path, header=0)
        # header: data_type,cell_type_id,mark_id,cell_type,mark/assay,filename,experiment

        for i in range(len(experiments_df)):
            datatype = experiments_df.iloc[i, 0]
            cell_type_id = experiments_df.iloc[i, 1]
            mark_id = experiments_df.iloc[i, 2]
            ct_name = experiments_df.iloc[i, 3]
            if "H1" in ct_name:
                ct_name = "H1"
            assay_name = experiments_df.iloc[i, 4]
            filename = f"{experiments_df.iloc[i, 5]}"
            bw_filepath = f"/project/compbio-lab/EIC/{datatype}/{filename}.bigwig"

            if datatype == "training_data":
                encode_data_path = f"/project/compbio-lab/encode_data/T_{ct_name}/{assay_name}/signal_BW_res25/"
            elif datatype == "validation_data":
                encode_data_path =  f"/project/compbio-lab/encode_data/V_{ct_name}/{assay_name}/signal_BW_res25/"
            elif datatype == "blind_data":
                encode_data_path =  f"/project/compbio-lab/encode_data/B_{ct_name}/{assay_name}/signal_BW_res25/"

            for chr in main_chrs:
                try:
                    binned_chr = get_binned_vals(bw_filepath, chr, resolution=25)
                    if os.path.exists(f"{encode_data_path}/{chr}.npz"):
                        os.system(f"rm {encode_data_path}/{chr}.npz")
                        np.savez_compressed(f"{encode_data_path}/{chr}.npz", np.array(binned_chr))
                        print(f'replaced/updated "{encode_data_path}/{chr}.npz')
                except:
                    print(f'FAILED @ "{encode_data_path}/{chr}.npz')

    else:
        d = GET_DATA()
        d.search_ENCODE(metadata_file_path=solar_data_path)
        d.filter_biosamples(metadata_file_path=solar_data_path)
        d.load_metadata(metadata_file_path=solar_data_path)
        d.get_all(metadata_file_path=solar_data_path)

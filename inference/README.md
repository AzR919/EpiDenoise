# CANDI Inference

Command line scrip to run inference using CANDI.
Impute or denoise assays

Examples:

```
./main.py --data_path <path/to/data> --model_path <path/to/model>
--assays H3F3A H3K23me2 --chromosomes 19 20 21
```

Available assays:

    - ATAC-seq
    - DNase-seq
    - H2AFZ
    - H2AK5ac
    - H2AK9ac
    - H2BK120ac
    - H2BK12ac
    - H2BK15ac
    - H2BK20ac
    - H2BK5ac
    - H3F3A
    - H3K14ac
    - H3K18ac
    - H3K23ac
    - H3K23me2
    - H3K27ac
    - H3K27me3
    - H3K36me3
    - H3K4ac
    - H3K4me1
    - H3K4me2
    - H3K4me3
    - H3K56ac
    - H3K79me1
    - H3K79me2
    - H3K9ac
    - H3K9me1
    - H3K9me2
    - H3K9me3
    - H3T11ph
    - H4K12ac
    - H4K20me1
    - H4K5ac
    - H4K8ac
    - H4K91ac
    - chipseq-control
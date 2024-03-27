# KnotFold

This is the official repository for the KnotFold, developed by [Genentech](https://www.gene.com/)'s Computational Structural Biology group.

## Testing KnotFold

The KnotFold functionality can be tested using this [Colab Notebook](https://colab.research.google.com/github/genentech/knotfold/blob/main/notebooks/KnotFold.ipynb)

## Using KnotFold locally

### 1. Install AlphaFold
To use KnotFold locally, first follow instructions to [install AlphaFold](https://github.com/google-deepmind/alphafold?tab=readme-ov-file#installation-and-running-your-first-prediction).

### 2. Clone KnotFold repo
Then, clone this repo:

```
git clone git@github.com:Genentech/knotfold.git
```

### 3. Generate baseline AlphaFold prediction (features.pkl)
First generate an AlphaFold prediction of the disulfide-rich peptide of interest: 

```
python run_alphafold.py \
--fasta_paths <path to fasta file containing sequence> \
--output_dir <output directory> \
--model_preset monomer \
... follow guidance from AlphaFold 
```

The important part of running this baseline AlphaFold prediction is to generate the `features.pkl` file that will be used for the artificial coevolution prompt engineering.

### 4. Run KnotFold

```
python cysteine_connectivity.py \
--input_dir <path to input directory containing baseline features.pkl> \
--output_dir <path to output dir> \
--pairs <desired pairs (e.g. 14,23,56)> \
--num_preds_per_model <number of predictions per model (default=5)> \
--aa_option <option for replacement of amino acids ['all', 'hydrophobic', 'small'] (default='all')> \
--mutation_option <option for how to mutate MSA ['breakN', 'Nalignments', 'breakN+1'] (default='breakN+1')> \
--data_dir <path to AlphaFold data resources> \
(--amber_relax \)
(--templated \)
(--gpu_relax \)
```

### 5. Force cysteines with PyRosetta (optional)

If you find that the output prediction is just shy of forming the correct disulfides, you can use PyRosetta to form the disulfides and relax the structure with FastRelax. To do so, you will need to install [PyRosetta](https://www.pyrosetta.org/downloads#h.iwt5ktel05jc). It may be most straightforward to do so in a separate environment from the one used for AlphaFold. 

Then run the `force_disulfide.py` script:

```
python force_disulfide.py \
--input_pdb <input pdb file> \
--pairs <pairs (e.g. 14,23,56)> \
--output_dir <output_directory> \
```

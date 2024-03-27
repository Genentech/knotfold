import os
import pickle
import sys
import json
import numpy as np
import argparse
import random
from glob import glob
import re

from alphafold.data import templates
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax

_SUBSAMPLE_MSA_FEATURE_NAMES = [
    'msa',
    'deletion_matrix',
    'msa_mask',
    'msa_row_mask',
    'bert_mask',
    'true_msa',
    'deletion_matrix_int',
    'msa_species_identifiers',
    'cluster_bias_mask'
]
_TRUNCATE_FEATURE_NAMES = ["cluster_bias_mask"]

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def mk_mock_template(lseq: str, multimer: bool) -> dict:

    r"""Generates mock templates that will not influence prediction
    Taken from ColabFold version 62d7558c91a9809712b022faf9d91d8b183c328c
    Parameters
    ----------
    seq: Query sequence
    Returns
    ----------
    Dictionary with blank/empty/meaningless features
    """

    # Define constants
    # there are 37 atom types in alphafold (this number is 37)
    lentype = templates.residue_constants.atom_type_num

    # Since alphafold's model requires a template input
    # We create a blank example w/ zero input, confidence -1
    # templates function returns a numpy array the wrapper np.array is not needed 
    aatypes = np.array(
        templates.residue_constants.sequence_to_onehot(
            "-" * lseq, templates.residue_constants.HHBLITS_AA_TO_ID
        )
    )

    if multimer:
        return {
            "template_all_atom_positions": np.zeros((lseq, lentype, 3))[None], #[None] adds another dimension 
            "template_all_atom_mask": np.zeros((lseq, lentype))[None],
            "template_aatype": aatypes[None],
        }
    else:
        return {
            "template_all_atom_positions": np.zeros((lseq, lentype, 3))[None],
            "template_all_atom_masks": np.zeros((lseq, lentype))[None],
            "template_sequence": [f"none".encode()], # encode - efficient storage of strings
            "template_aatype": aatypes[None],
            "template_confidence_scores": np.full(lseq, -1)[None],
            "template_domain_names": [f"none".encode()],
            "template_release_date": [f"none".encode()],
        }

def partition_break(list_in, n):
    #indices_sets = []
    #for i in range(n):
    #    random.shuffle(list_in)
    #    indices_sets.append(list_in[0:int(len(list_in)/2)])
    random.shuffle(list_in) #shuffle the list of sequences
    indices_sets = [list_in[i::n] for i in range(n)] #for the number of splits (3) get the shuffled sequences
    return indices_sets

def partition_Nalignments(list_in, n):
    list_out = []
    list_in = np.array(list_in)
    length = len(list_in)
    for i in range(n):
        list_out.append(list(list_in+i*length))
    return list_out

def mutation_method(n,pairs,msa,mode='breakN'):
    n_samples = msa.shape[0]
    list_in = list(range(n_samples))
    if mode=='breakN':
        indices_sets = partition_break(list_in,n)
    elif mode=='breakN+1':
        indices_sets = partition_break(list_in,n+1)
        flatpairs=[element for sublist in pairs for element in sublist]
        pairs.append(flatpairs)
        
    elif mode=='Nalignments':
        msa = np.tile(msa,(n,1))
        indices_sets = partition_Nalignments(list_in,n)
        
    return indices_sets, pairs, msa


def back_to_cysteine(msa, msa_index, cysteines,pair):
    for c in pair:
        msa[msa_index, cysteines[c-1]] = 1
    return msa

def replace_cystines(msa, cysteines, mode='hydrophobic'):
    if mode=='all':
        for c in cysteines:
            msa[:, c] = random.choices([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], k=len(msa[:, c])) #ACDEFGHIKLMNPQRSTVWY 
    elif mode=='hydrophobic':
        for c in cysteines:
            msa[:, c] = random.choices([0,1,4,7,9,10,17,18,19], k=len(msa[:, c])) #ACFILMVWY 
            #replace the cystines with a random amino acid (numbers corespond to AF mapping) 
    elif mode=='small':
        for c in cysteines:
            msa[:, c] = random.choices([0,1,9,15,16,17], k=len(msa[:, c])) #ACLSTV    
    else:
        raise Exception('Mode of cystine replacement needs to be properly specified')
    return msa

def main(args):
    """ Takes a previous full run of AlphaFold (with templating) as input """

    input_dir = args.input_dir
    output_dir = args.output_dir
    num_preds_per_model = int(args.num_preds_per_model)
    amber_relax = bool(args.amber_relax)
    templated = bool(args.templated)
    data_dir = args.data_dir
    gpu_relax = bool(args.gpu_relax)
    str_pairs = args.pairs.replace(',','_')

    aa_option = args.aa_option
    mutation_option = args.mutation_option

    output_dir = os.path.join(output_dir,str_pairs)
    if len(glob(output_dir+'/pred_rank0_*_relaxed.pdb'))>0:
        print('output exists, quitting')
        raise SystemExit()
    if len(glob(output_dir+'/pred_rank0_*_unrelaxed.pdb'))>0:
        print('an unrelaxed output exists, this can cause issues later in the pipeline \n please remove the unrelaxed files and try again, quitting')
        raise SystemExit()

    with open(os.path.join(input_dir, "features.pkl"), "rb") as f: #load features (come from an alphafold run)
        features = pickle.load(f)
    sequence = features['sequence'][0].decode()
    cysteines = [m.start() for m in re.finditer('C', sequence)]
    pairs = []
    # name = ''


    for i,p in enumerate(str_pairs.split("_")):
        pair = []
        # name += p+'_'
        for j,value in enumerate(p):
            pair.append(int(value))
        pairs.append(pair)
    print('zero indexed cysteine locations', cysteines)
    print('pairs to be forced', pairs)

    assert os.path.exists(input_dir), "Input path does not exist!"
    os.makedirs(output_dir, exist_ok=True)


    # Adjust msa features
    msa = features["msa"]
    print(type(msa))
    print(msa.shape)
    indices_sets, pairs, msa = mutation_method(len(pairs), pairs, msa, mutation_option)

    msa = replace_cystines(msa, cysteines, aa_option)
    for i, pair in enumerate(pairs): #with zero indexing turn the split up cystines back into cystines in the pairs 
        msa = back_to_cysteine(msa, indices_sets[i], cysteines, pair)
    features["msa"] = msa

    if mutation_option=='Nalignments':
        features['msa_species_identifiers'] = np.tile(features['msa_species_identifiers'],(1,len(pairs)))
        features['num_alignments']=features['num_alignments']*len(pairs)
        features['deletion_matrix_int'] = np.tile(features['deletion_matrix_int'],(len(pairs),1))
    
    
    print(msa.shape)
    print(templated)
    if not templated:
        lseq = features["template_all_atom_positions"].shape[1]
        template_features = mk_mock_template(lseq, multimer=False)
        features.update(template_features)


    model_runners = {}
    model_names = ('model_1', 'model_2') # there's an issue with model 3, model 4, and maybe model 5 -
    for model_name in model_names:
        model_config = config.model_config(model_name)

        # config templated params
        # templated is bool - default value for all of these except data.eval.subsample_templates is True 
        model_config.data.common.use_templates = templated
        model_config.data.common.reduce_msa_clusters_by_max_templates = templated
        model_config.model.embeddings_and_evoformer.template.embed_torsion_angles = templated
        model_config.model.embeddings_and_evoformer.template.enabled = templated
        model_config.data.eval.subsample_templates = templated

        # get the model parameters from AF 
        model_params = data.get_model_haiku_params(
            model_name=model_name,
            data_dir=data_dir
        )
        #class to run a model
        model_runner = model.RunModel(model_config, model_params)
        # have a model runner for each predicition you want to make 
        for i in range(num_preds_per_model):
            model_runners[f'{model_name}_pred_{i}'] = model_runner
    #save the updated features (ie the changed MSA)
    with open(os.path.join(output_dir, "updated_features.pkl"), "wb") as f:
        pickle.dump(features, f, protocol=4)

    # Predict
    unrelaxed_pdbs = {}
    unrelaxed_preds = {}
    ranking_confidences = {}
    results = {}

    #dict.items() returns iterable of key value tuple pairs
    for model_name, model_runner in model_runners.items():
        print(f"Running model: {model_name}")
        random_seed = random.randrange(sys.maxsize)
        processed_features = model_runner.process_features(features, random_seed=random_seed) 
        result = model_runner.predict(processed_features, random_seed)
        pred = protein.from_prediction(
            processed_features,
            result,
            b_factors=np.repeat(result["plddt"][:, None], residue_constants.atom_type_num, axis=-1),
            remove_leading_feature_dimension= not model_runner.multimer_mode
        )
        result.update({"random_seed": random_seed})
        ranking_confidences[model_name] = result["ranking_confidence"]
        results[model_name] = result
        unrelaxed_pdbs[model_name] = protein.to_pdb(pred)
        unrelaxed_preds[model_name] = pred
    
    #save unrelaxed pdb and the results dictionary from the prediction 
    for idx, (model_name, _) in enumerate(
            sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)
    ):
        with open(os.path.join(output_dir, f'pred_rank{idx}_{model_name}_unrelaxed.pdb'), 'w') as f:
            f.write(unrelaxed_pdbs[model_name])
        with open(os.path.join(output_dir, f'result_rank{idx}_{model_name}.pkl'), 'wb') as f:
            pickle.dump(results[model_name], f, protocol=4)

    if amber_relax:
        relax_metrics = {}
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            stiffness=RELAX_STIFFNESS,
            exclude_residues=RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=gpu_relax)

        for idx, (model_name, _) in enumerate(
                sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)
        ):
            relaxed_pdb_str, _, violations = amber_relaxer.process(prot=unrelaxed_preds[model_name])
            relax_metrics[model_name] = {
                'rank': idx,
                'remaining_violations': violations,
                'remaining_violations_count': sum(violations)
            }
            print(relax_metrics)
            with open(os.path.join(output_dir, f'pred_rank{idx}_{model_name}_relaxed.pdb'), 'w') as f:
                f.write(relaxed_pdb_str)
        with open(os.path.join(output_dir, "relax_metrics.json"), "w") as f:
            f.write(json.dumps(relax_metrics, indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, help="input AlphaFold result", required=True,
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="output dir for results", required=True,
    )
    parser.add_argument(
        "--pairs", "-p", type=str, help="connectivity pairs (e.g. 14,23,56)", required=True,
    )
    parser.add_argument(
        "--num_preds_per_model", "-n", type=int, help="Number of preds per model", default=5,
    )
    parser.add_argument(
        "--aa_option", "-a", type=str, help="option for replacment of aa ['all', 'hydrophobic', 'small']", default='all',
    )
    parser.add_argument(
        "--mutation_option", "-m", type=str, help="option for mutation introduction ['breakN', 'Nalignments', 'breakN+1']", default='breakN',
    )
    parser.add_argument(
        "--amber_relax", "-r", action="store_true", help="whether to use relaxation",
    )
    parser.add_argument(
        "--templated", "-t", action="store_true", help="add to template"
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, help="location of data dir for AlphaFold", required=True,
    )
    parser.add_argument(
        "--gpu_relax", "-g", action='store_true', help="whether to use gpu for relaxation"
    )

    args = parser.parse_args()

    main(args)

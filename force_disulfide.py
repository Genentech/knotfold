import os
from pyrosetta import *
from pyrosetta.rosetta.protocols.denovo_design import DisulfidizeMover
from pyrosetta.rosetta.protocols.relax import FastRelax
import argparse
from glob import glob
import re
init(options="-write_all_connect_info")

def force_disulfide(dm, pose, res1, res2,scorefxn):
    dm.make_disulfide(
        pose=pose,
        res1=res1,
        res2=res2,
        relax_bb=True,
        sfxn=scorefxn
    )

def main(args):
    input_pdb = args.input_pdb
    str_pairs = args.pairs
    output_dir = args.output_dir

    pose = pose_from_pdb(input_pdb)
    sequence = pose.sequence()

    scorefxn = get_fa_scorefxn()
    cysteines = [m.start() for m in re.finditer('C', sequence)]
    cysteines = [c+1 for c in cysteines]

    pairs = []
    for i,p in enumerate(str_pairs.split(",")):
        pair = []
        for j,value in enumerate(p):
            pair.append(int(value))
        pairs.append(pair)
    
    dm = DisulfidizeMover()
    current = dm.find_current_disulfides(
        pose=pose,
        subset1=pose.conformation().get_residue_mask(),
        subset2=pose.conformation().get_residue_mask(),
    )
    needed = []
    current = list(current)

    for pair in pairs:
        new_pair = [cysteines[pair[0]-1],cysteines[pair[1]-1]]
        if (tuple(new_pair) in current) or (tuple(new_pair[::-1]) in current):
            None
        else:
            needed.append(pair)
    
    if len(needed)==0:
        print('no disulfides needed, skipping')
        pose.dump_pdb(os.path.join(output_dir,'disulfidized_'+f.split('/')[-1]))

    else:
        for pair in needed:
            print('forcing',pair, cysteines[pair[0]-1],cysteines[pair[1]-1])
            force_disulfide(dm,pose,cysteines[pair[0]-1],cysteines[pair[1]-1],scorefxn)
    
        fr = FastRelax(scorefxn) 
        fr.apply(pose)
        
        pose.dump_pdb(os.path.join(output_dir,'disulfidized_'+input_pdb.split('/')[-1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pdb", "-i", type=str, help="input AlphaFold result", required=True,
    )
    parser.add_argument(
        "--pairs", "-p", type=str, help="connectivity pairs (e.g. 14,23,56)", required=True,
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="output directory", required=True,
    )

    args = parser.parse_args()

    main(args)

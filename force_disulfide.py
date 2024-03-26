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
    input_dir = args.input_dir
    if args.pairs is None:
        str_pairs = input_dir.split('/')[-1].replace('_',',')
    else:
        str_pairs = args.pairs

    if args.files:
        files = glob(args.files)
    else:
        files = glob(input_dir+'/pred_rank*_model_*_pred_*_relaxed.pdb')

        if len(files)==0:
            connect = str_pairs.replace(',','_')
            files = glob(input_dir+'/'+connect+'/pred_rank*_model_*_pred_*_relaxed.pdb')
            input_dir = input_dir+'/'+connect
    
     
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.input_dir is None:
        output_dir = os.path.dirname(files[0])
    else:
        output_dir=input_dir
    print(output_dir)
    print(files)
    print(os.path.isfile(os.path.join(output_dir,'disulfidized_'+files[0].split('/')[-1])))
    if len(glob(os.path.join(output_dir,'disulfidized_*'))) >=10:
        print('output exists, quitting')
        raise SystemExit()

    disulfide = glob(output_dir+'/disulfidized_pred_rank*_model_1*_pred_*_relaxed.pdb')
    files = [f for f in files if f not in disulfide]

    if len(files)==0:
        print('no files found, quitting')
        raise SystemExit()

    for f in files:

        pose = pose_from_pdb(f)
        sequence = pose.sequence()
        print('sequence',sequence)
        scorefxn = get_fa_scorefxn()
        cysteines = [m.start() for m in re.finditer('C', sequence)]
        cysteines = [c+1 for c in cysteines]
        print('cysteines',cysteines)
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
        print('pairs',pairs)
        print('current',current)
        for pair in pairs:
            print(pair)
            new_pair = [cysteines[pair[0]-1],cysteines[pair[1]-1]]
            print('new_pair',new_pair)
            if (tuple(new_pair) in current) or (tuple(new_pair[::-1]) in current):
                None
            else:
                needed.append(pair)
        print('needed',needed)
        
        if len(needed)==0:
            print('no disulfides needed, skipping')
            pose.dump_pdb(os.path.join(output_dir,'disulfidized_'+f.split('/')[-1]))

        else:
            for pair in needed:
                print('forcing',pair, cysteines[pair[0]-1],cysteines[pair[1]-1])
                force_disulfide(dm,pose,cysteines[pair[0]-1],cysteines[pair[1]-1],scorefxn)
        
            fr = FastRelax(scorefxn) 
            fr.apply(pose)
            
            pose.dump_pdb(os.path.join(output_dir,'disulfidized_'+f.split('/')[-1]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, help="input AlphaFold result", required=False,
    )
    parser.add_argument(
        "--output_dir","-d",type=str, required=False, help='Directory to save output, if not used defaults to input_dir',
    )
    parser.add_argument(
        "--files", '-f', type=str, help="pdb file that will be editied",
    )
    parser.add_argument(
        "--pairs", "-p", type=str, help="connectivity pairs (e.g. 14,23,56)", required=False,
    )

    args = parser.parse_args()

    main(args)

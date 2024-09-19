import torch
import argparse

from scripts.model import Decoder
from scripts.process_input import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('complex_pdb', type=str)
    parser.add_argument('chains_info', type=str)
    parser.add_argument('mutation_info', type=str)
    parser.add_argument('--load_model', default='ckpt/model.pt')

    args = parser.parse_args()

    ckpt = torch.load(args.load_model)
    config = ckpt['config']
    config_namespace = argparse.Namespace(**config)
    weight = ckpt['model_state']
    model = Decoder(config_namespace).cuda()
    model.load_state_dict(weight)
    vars(args).update(vars(config_namespace))

    with torch.no_grad():
        model.eval()
        protein = Proteins(args.complex_pdb, args.chains_info)
        mut_proteins = get_chains_dict(args.mutation_info, protein.complex)
        complex = completize(mut_proteins, args.esm)
        ddG_pre = model(*complex)
        print(f'Predicted ddG value for input mutant is: {ddG_pre.item():.3f} kcal/mol.')

# abCAN: a Practical and Biology-aligned Attention Network for Predicting Mutant Antibody Affinity.

## Dependencies
Our model in tested in Linux with the following packages:
- CUDA >= 11.8
- python == 3.8.18
- PyTorch == 2.0.0
- Numpy == 1.24.3
  
## Using our pretrained model for mutated antibody affinity prediction
Our model takes an input of a PDB file of the wild-type antibody-antigin complex with mutation information and outputs the predicted change in affinity upon mutation (ΔΔG).

**Note:** The wild type complex simply stands for complex before mutation, it can be any complex to be optimized (e.g. a complex from Protein Data Bank).

For example, complex `1ahw.pdb` is an antibody (chain A,B) and antigen (chain C) complex. Assume there're two mutations (position 138 Leucine changes to Alanine; position 139 Aspartic acid changes to Alanie) on chain C, run this command:
```
python predict.py ./data/1AHW.pdb AB_C KC138A,DC139A

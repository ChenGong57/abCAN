# abCAN: Antibody-antigen Complex Attention Network for Predicting Mutant Antibody Affinity.
---
This is the implementation of paper: 
## Dependencies
---
Our model in tested in LInux with the following packages:
- CUDA >= 11.8
- python == 3.8.18
- PyTorch == 2.0.0
- Numpy == 1.24.3
## Using our pretrained model for mutated antibody affinity prediction
---
Our model takes an input of wild type antibody-antigin complex pdb file with mutated information. The wild type complex simply stands for complex before mutation, it can be any complex to be optimized.
For example, for complex 1ahw.pdb with two mutations(position 138 Leucine changes to Alanine; position 139 Aspartic acid changes to Alanie) on chain C, run this command:
```
python predict.py --mut_pos KC138A,DC139A --ori_complex 1ahw.pdb

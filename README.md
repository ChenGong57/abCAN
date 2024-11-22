# abCAN: a Practical and Novel Attention Network for Predicting Mutant Antibody Affinity.
![image](https://github.com/ChenGong57/abCAN/blob/main/data/architecture.png)
WT stands for wild-type, referring to the pre-mutant antibody-antigen complex.
## Dependencies
Our model in tested in Linux with the following packages:
- CUDA >= 11.8
- python == 3.8.18
- PyTorch == 2.0.0
- Numpy == 1.24.3
  
## Using our pretrained model for mutated antibody affinity prediction
Our model takes an input of a PDB file of the wild-type antibody-antigin complex with mutation information and outputs the predicted change in affinity upon mutation (ΔΔG).

**Note:** The wild type complex simply stands for complex before mutation, it can be any complex to be optimized (e.g. a complex from Protein Data Bank).

For example, complex `2B2X.pdb` is an antibody (chain H,L) and antigen (chain A) complex. Assume there're three mutations (position 50 tyrosine changes to threonine; position 64 glutamic acid changes to lysine; position 99 phenylalanine changes to tryptophan) on chain H and two mutations (position 28 alutamine changes to serine; position 52 tyrosine changes to asparagine) on chain L, run this command:
```
python predict.py ./data/2B2X.pdb HL_A VH50T,EH64K,FH99W,QL28S,YL52N

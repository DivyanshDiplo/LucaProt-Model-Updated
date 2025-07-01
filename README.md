# 🛠️ Updates

This repository contains updates and improvements on top of the original [LucaProt](https://github.com/alibaba/LucaProt) codebase.

📄 **Reference:** Original README from LucaProt –  
[https://github.com/alibaba/LucaProt/blob/master/README.md](https://github.com/alibaba/LucaProt/blob/master/README.md)

---

## 1) 🔍 Prediction from Protein 3D Structure

While running `structure_from_esm_v1.py`, the following error was encountered:

ModuleNotFoundError: No module named 'openfold'


### ✅ Fix:
Update Pytorch and gnu compiler version
<pre>
  # Activate your conda environment
    conda activate $SCRATCH/conda/envs/lucaprot
    
    # Uninstall current PyTorch Lightning and related packages
    pip uninstall pytorch-lightning lightning-fabric lightning
    
    # Install compatible PyTorch Lightning version
    pip install pytorch-lightning==1.8.4
    
    # Verify the installation
    python -c "from pytorch_lightning.utilities.seed import seed_everything; print('PyTorch Lightning 1.8.4 installed successfully')"
    
    #gnu compiler in HPC 
    module load compiler/gcc/9.1.0
</pre>

Install all OpenFold dependencies (reference: [ESM README](https://github.com/facebookresearch/esm/blob/main/README.md)):
<pre>
  pip install fair-esm  # latest release, OR:
  pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
</pre>

<pre>
  pip install "fair-esm[esmfold]"
  # OpenFold and its remaining dependency
  pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
  pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
</pre>

If it gives out of memory errors use cpu-only mode instead of cpu-offloading
<pre>
cd LucaProt/src/protein_structure/    
export CUDA_VISIBLE_DEVICES=0
python structure_from_esm_v1.py \
    -name protein_id1_ESM,protein_id2ESM  \
    -seq VGGLFDYYSVPIMT,LPDSWENKLLTDLILFAGSFVGSDTCGKLF \
    -o pdbs/rdrp/  \
    --num-recycles 4 \
    --truncation_seq_length 4096 \
    --chunk-size 64 \
    --cpu-only \
    --batch_size 1
</pre>

after this it worked for me and it generated a pdb file called protein_1.pdb at `cd ~/scratch/LucaProt/src/protein_structure/pdbs/rdrp/pdb/`


## 2) 🔍 Using ColabFold (alphafold) for 3D structure Predictions

I edited the `structure_from_esm_v1.py` and renamed it to `structure_from_alphafold2.py`. Remove the orignal and add this at  `cd ~/scratch/LucaProt/src/protein_structure/`

Dependencies. You can paste this in a txt file and run `pip install -r filename.txt`
<pre>
  # requirements_alphafold.txt
  # Dependencies for the LucaProt AlphaFold/ColabFold wrapper script
  
  # The core package. This will pull in a compatible version of PyTorch, NumPy, etc.
  colabfold[alphafold]
  
  # Dependencies for your helper scripts (e.g., utils.py)
  biopython
  pandas
  # pynvml is needed by utils.py. Installing nvidia-ml-py is the modern way to get it.
  nvidia-ml-py
  scikit-learn

</pre>


Some Usefull commands (for my reference).
<pre>
  conda env remove -n lucaprot #replace lucaprot with the name of your conda environment
  conda env list
  conda activate lucaprot

  module load compiler/cuda/11.0/compilervars
  module load compiler/gcc/9.1.0
  
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  pip install nvidia-ml-py
  conda install -c conda-forge pynvml
  pip install scikit-learn
  
  pip install -r requirements_alphafold.txt
  rm -rf pdbs/rdrp/

  python structure_from_alphafold2.py \
    -name protein_id1,protein_id2  \
    -seq VGGLFDYYSVPIMT,LPDSWENKLLTDLILFAGSFVGSDTCGKLF \
    -o pdbs/rdrp/  \
    --num-recycle 4 \
  
</pre>

## 3) 🔍 Using Alphafold for prediction of structural embeddings

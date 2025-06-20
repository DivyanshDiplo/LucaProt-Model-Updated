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

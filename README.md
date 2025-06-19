# 🛠️ Updates

This repository contains updates and improvements on top of the original [LucaProt](https://github.com/alibaba/LucaProt) codebase.

📄 **Reference:** Original README from LucaProt –  
[https://github.com/alibaba/LucaProt/blob/master/README.md](https://github.com/alibaba/LucaProt/blob/master/README.md)

---

## 1) 🔍 Prediction from Protein 3D Structure

While running `structure_from_esm_v1.py`, the following error was encountered:

ModuleNotFoundError: No module named 'openfold'


### ✅ Fix:
Install all OpenFold dependencies (reference: [ESM README](https://github.com/facebookresearch/esm/blob/main/README.md)):

<pre>
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
</pre>

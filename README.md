# 🛠️ Updates

This repository contains updates and improvements on top of the original [LucaProt](https://github.com/alibaba/LucaProt) codebase.

📄 **Reference:** Original README from LucaProt –  
[https://github.com/alibaba/LucaProt/blob/master/README.md](https://github.com/alibaba/LucaProt/blob/master/README.md)

---

## 1) 🔍 Prediction from Protein 3D Structure

While running `structure_from_esm_v1.py`, the following error was encountered:
"ModuleNotFoundError: No module named 'openfold'"

### ✅ Fix:
The script has been updated to **use HuggingFace's ESMFold** instead of the original **ESM + OpenFold** approach.

- Performance remains similar.
- This eliminates the need to install and configure OpenFold separately.

📥 **Download the updated script:**  
[`structure_from_esm_v1.py`](./src/protein_structure/structure_from_esm_v1.py)

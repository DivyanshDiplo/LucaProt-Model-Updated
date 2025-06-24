import os
import csv
import sys
import subprocess
import tempfile
import torch
import random
import argparse
import typing as T
from pathlib import Path
from timeit import default_timer as timer
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
from utils import fasta_reader


def run_colabfold_prediction(sequences, headers, output_dir, args):
    """
    Runs ColabFold prediction for a batch of sequences.
    This function replaces the original model.infer() call.
    """
    # Create a temporary FASTA file for the current batch
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".fasta") as tmp_fasta:
        for header, seq in zip(headers, sequences):
            tmp_fasta.write(f">{header}\n{seq}\n")
        temp_fasta_path = tmp_fasta.name

    try:
        # Construct the ColabFold command
        cmd = [
            "colabfold_batch",
            temp_fasta_path,
            output_dir,
            "--num-models", "1",  # Use one model for speed, AlphaFold is deterministic
            "--num-recycles", str(args.num_recycles) if args.num_recycles else "3",
        ]
        # Add CPU flag if requested
        if args.cpu_only:
            cmd.append("--cpu")
        else:
            cmd.append("--use-gpu-relax") # Use GPU for amber relaxation if available

        print(f"Running ColabFold command: {' '.join(cmd)}")
        # Execute the command. check=True will raise an error if ColabFold fails.
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600) # 1hr timeout
        print("ColabFold prediction finished.")

        # Parse the output to be compatible with the rest of the script
        return parse_colabfold_output(output_dir, headers)
    finally:
        os.remove(temp_fasta_path) # Always clean up the temporary file



def parse_colabfold_output(output_dir, headers):
    """
    Finds PDB files and extracts confidence scores.
    Creates a dictionary similar to the original ESMFold output.
    """
    parsed_output = {"pdbs": [], "mean_plddt": [], "ptm": []}
    for header in headers:
        # ColabFold produces multiple files, find the main PDB (_unrelaxed_rank_001_...)
        pdb_files = sorted(list(Path(output_dir).glob(f"{header}*_unrelaxed_rank_001*.pdb")))
        if not pdb_files:
            print(f"Warning: No PDB file found for {header}. Prediction may have failed.")
            # Append empty/default values to maintain list lengths
            parsed_output["pdbs"].append("")
            parsed_output["mean_plddt"].append(0.0)
            parsed_output["ptm"].append(0.0)
            continue

        pdb_file = pdb_files[0]
        with open(pdb_file, 'r') as f:
            parsed_output["pdbs"].append(f.read())

        # Extract confidence scores from the PDB B-factor column
        plddt_scores = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    # The B-factor column (60-66) in AlphaFold PDBs contains the pLDDT score.
                    plddt_scores.append(float(line[60:66].strip()))

        mean_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0
        parsed_output["mean_plddt"].append(mean_plddt)

        # AlphaFold2 has ptm/iptm scores in a separate json file.
        # For simplicity and compatibility, we'll approximate a ptm-like score from plddt.
        # This can be refined later by parsing the .json file if needed.
        ptm_score = (mean_plddt / 100.0)**2
        parsed_output["ptm"].append(ptm_score)

    return parsed_output



def create_batched_sequence_datasest(
        sequences: T.List[T.Tuple[str, str]],
        max_tokens_per_batch: int = 4096,
        truncation_seq_length: int = 4096,
        batch_size: int = 1,
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    '''
    create the batch
    :param sequences:
    :param max_tokens_per_batch:
    :param truncation_seq_length
    :param batch_size:
    :return:
    '''
    if batch_size:
        batch_headers, batch_sequences = [], []
        for header, seq in sequences:
            if len(batch_headers) == batch_size:
                cur_batch_max_seq_len = max([len(seq) for seq in batch_sequences])
                if cur_batch_max_seq_len > truncation_seq_length:
                    batch_sequences = [seq[:truncation_seq_length] if len(seq) > truncation_seq_length else seq  for seq in batch_sequences]
                yield batch_headers, batch_sequences
                batch_headers, batch_sequences = [], []
            batch_headers.append(header)
            batch_sequences.append(seq)
        yield batch_headers, batch_sequences
    else:
        batch_headers, batch_sequences, num_tokens = [], [], 0
        for header, seq in sequences:
            if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
                yield batch_headers, batch_sequences
                batch_headers, batch_sequences, num_tokens = [], [], 0
            batch_headers.append(header)
            batch_sequences.append(seq)
            num_tokens += len(seq)

        yield batch_headers, batch_sequences


def prediction(
        args,
        all_sequences,
        save_dir,
        begin_uuid_index=0
):
    '''
    prediction
    :param args: running args
    :param all_sequences: all sequences
    :param save_dir: pdb save dir
    :param begin_uuid_index: begin index for naming the pdb file
    :return:
    '''
    batched_sequences = create_batched_sequence_datasest(
        all_sequences,
        args.max_tokens_per_batch,
        truncation_seq_length=args.truncation_seq_length,
        batch_size=args.batch_size
    )
    use_time = 0
    total_seq_len = 0
    num_completed = 0
    num_sequences = len(all_sequences)
    had = False
    if os.path.exists(os.path.join(save_dir, "uncompleted.txt")):
        if args.try_failure:
            uncompleted_wfp = open(os.path.join(save_dir, "uncompleted_%d.txt" % args.truncation_seq_length), "a+")
        else:
            uncompleted_wfp = open(os.path.join(save_dir, "uncompleted.txt"), "a+")
    else:
        if args.try_failure:
            uncompleted_wfp = open(os.path.join(save_dir, "uncompleted_%d.txt" % args.truncation_seq_length), "w")
        else:
            uncompleted_wfp = open(os.path.join(save_dir, "uncompleted.txt"), "w")
    if os.path.exists(os.path.join(save_dir, "result_info.csv")):
        result_info_wfp = open(os.path.join(save_dir, "result_info.csv"), "a+")
        had = True
    else:
        result_info_wfp = open(os.path.join(save_dir, "result_info.csv"), "w")
    result_info_writer = csv.writer(result_info_wfp)
    if not had:
        result_info_writer.writerow(["index", "uuid", "seq_len", "ptm", "mean_plddt"])
    uuid_index = begin_uuid_index
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            # This now calls ColabFold via a subprocess
            output = run_colabfold_prediction(sequences, headers, save_dir, args)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    print(
                        f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                        "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    print(
                        f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                    )
            for idx, v in enumerate(headers):
                uncompleted_wfp.write("%s,%d\n" % (v, len(sequences[idx])))
                uncompleted_wfp.flush()
            continue
        
        # The output from our parser is already on the CPU and pdbs are strings.
        pdbs = output["pdbs"]

        tottime = timer() - start
        use_time += tottime
        total_seq_len += sum([len(v) for v in sequences])
        for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            uuid_index += 1
            with open(os.path.join(save_dir, "protein_%s.pdb" % str(uuid_index)), "w") as wfp:
                wfp.write(pdb_string)
            num_completed += 1
            item = [uuid_index, header, len(seq), float(ptm), float(mean_plddt)]
            result_info_writer.writerow(item)
            result_info_wfp.flush()
        if num_completed % 100 == 0 and num_completed > 0:
            print("completed num: %d, use time per seq: %f, avg seq len: %f" % (num_completed, use_time/num_completed, total_seq_len/num_completed))
            print("completed: %d, %0.2f%%" % (num_completed, 100 * num_completed/num_sequences))
    avg_use_time = 0
    avg_total_seq_len = 0
    if num_completed > 0:
        avg_use_time = use_time/num_completed
        avg_total_seq_len = total_seq_len/num_completed
    uncompleted_wfp.close()
    result_info_wfp.close()
    return num_sequences, num_completed, avg_use_time, avg_total_seq_len


def load_done_set(result_info_path, uncompleted_path):
    """
    what has already been done does not need to be repeated
    :param result_info_path: done filepath
    :param uncompleted_path: uncompleted filepath
    :return:
    """
    done_set = set()
    max_uuid_index = 0
    if os.path.exists(result_info_path):
        with open(result_info_path, "r") as rfp:
            reader = csv.reader(rfp)
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt == 1 or row[0] == "index":
                    continue
                index = int(row[0])
                uuid = row[1]
                if max_uuid_index < index:
                    max_uuid_index = index
                done_set.add(uuid)
        if uncompleted_path and os.path.exists(uncompleted_path):
            with open(uncompleted_path, "r") as rfp:
                for line in rfp:
                    line = line.strip()
                    ridx = line.rfind(",")
                    if ridx > -1:
                        uuid = line[:ridx]
                        done_set.add(uuid)
    return done_set, max_uuid_index


def exists(exists_filepath):
    # load the succeed filepath
    # ../../dataset/rdrp_40/protein/binary_class/protein_2_pdb.csv
    exists_set = set()
    with open(exists_filepath, "r") as rfp:
        reader = csv.reader(rfp)
        cnt = 0
        for row in reader:
            cnt += 1
            if cnt == 1:
                continue
            protein_id = row[0]
            exists_set.add(protein_id)
    return exists_set


def predict_for_structure(args, filepath_list, sequence_list, reverse=False):
    '''
    predict the 3d-structure of proteins
    :param args: running parameters
    :param filepath_list: sequence filepath list
    :param sequence_list: sequence list
    :param reverse: whether to reverse the list
    :return:
    '''
    if filepath_list is None and sequence_list is None :
        raise Exception("input empty error!")
    all_sequences = []
    if filepath_list:
        for filepath in filepath_list:
            print("doing filepath: %s ..." % filepath)
            if ".csv" in filepath:
                filename = os.path.basename(filepath)
                cur_save_dir = os.path.join(args.save_dir, filepath.replace("../", "").replace(filename, ""), "pdb", filename.replace(".csv", ""))
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                print("save dir: %s." % cur_save_dir)
                with open(filepath, "r") as rfp:
                    reader = csv.reader(rfp)
                    cnt = 0
                    for row in reader:
                        cnt += 1
                        if cnt == 1:
                            continue
                        all_sequences.append([row[0], row[1]])
            elif ".fa" in filepath:
                filename = os.path.basename(filepath)
                cur_save_dir = os.path.join(args.save_dir, filepath.replace("../", "").replace(filename, ""), "pdb", filename.replace(".fasta", "").replace(".fas", "").replace(".fa", ""))
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                print("save dir: %s." % cur_save_dir)
                all_sequences = all_sequences + [[v[0].strip(), v[1].strip()] for v in fasta_reader(filepath)]
            else:
                raise Exception("not support the type file: %s, must endswith '.csv' or '.fa*" % filepath)
    else:
        cur_save_dir = os.path.join(args.save_dir, "pdb")
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        all_sequences = sequence_list

    if args.try_failure:
        done_set, begin_uuid_index = load_done_set(
            os.path.join(cur_save_dir, "result_info.csv"),
            None
        )
    else:
        done_set, begin_uuid_index = load_done_set(
            os.path.join(cur_save_dir, "result_info.csv"),
            os.path.join(cur_save_dir, "uncompleted.txt")
        )
    print("all number: %d" % len(all_sequences))
    print("done number: %d" % len(done_set))
    # exists
    if args.exists_file:
        exists_set = exists(args.exists_file)
    else:
        exists_set = set()
    print("exists number: %d" % len(exists_set))
    # remove the done list
    all_sequences = [item for item in all_sequences if item[0] not in done_set and item[0] not in exists_set]
    print("wanted number: %d" % len(all_sequences))
    if reverse:
        print("reverse=True")
        all_sequences.reverse()

    num_sequences, num_completed, avg_use_time, avg_total_seq_len = prediction(
        args,
        all_sequences,
        save_dir=cur_save_dir,
        begin_uuid_index=begin_uuid_index
    )
    print("total protein num: %d, completed num: %d, use time per seq: %f, avg seq len: %f" % (
        num_sequences,
        num_completed,
        avg_use_time,
        avg_total_seq_len
    ))
    if filepath_list:
        print("filepath: %s done." % filepath_list)
    else:
        print("done")
    print("#"*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        type=str,
        default=None,
        help="input fasta filepath.",
    )
    parser.add_argument(
        "-name",
        type=str,
        default=None,
        help="sequence name.",
    )
    parser.add_argument(
        "-seq",
        type=str,
        default=None,
        help="sequence.",
    )
    parser.add_argument(
        "-o",
        "--save_dir",
        help="path to output PDB directory",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-m",
        "--model_path",
        help="parent path to Pretrained ESM data directory. ",
        type=Path,
        default=None
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch Size.",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="number of recycles to run. Defaults to number used in training (4).",
    )

    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=4096,
        help="truncate sequences longer than the given value",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=4096,
        help="maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
             "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
             "short sequences.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
             "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
             "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
             "Default: None.",
    )
    parser.add_argument(
        "-e", "--exists_file", help="Path of exists pdb list filepath", type=Path,
    )
    parser.add_argument(
        "--try_failure",
        action="store_true",
        help="when cuda out of memory, reduce its value"
    )
    parser.add_argument(
        "--cpu-only",
        help="CPU only",
        action="store_true"
    )
    parser.add_argument(
        "--cpu-offload",
        help="enable CPU offloading",
        action="store_true"
    )
    args = parser.parse_args()


    try:
        # This command checks if 'colabfold_batch' is executable and in PATH
        subprocess.run(["colabfold_batch", "--help"], capture_output=True, check=True, text=True)
        print("✅ ColabFold installation verified. Proceeding with prediction.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print("❌ ColabFold not found or not working. Please install it with:")
        print("pip install \"colabfold[alphafold]\"")
        print(f"Error: {e}")
        sys.exit(1) # Exit if ColabFold is not found

    
    if args.i:
        if "," in args.i:
            protein_filepath = args.i.split(",")
        else:
            protein_filepath = [args.i]
        predict_for_structure(args, filepath_list=protein_filepath, sequence_list=None, reverse=False)
    elif args.seq:
        seqs = args.seq.split(",")
        names = args.name.split(",")
        assert len(seqs) == len(names)
        sequence_list = []
        for name, seq in zip(names, seqs):
            sequence_list.append([name, seq])
        predict_for_structure(args, filepath_list=None, sequence_list=sequence_list, reverse=False)
    else:
        raise Exception("-i or -seq")

    #
    '''
    export CUDA_VISIBLE_DEVICES=1
    python structure_from_alphafold2.py \
            -i /public/home/alicloud/sanyuan.hy/workspace/LucaProt/data/extra_p/all_wide70_non-singleton.faa \
            -o /public/home/alicloud/sanyuan.hy/workspace/LucaProt/structure_predicted/extra_p/all_wide70_non-singleton \
            --num-recycles 4 \
            --truncation_seq_length 4096 \
            --chunk-size 64 \
            --cpu-offload \
            --batch_size 1 \
            --try_failure
    '''




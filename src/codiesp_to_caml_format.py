# !!!!REDUNDANT!!!!
# THERE IS A NOTEBOOK IN LLM_ICD_CODING REPO FOR THIS PURPOSE :) :) :)
raise RuntimeError("nice") 
import pandas as pd
import os, typing


splits = ("train", "test") # The splits to convert
path   = "..\\data\\codiesp\\{split}\\text_files_gpt_en\\"

for split in splits:
    split_path = path.format(split=split)
    
    labels = pd.read_csv(
        "..\\data\\codiesp\\{split}\\{split}D.tsv".format(split=split),
        sep='\t',
        names=("caseID", "code")
    )

    rows : list[dict[str, typing.Any]] = {}
    for f in os.listdir(split_path):
        case_id = f.rstrip(".txt")
        case_labels = list(set(labels.loc[labels["caseID"] == case_id]["code"]))
        case_labels_semicolon_separated = ";".join(case_labels)

        case_note = open(split_path + f, "r").read()
        # - lowercase all tokens
        # - remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
        lowercased = case_note.lower()


"""Adapted from the CAML repo."""

DATA_DIR = "..\\data"


def load_code_descriptions():
    #load description lookup from the appropriate data files
    desc_dict = {}

    with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
        for i,row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict

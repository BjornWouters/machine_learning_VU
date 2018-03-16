from Bio import SeqIO, SeqUtils
from googlesearch import search
import mygene
import numpy as np
import pandas as pd
import re
import urllib.error
import urllib.request


def read_data(data_file):
    df = pd.read_csv(data_file)
    return df


def create_columns(df):
    amino_acids = ['Gly', 'Ala', 'Val', 'Leu', 'Ile',
                   'Pro', 'Phe', 'Tyr', 'Trp', 'Ser',
                   'Thr', 'Cys', 'Met', 'Asn', 'Gln',
                   'Lys', 'Arg', 'His', 'Asp', 'Glu',
                   'Stop']
    for aa in amino_acids:
        df['C_'+aa] = 0
        df['M_'+aa] = 0

    variations = df.Variation.tolist()
    regular_expression = re.compile('^[a-zA-Z ]+$')
    # Just for interpretation purpose
    filtered_variation = set(filter(regular_expression.match,
                                    variations))

    interesting_variation = ['Truncating Mutations',
                             'Deletion', 'Amplification',
                             'Fusions']

    for varation in interesting_variation:
        df[varation] = 0

    df['from_start'] = np.nan
    df['from_end'] = np.nan


def retrieve_uniprot_identifier(hgnc_symbol):
    query = '{} uniprot homo sapiens'.format(hgnc_symbol)
    url = next(search(query))
    uniprot_accession = url.split('/')[-1]
    return uniprot_accession


def fetch_protein_info(gene):
    mg = mygene.MyGeneInfo()
    result = mg.query(gene, size=1, species='human')
    hgnc_symbol = result['hits'][0]['symbol']
    uniprot_id = retrieve_uniprot_identifier(hgnc_symbol)
    data = urllib.request.urlopen(
        "http://www.uniprot.org/uniprot/" + uniprot_id + ".xml")
    record = SeqIO.read(data, "uniprot-xml")
    return record


def assign_mutation_location(df, index, protein_record):
    sequence_length = len(protein_record.seq)
    variation = df.loc[index].Variation

    has_index = re.findall(r'\d+', variation)
    if has_index:
        mutation_index = int(has_index[0])
        df.at[index, 'from_start'] = int(mutation_index)
        df.at[index, 'from_end'] = int(sequence_length - mutation_index)
        aa_change = list(variation)

        try:
            control = SeqUtils.IUPACData.protein_letters_1to3[aa_change[0]]
            df.at[index, 'C_'+control] = 1
        except KeyError:
            if aa_change[0] == '*':
                df.at[index, 'C_Stop'] = 1
        try:
            mutation = SeqUtils.IUPACData.protein_letters_1to3[aa_change[-1]]
            df.at[index, 'M_'+mutation] = 1
        except KeyError:
            if aa_change[-1] == '*':
                df.at[index, 'M_Stop'] = 1

    elif variation in ['Truncating Mutations', 'Deletion',
                       'Amplification', 'Fusions']:
        df.at[index, variation] = 1


def main():
    data_file = 'dataset/stage2_test_variants.csv'
    df = read_data(data_file)
    create_columns(df)

    current_gene = str()
    for index, row in df.iterrows():
        try:
            gene = row.Gene
            print('Working on gene: ' + str(gene) + '\n')

            if gene != current_gene:
                protein_record = fetch_protein_info(gene)
                current_gene = gene

            assign_mutation_location(df, index, protein_record)
        except urllib.error.HTTPError:
            print('No Uniprot info on gene: ' + str(gene) + '\n')
            continue
        except urllib.error.URLError:
            print('Outdated Uniprot identifier on gene: ' + str(gene) + '\n')
            continue

    df.to_csv('output_test.csv')


if __name__ == '__main__':
    main()

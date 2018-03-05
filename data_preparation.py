from Bio import SeqIO
from googlesearch import search
import mygene
import numpy as np
import pandas as pd
import re
import urllib.request


def read_data(data_file):
    df = pd.read_csv(data_file)
    return df


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
    data = urllib.request.urlopen("http://www.uniprot.org/uniprot/" + uniprot_id + ".xml")
    record = SeqIO.read(data, "uniprot-xml")
    return record


def assign_mutation_location(df, protein_record, gene):
    df['from_start'] = np.nan
    df['from_end'] = np.nan
    sequence_length = len(protein_record.seq)
    variation = df.loc[gene].Variation
    if variation.size > 1:

    mutation_index = re.findall(r'\d+', variation)
    if mutation_index:
        df.at[gene, 'from_start'] = mutation_index
        df.at[gene, 'from_end'] = sequence_length - mutation_index


def main():
    data_file = 'dataset/training_variants'
    df = read_data(data_file)
    for index, row in df.Gene.iterrows():
        print('Working on gene: ' + str(indexrr45) + '\n')
        protein_record = fetch_protein_info(gene)
        assign_mutation_location(df, protein_record, gene)
    df.to_csv('output.csv')


if __name__ == '__main__':
    main()

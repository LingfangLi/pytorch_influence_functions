#! /usr/bin/env python3
import logging
import sys

# append this as the system path
sys.path.append('InfluenceFunctions')
import pytorch_influence_functions as ptif
from Bert import load_model
from custom_data import *

if __name__ == "__main__":
    config = ptif.get_default_config()
    # load the pretrained transformer model
    model = load_model(3208, 3208)
    total_params = sum(p.numel() for p in model.parameters()) #numel:Returns the total number of elements in the input tensor.
    # it's the index of test sample will be used in calculation later
    sample_list = [7] # list, contain element 0

    # get the dataloader along with source text list and target text list.
    train_loader,train_src_text_list_original,train_trg_text_list_original= get_data_loader(TRAIN_NAME)
    test_loader,test_src_text_list_original,test_trg_text_list_original = get_data_loader(TEST_NAME)

    # set the directory for log file
    ptif.init_logging('D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\logfile.log')

    # Implement influence function
    influences = ptif.calc_img_wise(config, model, train_loader, test_loader, sample_list)

    harmful = influences[str(sample_list[0])]['harmful']
    helpful = influences[str(sample_list[0])]['helpful']
    logging.info('The most 4 helpful and harmful training inputs are:')
    for i in range(4):
        logging.info(f'{i}st harmful {train_src_text_list_original[harmful[i]]}')
        logging.info(f'{i}st helpful {train_src_text_list_original[helpful[i]]}')





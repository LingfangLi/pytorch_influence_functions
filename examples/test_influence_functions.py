#! /usr/bin/env python3
import logging
import sys
# append this as the system path
sys.path.append('InfluenceFunctions')
import pytorch_influence_functions as ptif
from Bert import load_model
from custom_data import *

def calculate_overlap(word_list1, word_list2):
    """计算两个词汇列表之间唯一重合的词的数量"""
    set1 = set(word_list1)
    set2 = set(word_list2)
    return len(set1.intersection(set2))

if __name__ == "__main__":
    config = ptif.get_default_config()
    # load the pretrained transformer model
    model = load_model(3208, 3208)
    total_params = sum(p.numel() for p in model.parameters()) #numel:Returns the total number of elements in the input tensor.
    # it's the index of test sample will be used in calculation later

    num_test_samples = 1000  # 假设有1000个测试样本
    sample_list = [i for i in range(num_test_samples)] # list, contain element 0

    # get the dataloader along with source text list and target text list.
    train_loader,train_src_text_list_original,train_trg_text_list_original= get_data_loader(TRAIN_NAME)
    test_loader,test_src_text_list_original,test_trg_text_list_original = get_data_loader(TEST_NAME)

    # set the directory for log file
    ptif.init_logging('D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\logfile.log')

    # Implement influence function
    influences = ptif.calc_img_wise(config, model, train_loader, test_loader, sample_list)

    # 初始化变量以存储总重合数和测试样本数
    total_overlap = 0

    for i in range(num_test_samples):
        # 对于每个测试样本，找到其最有帮助的3个样本
        helpful_indices = influences[str(i)]['helpful'][:3]

        test_sample_text = test_src_text_list_original[i].split()  # 分词假设
        overlap_count = 0

        for idx in helpful_indices:
            helpful_sample_text = train_src_text_list_original[idx].split()  # 分词假设
            overlap_count += calculate_overlap(test_sample_text, helpful_sample_text)
        print(f'overlap_count is : {overlap_count}')

        total_overlap += overlap_count

    # 计算平均重合词汇数
    average_overlap = total_overlap / num_test_samples
    logging.info(f'Average overlap: {average_overlap}')

    # harmful = influences[str(sample_list[0])]['harmful']
    # helpful = influences[str(sample_list[0])]['helpful']
    # logging.info('The most 4 helpful and harmful training inputs are:')
    # for i in range(4):
    #     logging.info(f'{i}st harmful {train_src_text_list_original[harmful[i]]}')
    #     logging.info(f'{i}st helpful {train_src_text_list_original[helpful[i]]}')





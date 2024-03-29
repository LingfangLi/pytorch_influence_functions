#! /usr/bin/env python3


import time
import datetime
import copy
import logging
from custom_data import *

from pathlib import Path
from pytorch_influence_functions.influence_function import s_test, grad_z
from pytorch_influence_functions.utils import save_json, display_progress


def calc_img_wise(config, model, train_loader, test_loader, sample_list):
    """Calculates the influence function one test point at a time. Calcualtes
    the `s_test` and `grad_z` values on the fly and discards them afterwards.

    Arguments:
        config: dict, contains the configuration from cli params"""
    # deepcopy creates a new and separate copy of an entire object or list with its own unique memory address.
    influences_meta = copy.deepcopy(config)
    test_sample_num = len(sample_list)  # 1
    test_start_index = config['test_start_index']  # 0
    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    # Set up logging and save the metadata conf file
    # logging.info(f"Running on: {test_sample_num} images per class.")
    logging.info(f"Starting at sentence: {test_start_index}.")
    influences_meta['test_sample_index_list'] = sample_list
    influences_meta_fn = f"influences_results_meta_{test_start_index}-" \
                         f"{test_sample_num}.json"
    influences_meta_path = outdir.joinpath(influences_meta_fn)
    save_json(influences_meta, influences_meta_path)

    # get the dataloader along with source text list and target text list.
    train_loader, train_src_text_list_original, train_trg_text_list_original = get_data_loader(TRAIN_NAME)
    test_loader, test_src_text_list_original, test_trg_text_list_original = get_data_loader(TEST_NAME)

    influences = {}
    # Main loop for calculating the influence function one test sample per iteration.
    harmful_array = []
    helpful_array = []
    test_array = []
    for j in range(len(sample_list)):
        i = sample_list[j]

        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            model, train_loader, test_loader, test_id_num=i, gpu=config['gpu'],
            damp=config['damp'], scale=config['scale'],
            recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        helpful_indices = helpful
        harmful_indices = harmful

        test_sample_text = test_src_text_list_original[j].split()  # get test sentences base on index j
        test_array.append(test_sample_text)
        helpful_sample_text = ""
        harmful_sample_text = ""
        # for k in range(3):
        # for idx1, idx2 in zip(helpful_indices,harmful_indices): # concatenate all the helpful training sentences but separate with ","
        for idx1 in harmful_indices[:3]:
            # concatenate all the helpful training sentences but separate with ","
            harmful_sample_text += harmful_sample_text + "----" + train_src_text_list_original[idx1]
        harmful_array.append(harmful_sample_text)

        for idx2 in helpful_indices[:3]:
            # concatenate all the helpful training sentences but separate with ","
            helpful_sample_text+= helpful_sample_text+ "----" + train_src_text_list_original[idx2]
            # harmful_sample_text += harmful_sample_text+","+train_src_text_list_original[idx2]  #
        helpful_array.append(helpful_sample_text)
        # after every 10 test samples, write 3 cloumns [test sample, cancatenated harmful sentences,
        # cancatenated harmful sentences] into csv files

        if j % 3 == 0:
            # create a file which have file that name has j, sentence, 10 samples
            with open(f'sentence_output{j}.txt', 'w', newline='') as f:
                for k in range(len(test_array)):
                    f.write(str(test_array[k]))
                    f.write("\t")
                    f.write(str(harmful_array[k]))
                    f.write("\t")
                    f.write(str(helpful_array[k]))
                    f.write("\n")
            f.close()
            harmful_array = []
            helpful_array = []
            test_array = []
        end_time = time.time()

        ###########
        # Different from `influence` above
        ###########
        influences[str(i)] = {}
        label = test_loader.dataset[i][-1].cpu().numpy().tolist()
        influences[str(i)]['label'] = label
        influences[str(i)]['num_in_dataset'] = j
        influences[str(i)]['time_calc_influence_s'] = end_time - start_time
        infl = [x.cpu().numpy().tolist() for x in influence]
        influences[str(i)]['influence'] = infl
        influences[str(i)]['harmful'] = harmful[:100]  # about the harmful, helpful, should i change or not
        influences[str(i)]['helpful'] = helpful[:100]

        tmp_influences_path = outdir.joinpath(f"influence_results_tmp_"
                                              f"{test_start_index}_"
                                              f"{test_sample_num}"
                                              f"_last-i_{i}.json")
        save_json(influences, tmp_influences_path)
        display_progress("Test samples processed: ", j, len(sample_list))

        logging.info(f"The results for test sample {i} are:")  # write into logfile.log
        # logging.info("Influences: ")
        # logging.info(influence)
        logging.info("Most harmful img IDs: ")
        logging.info(harmful[:3])
        logging.info("Most helpful img IDs: ")
        logging.info(helpful[:3])
        # logging.info("The helpful list is")
        # logging.info(len(helpful))
        # logging.info(helpful)

        influences_path = outdir.joinpath(f"influence_results_{test_start_index}_"
                                          f"{test_sample_num}.json")
        save_json(influences, influences_path)

    return influences


def calc_influence_single(model, train_loader, test_loader, test_id_num, gpu, damp, scale,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness，存的是 negative influence的那些 points 的idx？
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided.
    if not s_test_vec:  # list of torch tensor, contains s_test vectors.
        src_list, input_trg_list, e_mask, d_mask, t_test = test_loader.dataset[test_id_num]  # get specific test sample
        src_list = test_loader.collate_fn([src_list])  # collate_fn通常期望接收一个样本列表作为输入，并将这些样本组合成一个批次。
        # 这里，将src_list作为唯一的元素放在一个新列表中，意味着你想将src_list中的所有元素视为一个整体、一个批次进行处理。
        # Tensor->Tensor, size 200->1 tensor[20 elements]->tensor[[20 elements]]
        input_trg_list = test_loader.collate_fn([input_trg_list])
        e_mask = test_loader.collate_fn([e_mask])
        d_mask = test_loader.collate_fn([d_mask])
        t_test = test_loader.collate_fn([t_test])
        # s_test vector on single test sample
        s_test_vec = calc_s_test_single(model, src_list, input_trg_list, e_mask, d_mask, t_test, train_loader,
                                        gpu, damp, scale, recursion_depth=recursion_depth,
                                        r=r)  # list[tensor],size[50,512]
        # print(f's test vec is {s_test_vec}')
    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):  # for each training data point
        src_list, input_trg_list, e_mask, d_mask, t_train = train_loader.dataset[i]
        src_list = test_loader.collate_fn([src_list])
        input_trg_list = test_loader.collate_fn([input_trg_list])
        e_mask = test_loader.collate_fn([e_mask])
        d_mask = test_loader.collate_fn([d_mask])
        t_train = test_loader.collate_fn([t_train])
        if time_logging:
            time_a = datetime.datetime.now()
        # calculate this training sample's gradient from loss to parameters
        grad_z_vec = grad_z(src_list, input_trg_list, e_mask, d_mask, t_train, model,
                            gpu=gpu)  # list[tensor], size[50,512], loss相对于paramerters的gradient
        # print(grad_z_vec[5].shape) #
        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(f"Time for grad_z iter:"
                         f" {time_delta.total_seconds() * 1000}")
        tmp_influence = -sum(  # I(z, ztest)=−stest · ∇θL(zi, ˆ θ), # sum all the value after k and j operation
            [
                ####################
                # TODO: understand this equation and why grad_z_vec and s_test_vec have the same size
                ####################
                # extracts the tensor data from the sum operation. However, the use of .data is
                # discouraged in newer versions of PyTorch due to potential issues with autograd. Instead, using .item()
                # for single-element tensors or moving the tensor to CPU and converting it to a NumPy array (if necessary)
                # without using .data is recommended.
                ####################
                # TODO: k j
                ####################
                torch.sum(k * j).data
                # sum all the elements in k*j, [1,2] * [3,4]= [3,8], 3+8=11. size of k and j:Tensor(3208,512)(512,512)(512,)(2048,512)(2048,)(512,2048)
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size  # get average value of influence
        influences.append(tmp_influence.cpu())  # influences contain all the influence for all training data
        # display_progress("Calc. influence function: ", i, train_dataset_size)
    helpful = np.argsort(
        influences)  # This function call does not sort the influences array itself but returns an array
    # of indices. These indices are ordered such that, if used to index influences, the result would be a sorted array.
    harmful = helpful[::-1]
    return influences, harmful.tolist(), helpful.tolist(), test_id_num


# the arguments are from test data, so it's to calculate the Stest for test data
def calc_s_test_single(model, src_list, input_trg_list, e_mask, d_mask, t_test, train_loader, gpu=1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params)). Is it like average the s_test

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    ################################
    # TODO: r*recursion_depth should equal the training dataset size. How the set r and recursion_depth.
    ################################
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    # append the s_test vector of the test sample
    for i in range(r):
        s_test_vec_list.append(s_test(src_list, input_trg_list, e_mask, d_mask, t_test, model, train_loader,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))  # append the hvp
        # display_progress("Averaging r-times: ", i, r)

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################

    # calculate the average s_test_vec through iteration(s)
    s_test_vec = s_test_vec_list[0]  # size 50
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]
    # print("s_test_vec: ", s_test_vec)
    return s_test_vec
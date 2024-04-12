import os
import time
import heapq
import torch
import torch.nn as nn
import pickle
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
import json
import random
from .ablate import AblateGPT
import heapq
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors




def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def visualize_weights(membership_matrix, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(20, 20))
    cmap = mcolors.ListedColormap(['transparent', 'transparent', 'transparent', 'red'])  # Adjusted color for each category
    im = plt.imshow(membership_matrix, cmap=cmap, aspect='equal')
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Neither', 'P only', 'Q only', 'Both'])  # Correctly label each category
    plt.title("Membership of Elements in P, Q, and Both")
    plt.xlabel("Index Dimension 2")
    plt.ylabel("Index Dimension 1")
    plt.show()

def visualize_weights_binary(weight_matrix, mask, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Assuming weight_matrix is 2D (for simplicity)
    # Generate a binary matrix: 1s for unpruned weights, 0s for pruned
    binary_matrix = np.ones_like(weight_matrix)
    boolean_mask = mask.astype(bool)
    binary_matrix[boolean_mask] = 0  # Mark pruned weights as 0
    
    plt.figure(figsize=(20, 20))
    plt.imshow(binary_matrix, cmap='gray', interpolation='none')
    plt.title(f"Neuron Score Comparing {filename.split('/')[-1][:-4]}")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def append_pruning_details_to_file(filename, layer_name, prune_percentage, image_path):
    with open(filename, "a") as file:
        file.write(f"Layer: {layer_name}, Prune Percentage: {prune_percentage}, Image: {image_path}\n")

def get_mask(filtered_indices, weight_dim, W_mask):
    filtered_indices_rows = filtered_indices // weight_dim
    filtered_indices_cols = filtered_indices % weight_dim
    W_mask[filtered_indices_rows, filtered_indices_cols] = (
                    True  # prune weights that has relatively high safety while not in top utility scores
                )
    return W_mask

def find_overlap(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align_short",
    p=0.1,
    q=0.1,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    print(
        "compare when p = {}, q = {}, with data = {}".format(
            p, q, prune_data
        )
    )
    if args.prune_part:
        print("only compare the layer with low jaccard index")
    else:
        print("compare every linear layer")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if not args.prune_part:
            for name in subset:
                print(f"comparing in layer {i} name {name}")
                if args.model == "llama2-7b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{prune_data}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    tmp = prune_data
                    prune_data = "alpaca_cleaned_no_safety"
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wanda/{prune_data}/wanda_score/{prune_data}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    )
                    prune_data = tmp
                elif args.model == "llama2-13b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{prune_data}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wanda/{prune_data}/wanda_score/{prune_data}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    )
                else:
                    raise NotImplementedError

                assert (
                    W_metric1.shape == W_metric2.shape
                ) 

                top_p = int(
                    p * W_metric1.shape[1] * W_metric1.shape[0]
                )  # top_p utility
                top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])  # top_q safety

                top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
                unique_p = torch.unique(top_p_indices)
                unique_q = torch.unique(top_q_indices)
                union_indices = torch.unique(torch.cat((unique_q, unique_p), dim=0))

                # Create a boolean mask for elements both in unique_q and unique_p
                mask = torch.isin(unique_q, unique_p)
                # Mask for elements uniquely in unique_q (present in unique_q but not in unique_p)
                mask_unique_q = ~torch.isin(unique_q, unique_p)
                # Mask for elements uniquely in unique_p (present in unique_p but not in unique_q)
                mask_unique_p = ~torch.isin(unique_p, unique_q)
                
                weight_dim = subset[name].weight.data.shape[1]
                W_mask_union = get_mask(union_indices, weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))
                W_mask_p = get_mask(unique_p, weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))
                W_mask_q = get_mask(unique_q, weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))
                W_mask_inters = get_mask(unique_p[mask], weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))

                matrix = (torch.zeros_like(subset[name].weight.data) == 1).int()
                matrix[W_mask_p] = 1
                matrix[W_mask_q] = 2
                matrix[W_mask_inters] = 3

                # After calculating W_mask...
                visualization_filename = os.path.join(args.save, f"visualization_layer_{i}_{name}.png")
                details_filename = os.path.join(args.save, "pruning_details.txt")

                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim
                W_mask = torch.zeros_like(subset[name].weight.data) == 1
                W_mask[filtered_indices_rows, filtered_indices_cols] = (
                    True  # prune weights that has relatively high safety while not in top utility scores
                )
                
                # Visualize the weights and save the image
                visualize_weights_binary(subset[name].weight.data.cpu().to(torch.float32).numpy(), 
                                         W_mask.cpu().to(torch.float32).numpy(),  
                                         visualization_filename)

                # Calculate prune percentage
                overlap_percentage = W_mask.sum().item() / W_mask_union.sum().item() * 100

                # Append details to file
                append_pruning_details_to_file(details_filename, f"Layer {i} {name}", overlap_percentage, visualization_filename)

                # subset[name].weight.data[W_mask] = 0  ## set weights to zero

def find_overlap_test(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align",
    prune_data_2="alpaca_cleaned_no_safety",
    p=0.1,
    q=0.1,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    print(
        "compare when p = {}, q = {}, between data = {} and {}".format(
            p, q, prune_data, prune_data_2
        )
    )
    if args.prune_part:
        print("only compare the layer with low jaccard index")
    else:
        print("compare every linear layer")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if not args.prune_part:
            for name in subset:
                print(f"comparing in layer {i} name {name}")
                if args.model == "llama2-7b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{prune_data}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wanda/{prune_data_2}/wanda_score/{prune_data_2}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{prune_data_2}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    )
                elif args.model == "llama2-13b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{prune_data_2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wanda/{prune_data_2}/wanda_score/{prune_data_2}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{prune_data_2}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    )
                else:
                    raise NotImplementedError

                assert (
                    W_metric1.shape == W_metric2.shape
                ) 

                top_p = int(
                    p * W_metric1.shape[1] * W_metric1.shape[0]
                )  # top_p utility
                top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])  # top_q safety

                top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
                unique_p = torch.unique(top_p_indices)
                unique_q = torch.unique(top_q_indices)
                union_indices = torch.unique(torch.cat((unique_q, unique_p), dim=0))

                # Create a boolean mask for elements both in unique_q and unique_p
                mask = torch.isin(unique_q, unique_p)
                # Mask for elements uniquely in unique_q (present in unique_q but not in unique_p)
                mask_unique_q = ~torch.isin(unique_q, unique_p)
                # Mask for elements uniquely in unique_p (present in unique_p but not in unique_q)
                mask_unique_p = ~torch.isin(unique_p, unique_q)
                
                weight_dim = subset[name].weight.data.shape[1]
                W_mask_union = get_mask(union_indices, weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))
                W_mask_p = get_mask(unique_p, weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))
                W_mask_q = get_mask(unique_q, weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))
                W_mask_inters = get_mask(unique_p[mask], weight_dim, (torch.zeros_like(subset[name].weight.data) == 1))

                matrix = (torch.zeros_like(subset[name].weight.data) == 1).int()
                matrix[W_mask_p] = 1
                matrix[W_mask_q] = 2
                matrix[W_mask_inters] = 3

                # After calculating W_mask...
                visualization_filename = os.path.join(args.save, f"visualization_layer_{i}_{name}.png")
                details_filename = os.path.join(args.save, "pruning_details.txt")

                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim
                W_mask = torch.zeros_like(subset[name].weight.data) == 1
                W_mask[filtered_indices_rows, filtered_indices_cols] = (
                    True  # prune weights that has relatively high safety while not in top utility scores
                )
                
                # Visualize the weights and save the image
                visualize_weights_binary(subset[name].weight.data.cpu().to(torch.float32).numpy(), 
                                         W_mask.cpu().to(torch.float32).numpy(),  
                                         visualization_filename)

                # Calculate prune percentage
                overlap_percentage = W_mask.sum().item() / W_mask_union.sum().item() * 100

                # Append details to file
                append_pruning_details_to_file(details_filename, f"Layer {i} {name}", overlap_percentage, visualization_filename)

                # subset[name].weight.data[W_mask] = 0  ## set weights to zero


def find_distribution(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align",
    p=0.1,
    q=0.1,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    print(
        "find distribution from 1% to 100% in 100 steps in data = {}".format(
            p, q, prune_data
        )
    )
    if args.prune_part:
        print("only investigate the layer with low jaccard index")
    else:
        print("investigate every linear layer")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if not args.prune_part:
            for name in subset:
                print(f"investigating in layer {i} name {name}")
                if args.model == "llama2-7b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{prune_data}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wanda/{prune_data}/wanda_score/{prune_data}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    )
                elif args.model == "llama2-13b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{prune_data}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wanda/{prune_data}/wanda_score/{prune_data}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    )
                else:
                    raise NotImplementedError

                assert (
                    W_metric1.shape == W_metric2.shape
                ) 

                
                # # Prepare the range of percentages to test
                # percentage_range = np.linspace(0.01, 1, 100)  # From 1% to 100% in 100 steps

                # # Lists to store the number of top elements for each percentage
                # top_p_counts = []
                # top_q_counts = []

                # # Calculate the number of top elements for each percentage
                # for percentage in percentage_range:
                #     top_p = int(percentage * W_metric1.numel())
                #     top_q = int(percentage * W_metric2.numel())

                #     top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True).indices
                #     top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True).indices

                #     # Record the count of unique indices (neurons)
                #     unique_p = torch.unique(top_p_indices)
                #     unique_q = torch.unique(top_q_indices)

                #     top_p_counts.append(len(unique_p))
                #     top_q_counts.append(len(unique_q))

                # # Now plot the distribution of top elements as a function of the percentage
                # plt.figure(figsize=(12, 6))

                visualization_filename = os.path.join(args.save, f"distribution_layer_{i}_{name}.png")

                # plt.plot(percentage_range, top_p_counts, label='snip score', marker='o')
                # plt.plot(percentage_range, top_q_counts, label='wanda score', marker='x')

                # plt.title('Distribution of Neurons by Percentage')
                # plt.xlabel('Percentage of Total Elements')
                # plt.ylabel('Number of Unique Top Elements')
                # plt.legend()

                # Flatten the tensors to get the distributions as 1D arrays
                flat_W_metric1 = W_metric1.flatten().to(torch.float32).cpu().numpy()
                flat_W_metric2 = W_metric2.flatten().to(torch.float32).cpu().numpy()
                
                # Get min and max values for each matrix
                min_val_1, max_val_1 = flat_W_metric1.min(), flat_W_metric1.max()
                min_val_2, max_val_2 = flat_W_metric2.min(), flat_W_metric2.max()

                # Set up thresholds as proportions of the range from min to max
                thresholds_1 = np.linspace(min_val_1, max_val_1, 100)
                thresholds_2 = np.linspace(min_val_2, max_val_2, 100)

                # Calculate the percentage of values above each threshold
                percentages_1 = [np.mean(flat_W_metric1 > threshold) for threshold in thresholds_1]
                percentages_2 = [np.mean(flat_W_metric2 > threshold) for threshold in thresholds_2]

                # Plot the distribution curves
                plt.figure(figsize=(12, 6))
                # Set up two subplots
                fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # Two rows, one column

                # Plot for W_metric1
                axes[0].semilogy(thresholds_1, percentages_1, label='W_metric1', marker='o', color='blue')
                axes[0].set_title(f'Percentage of Neuron Values Above Thresholds for W_metric1')
                axes[0].set_xlabel('Value Threshold')
                axes[0].set_ylabel('Percentage Above Threshold')
                axes[0].legend()
                axes[0].grid(True)

                # Plot for W_metric2
                axes[1].semilogy(thresholds_2, percentages_2, label='W_metric2', marker='x', color='orange')
                axes[1].set_title(f'Percentage of Neuron Values Above Thresholds for W_metric2')
                axes[1].set_xlabel('Value Threshold')
                axes[1].set_ylabel('Percentage Above Threshold')
                axes[1].legend()
                axes[1].grid(True)

                plt.tight_layout()
                plt.show()
                os.makedirs(os.path.dirname(visualization_filename), exist_ok=True)
                plt.savefig(visualization_filename)
                plt.close()

                # Append details to file
                # append_pruning_details_to_file(details_filename, f"Layer {i} {name}", overlap_percentage, visualization_filename)
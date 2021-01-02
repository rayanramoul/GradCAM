import logging

import sklearn.metrics
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models

from tools.snippets import quick_log_setup
from tools.voc import (
        VOC_CLASSES, VOC_ocv, enforce_all_seeds,
        transforms_voc_ocv_eval, sequence_batch_collate_v2)

log = logging.getLogger(__name__)


def evaluate_voc_classifier():
    """
    Predict class labels images in the VOC2007 testing set, display Average
    Precision stats
    """
    # / Config
    initial_seed = 42
    # Number of processes for data loading. Can be set to 0 for easier debugging
    num_workers = 4
    # This folder will be used to save VOC2007 dataset
    voc_folder = 'voc_dataset'
    # Path to the classification model trained on VOC2007 trainset
    inputs_ckpt = 'model_at_epoch_019.pth.tar'

    # Dataset and Dataloader to quickly access the VOC2007 data
    dataset_test = VOC_ocv(
            voc_folder, year='2007', image_set='test',
            download=True, transforms=transforms_voc_ocv_eval)
    dataloader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=32,
            shuffle=False, num_workers=num_workers,
            collate_fn=sequence_batch_collate_v2)

    # Define resnet50 model, load the imagenet weights
    device =  "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_classes = 20
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    model.to(device)

    # Fix seed
    enforce_all_seeds(initial_seed)

    # Load the finetuned model
    states = torch.load(inputs_ckpt, map_location=device)
    model.load_state_dict(states['model_sdict'])

    # Obtain predictions
    targets = []
    outputs = []
    for i_batch, (data, target, meta) in enumerate(tqdm(dataloader_test)):
        data, target = map(lambda x: x.to(device), (data, target))
        with torch.no_grad():
            output = model(data)
            output_sigm = torch.sigmoid(output)
            output_np = output_sigm.detach().cpu().numpy()
            targets.append(target.cpu())
            outputs.append(output_np)
    targets = np.vstack(targets)
    outputs = np.vstack(outputs)

    # Compute average precision scores
    aps = {}
    for label, X, Y in zip(VOC_CLASSES, outputs.T, targets.T):
        aps[label] = sklearn.metrics.average_precision_score(Y, X)
    aps['MEAN'] = np.mean(list(aps.values()))
    s = pd.Series(aps)*100
    log.info('Multilabel performance (AP):\n{}'.format(s))


if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    evaluate_voc_classifier()

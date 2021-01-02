import logging

from tools.snippets import (quick_log_setup, mkdir)
from tools.voc import (VOC_ocv, transforms_voc_ocv_eval)
from tools.dogs import (
        eval_stats_at_threshold, read_metadata,
        produce_gt_dog_boxes, produce_fake_centered_dog_boxes,
        visualize_dog_boxes)


def dog_detection():
    """
    Here are two simple baselines for dog detection:
      - predicting full frame as a dog
      - predicting a box in the center of the frame (50% of the area)

    Two helper functions here:
      - a function that evaluates Average Precision and Recall
      - a function that visualizes predictions and ground truth
    """
    # / Config
    # This folder will be used to save VOC2007 dataset
    voc_folder = 'voc_dataset'

    # Dataset and Dataloader to quickly access the VOC2007 data
    dataset_test = VOC_ocv(
            voc_folder, year='2007', image_set='test',
            download=True, transforms=transforms_voc_ocv_eval)

    # Load metadata and create GT boxes
    metadata_test = read_metadata(dataset_test)
    all_gt_dogs = produce_gt_dog_boxes(metadata_test)

    # / Baselines
    # Produce fake fullsized dogs
    all_detected_dogs_fullsize = produce_fake_centered_dog_boxes(
            metadata_test, scale=1)
    stats_df = eval_stats_at_threshold(
            all_detected_dogs_fullsize, all_gt_dogs)
    log.info('FULLBOX dogs:\n{}'.format(stats_df))

    # Produce fake centerboxd dogs
    all_detected_dogs_cbox = produce_fake_centered_dog_boxes(
            metadata_test, scale=0.5)
    stats_df = eval_stats_at_threshold(
            all_detected_dogs_cbox, all_gt_dogs)
    log.info('CENTERBOX dogs:\n{}'.format(stats_df))

    # Visualize the boxes
    fold = mkdir('visualized_dog_centerboxes')
    visualize_dog_boxes(fold,
            all_detected_dogs_cbox, all_gt_dogs, metadata_test)


if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    dog_detection()

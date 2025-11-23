# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 

    from ensurepip import version
    from pathlib import Path
    from uuid import uuid4
    import rerun as rr
    return Path, mo, rr, uuid4


@app.cell
def _(mo):
    # Show the mapillary log for illustration :)

    _src = ("https://static.wixstatic.com/media/c2f756_ed542bdf6e054a3790b79c16cb3297ec~mv2.webp/v1/fill/w_680,h_382,al_c,q_80,usm_0.66_1.00_0.01,enc_avif,quality_auto/Image-empty-state_edited.webp"
    )
    mo.image(src=_src, width="180px", height="180px", rounded=True)
    return


@app.cell
def _(Path):
    # Inputs
    # You need to add the path of mapillary dataset 

    DATASET_ROOT = Path(__file__).parent / 'mapillary_dataset'  # Change this to your dataset path
    VERSION = 'v2.0'
    return DATASET_ROOT, VERSION


@app.cell
def _(rr, uuid4):
    rr.init(application_id=str(uuid4()), spawn=True)
    return


@app.cell
def _():
    # In this file, it shows how to load the Mapillary dataset for Instance segmentation
    """
    PyTorch Dataset for Mapillary Vistas Instance Segmentation
    """
    import os
    import json
    import numpy as np
    import torch
    from torch.utils.data import Dataset, random_split
    from PIL import Image
    from torchvision.ops.boxes import masks_to_boxes
    from torchvision import tv_tensors
    from torchvision.transforms.v2 import functional as F
    import glob

    class MapillaryInstanceDataset(Dataset):
        """
        Mapillary Vistas Dataset for Instance Segmentation
        """
        def __init__(self, root_dir, version='v2.0', split='validation', transforms=None):
            """
            Args:
                root_dir: Root directory containing the dataset
                version: Dataset version ('v1.2' or 'v2.0')
                split: 'training' or 'validation'
                transforms: Transformations to apply
            """
            self.root_dir = root_dir
            self.version = version
            self.split = split
            self.transforms = transforms

            # Load config file to get label information
            config_path = os.path.join(root_dir, f'config_{version}_testing.json')
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get the labels
            self.labels = config['labels']

            # Check think link to understand thing and classes:
            # https://www.basic.ai/blog-post/comprehensive-guide-of-panoptic-segmentation

            # Get only "thing" classes (classes with instances)
            self.thing_classes = [label for label in self.labels if label["instances"]]
            self.label_id_to_class_id = {}

            # Add mapping to colors for visualization later and to be unique for instances 

            self.label_id_to_color = {}

            # Create mapping from original label_id to class_id (1-indexed for PyTorch)
            # Class 0 is reserved for background
            thing_class_counter = 0
            for label_id, label in enumerate(self.labels):
                if label["instances"]:
                    thing_class_counter += 1
                    self.label_id_to_class_id[label_id] = thing_class_counter
                    self.label_id_to_color[label_id] = label["color"]

            # Get all image IDs
            images_dir = os.path.join(root_dir, split, 'images')
            self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
            self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]

            print(f"Loaded {len(self.image_ids)} images from {split} split")
            print(f"Number of thing classes (with instances): {len(self.thing_classes)}")

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            # Get image ID
            image_id = self.image_ids[idx]

            # Load image
            image_path = os.path.join(self.root_dir, self.split, 'images', f'{image_id}.jpg')
            image = Image.open(image_path).convert('RGB')

            # Load instance image
            instance_path = os.path.join(self.root_dir, self.split, self.version, 'instances', f'{image_id}.png')
            instance_image = Image.open(instance_path)
            instance_array = np.array(instance_image, dtype=np.uint16)

            # Split instance_array into labels and instance IDs
            # instance_array encoding: label_id = value / 256, instance_id = value % 256
            # instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
            # instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

            # Get unique instances (combination of label and instance ID)
            unique_instances = np.unique(instance_array)

            # Filter out background (0) and process each instance
            masks_list = []
            labels_list = []
            # boxes_list = []
            colors_list = []

            # In __getitem__, add validation:

            for inst_value in unique_instances:
                if inst_value == 0:
                    continue

                # label_id = inst_value // 256
                # instance_id = inst_value % 256

                label_id = inst_value // 256
                # instance_id = np.array(inst_value % 256, dtype=np.uint8)

                # Validate label_id is within bounds
                if label_id >= len(self.labels):
                    print(f"Warning: label_id {label_id} out of bounds (total labels: {len(self.labels)})")
                    continue

                # Only keep instances from "thing" classes
                if not self.labels[label_id]["instances"]:
                    continue

                # Create binary mask for this instance
                mask = (instance_array == inst_value).astype(np.uint8)

                masks_list.append(mask)

                # Get class ID using the mapping
                class_id = self.label_id_to_class_id.get(label_id)
                labels_list.append(class_id)

                # Get the color for visualization later
                color_itm = self.label_id_to_color.get(label_id, [0, 0, 0])
                colors_list.append(color_itm)


            # Handle case with no instances
            if len(masks_list) == 0:
                # Create dummy data for images without instances
                masks = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
            else:
                # Convert to tensors
                masks = np.stack(masks_list, axis=0)
                masks = torch.as_tensor(masks, dtype=torch.uint8)

                # Get bounding boxes from masks
                boxes = masks_to_boxes(masks)

                labels = torch.as_tensor(labels_list, dtype=torch.int64)

                # Calculate area
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Convert image to tensor
            image = F.to_image(image)

            # Prepare target dictionary
            target = {}
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
            target["masks"] = tv_tensors.Mask(masks)
            target["labels"] = labels
            target["colors"] = colors_list
            target["image_id"] = idx
            target["area"] = area
            target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

            # expose original file paths so callers (DataLoader) can access them ---
            target["image_path"] = image_path            # full image path string
            target["instance_path"] = instance_path      # instance png path string

            # Apply transformations
            if self.transforms is not None:
                image, target = self.transforms(image, target)

            return image, target

        def get_num_classes(self):
            """Returns number of classes including background"""
            return len(self.thing_classes) + 1  # +1 for background
    return Image, MapillaryInstanceDataset, json, np, os, torch


@app.cell
def _(torch):
    #%%
    """
    Data Splitting and DataLoader Creation
    """

    from torchvision.transforms import v2 as T

    def get_transform(train):
        """
        """
        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            # Add more augmentations if needed
            # transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)
    return (get_transform,)


@app.cell
def _(DATASET_ROOT, MapillaryInstanceDataset, VERSION, get_transform):
    #%%
    """
    Usage Example: Here you can understand how to load the dataset for instance segmentaion
    """

    # Create training dataset and split into train/test
    train_full = MapillaryInstanceDataset(
        root_dir= DATASET_ROOT,
        version=VERSION,
        split='training',
        transforms=get_transform(train=True)
    )
    train_total = len(train_full)

    print(f"\nTraining folder samples: {train_total}")


    # Create validation dataset and split into val/test (both come from 'validation' folder)
    val_full = MapillaryInstanceDataset(
        root_dir=DATASET_ROOT,
        version=VERSION,
        split='validation',
        transforms=get_transform(train=False)
    )
    val_total = len(val_full)

    print(f"\nValidation folder samples: {val_total}")


    def collate_fn(batch):
        return tuple(zip(*batch))


    num_classes = train_full.get_num_classes()

    print(f"\nNumber of classes (including background): {num_classes}")
    return train_full, val_full


@app.cell
def _(DATASET_ROOT, VERSION, json, os):
    #%%
    """
    Helper function to get class names
    """

    def get_class_names(root_dir, version='v2.0'):
        """
        Get list of class names (thing classes only)
        Returns list where index 0 is background, index 1+ are thing classes
        """
        config_path = os.path.join(root_dir, f'config_{version}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        labels = config['labels']
        class_names = ['__background__']
        class_colors = [[0, 0, 0]]  # Background color (black)

        for label in labels:
            if label["instances"]:
                class_names.append(label["readable"])
                class_colors.append(label["color"])

        return class_names, class_colors

    # Get class names
    class_names, class_colors = get_class_names(DATASET_ROOT, VERSION)
    print(f"\nClass names ({len(class_names)} total):")
    print(f"\nClass colors ({len(class_colors)} total):")
    for j, color in enumerate(class_colors):
        print(f"  {j}: {color}")

    limit_classes = None
    for k, name in enumerate(class_names[:limit_classes]):  # Print first 10
        print(f"  {k}: {name}")

    for class_name_itm, class_color_itm, class_id_itm in zip(class_names, class_colors, range(len(class_names))):
        print(f"Class ID: {class_id_itm}, Name: {class_name_itm}, Color: {class_color_itm}")
    return (class_names,)


@app.cell
def _(np):
    import matplotlib.pyplot as plt
    def visualize_instance_segmentation(image_, target, class_names=None):
        """
        Visualize image with instance masks overlayed.
        Args:
            image_: Tensor, shape (C, H, W)
            target: Dict with 'masks', 'boxes', 'labels'
            class_names: Optional, list of class names (for legend)
        """
        # Convert image tensor to numpy (H, W, C)
        img_np = image_.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_np, origin='upper')
        ax = plt.gca()

        # Overlay each instance mask
        n_instances = target["masks"].shape[0]
        for i in range(n_instances):
            mask = target["masks"][i].cpu().numpy()
            label = int(target["labels"][i])
            # Normalize color from [0, 255] to [0, 1]
            color = np.array(target["colors"][i]) / 255.0

            # Mask is boolean, overlay semi-transparent
            masked_img = np.zeros((*mask.shape, 4), dtype=np.float32)
            masked_img[..., :3] = color
            masked_img[..., 3] = 0.4 * mask  # Transparency

            ax.imshow(masked_img, interpolation="none")

            # Draw bounding box
            box = target["boxes"][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=0.5)
            ax.add_patch(rect)

            # Put label
            if class_names:
                cls_name = class_names[label]
            else:
                cls_name = str(label)
            ax.text(x1, y1, cls_name, color='white', fontsize=5, bbox=dict(facecolor=tuple(color), alpha=0.7))

        plt.axis('off')
        plt.show()


        print("Show the image using matplotlib :) ")
    return (visualize_instance_segmentation,)


@app.cell
def _(Image, np, os, rr):
    def visualize_instance_segmentation_rr(image, target, class_names=None):
        """
        Visualize image with instance masks overlayed in Rerun.
        Args:
            image: Tensor, shape (C, H, W)
            target: Dict with 'masks', 'boxes', 'labels'
            class_names: Optional, list of class names (for legend)
        """

        # clean up previous logs from rerun
        rr.log("image", rr.Clear(recursive=True)) 

        # Convert tensor to numpy and scale to [0, 255]
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # Changed: scale to 0-255

        # Log image directly from tensor (rerun handles the display)
        rr.log("image", rr.Image(img_np, opacity=0.5))

        # Overlay each instance mask
        n_instances = target["masks"].shape[0]

        for i in range(n_instances):
            label = int(target["labels"][i])

            if class_names:
                cls_name = class_names[label]
            else:
                cls_name = f"Class {label}"


            # Normalize color from [0, 255] to [0, 1]
            color = np.array(target["colors"][i]) / 255.0

            mask = target["masks"][i].cpu().numpy()

            # Mask is boolean, overlay semi-transparent
            masked_img = np.zeros((*mask.shape, 4), dtype=np.float32)
            masked_img[..., :3] = color
            masked_img[..., 3] = 0.4 * mask  # Transparency

            # Draw bounding box
            box = target["boxes"][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)

            masked_img_PIL = Image.fromarray((masked_img * 255).astype(np.uint8))
            temp_masked_img_path = f"temp_masked_img_{i}.png"
            masked_img_PIL.save(temp_masked_img_path)
            rr.log(
                f"image/instances/{i}/masked_img",
                rr.EncodedImage(path=temp_masked_img_path, opacity=0.7),
                )
            os.remove(temp_masked_img_path)


            # log the boxes one by one to be able to see them in rerun
            rr.log(
                    f"image/instances/{i}/boxes2d",
                    rr.Boxes2D(
                        array=np.array([x1, y1, x2, y2]),
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[cls_name],
                        colors=[tuple(color)], # Convert back to 0-255
                        class_ids=[label],
                        show_labels=True,
                    )
                )
    return (visualize_instance_segmentation_rr,)


@app.cell
def _(visualize_instance_segmentation, visualize_instance_segmentation_rr):
    # how to get an image user friendly visualizer using its name marimo

    def visualize_image_by_name_rr(image_name_id, dataset, class_names):
        """
        Retrieve and visualize a specific image by its name ID.

        Args:
            image_name_id: String, the image filename without extension (e.g., 'image_001')
            dataset: MapillaryInstanceDataset instance
            class_names: List of class names
        """
        # Find the index of the image with matching ID
        if image_name_id not in dataset.image_ids:
            print(f"Error: Image '{image_name_id}' not found in dataset")
            print(f"Available images: {dataset.image_ids[:5]}... ({len(dataset.image_ids)} total)")
            return None

        # Get the index
        idx = dataset.image_ids.index(image_name_id)

        # Retrieve the image and target
        image, target = dataset[idx]

        # Visualize using rerun
        visualize_instance_segmentation_rr(image, target, class_names)
        visualize_instance_segmentation(image, target, class_names)
    return (visualize_image_by_name_rr,)


@app.cell
def _(mo):
    # Split selection
    dropdown_split = mo.ui.dropdown(
        value='training',
        options=['training', 'validation'],
        label="Choose split"
    )
    return (dropdown_split,)


@app.cell
def _(dropdown_split):
    dropdown_split
    return


@app.cell
def _(dropdown_split, mo, train_full, val_full):
    # Image options, reactive to split
    all_image_names_train = train_full.image_ids
    all_image_names_validation = val_full.image_ids

    image_options = (
        all_image_names_train if dropdown_split.value == "training"
        else all_image_names_validation
    )

    dropdown_image = mo.ui.dropdown(options=image_options, label="Choose image")
    return (dropdown_image,)


@app.cell
def _(dropdown_image):
    dropdown_image
    return


@app.cell
def _(
    class_names,
    dropdown_image,
    dropdown_split,
    train_full,
    val_full,
    visualize_image_by_name_rr,
):
    # Visualization, reactive to split and image

    def get_dataset(split):
        return train_full if split == "training" else val_full

    if dropdown_image.value != None:
        visualization = visualize_image_by_name_rr(
            dropdown_image.value,
            get_dataset(dropdown_split.value),
            class_names
        )
    return


@app.cell
def _():
    # Done, Yooooeeeee :) 
    return


if __name__ == "__main__":
    app.run()

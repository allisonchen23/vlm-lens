import numpy as np
import os
import random
import sys

sys.path.insert(0, '../src')
sys.path.insert(0, '..')
from main import get_model
from models.config import Config, IMAGE_TOKEN_IDS
import db_utils, utils, visualizations

# %% [markdown]
# ## Obtain paths for non-face TLL image pairs

# %%
def filter_images(data_dir,
                  visualize=False,
                  save_dir=None,
                  overwrite=False):
    if save_dir is not None:
        right_save_path = os.path.join(save_dir, "filtered_right_paths.txt")
        left_save_path = os.path.join(save_dir, "filtered_left_paths.txt")
        if os.path.exists(right_save_path) and os.path.exists(left_save_path) and not overwrite:
            utils.informal_log("File exists at {} and {} and not overwriting".format(
                left_save_path, right_save_path))
            return utils.read_file(left_save_path), utils.read_file(left_save_path)

    metadata_path = os.path.join(data_dir, "metadata.pkl")
    metadata = utils.read_file(metadata_path)

    # Map image names from Images/ to right/left names
    paired_image_names = metadata['image_list']
    single_image_names = sorted(os.listdir(os.path.join(data_dir, "right")))
    paired_single_dict = dict(zip(paired_image_names, single_image_names))

    # Get list of image names that are in "no_faces"
    filtered_boolean = metadata['no_faces']
    filtered_paired_image_names = np.array(paired_image_names)[filtered_boolean]
    # Get the corresponding right/left image names
    filtered_single_image_names = [paired_single_dict[paired_name] for paired_name in filtered_paired_image_names]
    assert len(filtered_paired_image_names) == len(filtered_single_image_names)
    if visualize:
        rand_int = random.randint(0, len(filtered_single_image_names) - 1)
        utils.informal_log("Randomly visualizing image {}".format(rand_int))
        visualizations.show_image_rows(
            [[utils.read_file(os.path.join(data_dir, "Images", filtered_paired_image_names[rand_int])),
            utils.read_file(os.path.join(data_dir, "left", filtered_single_image_names[rand_int])),
            utils.read_file(os.path.join(data_dir, "right", filtered_single_image_names[rand_int]))]]
        )

    # Make separate lists for right and left
    left_save_paths = []
    right_save_paths = []
    for filename in filtered_single_image_names:
        left_save_paths.append(os.path.join(data_dir, "left", filename))
        right_save_paths.append(os.path.join(data_dir, "right", filename))
    if save_dir is not None:
        utils.write_file(left_save_paths, left_save_path, overwrite=overwrite)
        utils.write_file(right_save_paths, right_save_path, overwrite=overwrite)

    return left_save_paths, right_save_paths


# %%
# data_dir = "../data_local/tll/totally_looks_like"
# save_dir = "../data_local/tll"

# image_names = filter_images(
#     data_dir=data_dir,
#     visualize=True,
#     save_dir=save_dir,
#     overwrite=False)

# %% [markdown]
# ## Run Model on Left and Right Images

# %% [markdown]
# ### Left Images

# %%
def run_model(config_path):
    sys.argv = ['notebooks/get_representations.ipynb',
                '--config', config_path]

    config = Config()

    # %%
    model = get_model(config.architecture, config)


    n_modules = 0
    layer_names = []
    for name, module in model.model.named_modules():
        if model.config.matches_module(name):
            print(name)
            layer_names.append(name)
            n_modules += 1
    utils.informal_log("{} modules matched".format(n_modules))

    # %%
    # Run model -- first checking if we would overwrite anything
    db_path = model.config.output_db
    utils.informal_log("Database path: {}".format(db_path))
    proceed = True
    if os.path.exists(db_path):
        proceed = False
        # response = input("File exists at {}. Are you sure you want to overwrite? (Y/N)".format(db_path))
        # if response.lower() != "y":
        #     proceed = False
        # else:
        #     os.remove(db_path)

    if proceed:
        # Run model on images
        model.run(save_tokens=True)
    else:
        utils.informal_log("Not overwriting file at {}".format(db_path))

    db_utils.save_embeddings_npy(
        db_path=db_path,
        layer_names=layer_names,
        overwrite=False)


utils.informal_log("Running model on Left Images of TLL")
run_model(
    config_path="../configs/models/qwen/Qwen2-VL-7B-Instruct-TLL-Left.yaml"
)

utils.informal_log("Running model on Right Images of TLL")
run_model(
    config_path="../configs/models/qwen/Qwen2-VL-7B-Instruct-TLL-Right.yaml"
)
# Library imports
import numpy as np
from tqdm import tqdm

# Local imports
import db_utils
from models.config import IMAGE_TOKEN_IDS

AVAILABLE_MODALITIES = ['text', 'vision', 'text+vision']
def compute_image_pair_similarities(database_path,
                                    model,
                                    layer_names=None,
                                    modalities=['visual']):
    '''
    For each layer, compute the image pair wise similarity matrices.
    Assumes we are taking the mean embedding value.

    Arg(s):
        database_path : str
        model : models.base.ModelBase
        layer_names : list[str] or None
        extract_visual_tokens : bool

    Returns:
        tuple(list[str], list[np.array], list[float]) : layer_names, embeddings, and similarities
    '''
    # Check for valid modalities
    assert len(modalities) > 0
    for mode in modalities:
        if mode not in AVAILABLE_MODALITIES:
            raise ValueError("Modality {} not supported. Try one of {}".format(mode, AVAILABLE_MODALITIES))
        if mode == "text":
            raise ValueError("Modality {} currently not yet supported. T.T Need to get on it!".format(mode))

    # Get input_ids if we need to separate image and text
    if "vision" in modalities or "text" in modalities:
        input_ids = db_utils.get_embeddings_by_layer(
            db_path=database_path,
            layer_name="input_ids",
            device="cuda")
        input_ids, _ = db_utils.unwrap_embeddings(input_ids) # empty value is input_ids_same_shapes

    # If layer_names is None, assume all layer names in config are desired
    if layer_names is None:
        layer_names = []
        for name, _ in model.model.named_modules():
            if model.config.matches_module(name):
                layer_names.append(name)

    # Return values
    module_embeddings = []
    module_similarities = []
    module_names = []
    # Get embeddings for each layer and compute image-pair similarity scores
    for layer_name in tqdm(layer_names):
        if not model.config.matches_module(layer_name):
            print("{} not present in config".format(layer_name))
            continue

        # This may take a long time if the data is not stored in memory recently
        module_embedding = db_utils.get_embeddings_by_layer(
            db_path=database_path,
            layer_name=layer_name
        )
        # Try to get module embeddings as np.array. Will throw an error if mismatched shapes
        try:
            module_embedding, module_embedding_same_shapes = db_utils.unwrap_embeddings(module_embedding)
        except Exception as e:
            print(layer_name, module_embedding)

        for modality in modalities:
            # Optionally extract image tokens if this is a text-image layer
            if modality == "vision" and layer_name.startswith("model"): # TODO: this might be different criteria for non Qwen Models
                modality_embedding, n_embeddings = db_utils.extract_visual_embeddings(
                    input_ids=input_ids,
                    llm_embeddings=module_embedding,
                    image_token_id=IMAGE_TOKEN_IDS[model.config.architecture],
                    same_shapes=module_embedding_same_shapes)
            elif modality == "text" and layer_name.startswith("model"): # TODO: this might be different criteria for non Qwen Models
                pass
            else: # text+vision
                modality_embedding = np.copy(module_embedding)
                n_embeddings = None  # if None, compute mean embedding over whole sequence

            # Calculate mean embedding
            mean_embeddings = db_utils.compute_mean_embeddings(
                embeddings=modality_embedding,
                n_embeddings=n_embeddings)


            # Calculate similarities of pairs of images
            module_sim = db_utils.cosine_similarity_numpy(mean_embeddings, mean_embeddings)
            # Assert similarity is symmetric
            assert np.array_equal(module_sim, module_sim.T)

            # Select only Upper Triangular Matrix
            n_samples = module_sim.shape[0]
            ut_idxs = np.triu_indices(n_samples, k=1)
            sim_values = module_sim[ut_idxs]
            assert len(sim_values) == n_samples * (n_samples - 1) / 2

            # Store values in list
            if modality == "text" or modality == "vision":
                module_name = layer_name + "-" + modality
            else:
                module_name = layer_name
            module_names.append(module_name)
            module_embeddings.append(mean_embeddings)
            module_similarities.append(sim_values)

    return (module_names, module_embeddings, module_similarities)

        # Compute mean embedding of visual tokens only (if applicable)
        # if layer_name.startswith("model"):
        #     module_visual_embedding, n_visual_tokens = db_utils.extract_visual_embeddings(
        #         input_ids=input_ids,
        #         llm_embeddings=module_embedding,
        #         image_token_id=IMAGE_TOKEN_IDS[model.config.architecture],
        #         same_shapes=module_embedding_same_shapes
        #     )
        #     # Calculate mean embedding
        #     visual_mean_embeddings = db_utils.compute_mean_embeddings(
        #         embeddings=module_visual_embedding,
        #         n_embeddings=n_visual_tokens)
        #     # Compute similarity
        #     module_sim = db_utils.cosine_similarity_numpy(mean_embeddings, mean_embeddings)
        #     # Assert similarity is symmetric
        #     assert np.array_equal(module_sim, module_sim.T)

        #     # Select only Upper Triangular Matrix
        #     n_samples = module_sim.shape[0]
        #     ut_idxs = np.triu_indices(n_samples, k=1)
        #     sim_values = module_sim[ut_idxs]
        #     assert len(sim_values) == n_samples * (n_samples - 1) / 2

        #     # Store values in list
        #     module_embeddings.append(visual_mean_embeddings)
        #     module_similarities.append(sim_values)


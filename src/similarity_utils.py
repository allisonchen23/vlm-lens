# Library imports
import numpy as np

# Local imports
import db_utils
from models.config import IMAGE_TOKEN_IDS

def compute_image_pair_similarities(database_path,
                                    model,
                                    layer_names=None,
                                    extract_visual_tokens=True):
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
    # Get input_ids
    if extract_visual_tokens:
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
                layer_names.append(name[0])

    layer_embeddings = []
    layer_similarities = []

    # Get embeddings for each layer and compute image-pair similarity scores
    for layer_name in layer_names:
        if not model.config.matches_module(layer_name):
            continue
        print(layer_name)
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

        # Optionally extract image tokens if this is a text-image layer
        if extract_visual_tokens and layer_name.startswith("model"): # TODO: this might be different criteria for non Qwen Models
            module_embedding, n_embeddings = db_utils.extract_visual_embeddings(
                input_ids=input_ids,
                llm_embeddings=module_embedding,
                image_token_id=IMAGE_TOKEN_IDS[model.config.architecture],
                same_shapes=module_embedding_same_shapes)
        else:
            n_embeddings = None  # if None, compute mean embedding over whole sequence

        # Calculate mean embedding
        mean_embeddings = db_utils.compute_mean_embeddings(
            embeddings=module_embedding,
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
        layer_embeddings.append(mean_embeddings)
        layer_similarities.append(sim_values)

    return (layer_names, layer_embeddings, layer_similarities)

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
        #     layer_embeddings.append(visual_mean_embeddings)
        #     layer_similarities.append(sim_values)


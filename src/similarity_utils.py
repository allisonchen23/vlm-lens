# Library imports
import numpy as np
import os
from tqdm import tqdm

# Local imports
import db_utils
from models.config import IMAGE_TOKEN_IDS
import visualizations, utils

AVAILABLE_MODALITIES = ['text', 'vision', 'text+vision']
EPS = 1e-4

def compute_image_pair_similarity_at_layer(model,
                                           layer_name,
                                           database_path,
                                           modalities,
                                           input_ids,
                                           # Parameters for unwrapping attention
                                           unwrap_fn=None,
                                           attn_component=None):
    """
    Compute the image pair similarities for a given layer, potentially selecting different modalities

    Arg(s):
        model : model.ModelBase
        layer_name : str
        database_path : str
        modalities : list[str]
        input_ids : np.array

    Returns:
        module_names : list[str]
        module_embeddings : list[np.array]
        module_sims : list[np.array]

        Length of lists = len(modalities)
    """
    if not model.config.matches_module(layer_name):
        print("{} not present in config".format(layer_name))
        return

    # This may take a long time if the data is not stored in memory recently
    if database_path.endswith(".db"):
        module_embedding = db_utils.get_embeddings_by_layer(
            db_path=database_path,
            layer_name=layer_name)
        # Try to get module embeddings as np.array. Will throw an error if mismatched shapes
        try:
            module_embedding, module_embedding_same_shapes = db_utils.unwrap_embeddings(module_embedding)
        except Exception as e:
            print(layer_name, module_embedding)
    else: # Is a directory hopefully
        filepath = os.path.join(database_path, "{}.npy".format(layer_name))
        if not os.path.exists(filepath):
            raise ValueError("File at {} does not exist".format(filepath))
        module_embedding = utils.read_file(filepath)
        module_embedding_same_shapes = True # If stored as a np.ndarray, representations for images must have same shape


    module_name = layer_name
    # Might need to unwrap attention QKV
    if unwrap_fn is not None:
        assert attn_component is not None, "Must pass in an attention component to extract"
        assert attn_component in ["query", "key", "value"], \
        "Unrecognized attention component: {}".format(attn_component)
        module_embedding = unwrap_fn(module_embedding)
        if attn_component == "query":
            module_embedding = module_embedding[0, ...]
        elif attn_component == "value":
            module_embedding = module_embedding[1, ...]
        else: # value
            module_embedding = module_embedding[2, ...]

        module_name += "_{}".format(attn_component)

    layer_modality = model.get_layer_modality(layer_name)
    assert layer_modality in ["vision", "text"]

    module_names = []
    module_embeddings = []
    module_sims = []
    for modality in modalities:
        if layer_modality == "vision": # In the vision space of the model
            if modality == "vision":
                modality_embedding = np.copy(module_embedding)
                n_embeddings = None
                modality_name = module_name
            elif modality == "text":
                pass
            else: # text + vision
                continue # Because there is no additional text in the vision layers, skip
        elif layer_modality == "text": # In the text space of the model
            if modality == "vision":
                modality_embedding, n_embeddings = db_utils.extract_visual_embeddings(
                    input_ids=input_ids,
                    llm_embeddings=module_embedding,
                    image_token_id=IMAGE_TOKEN_IDS[model.config.architecture],
                    same_shapes=module_embedding_same_shapes)
                modality_name =  module_name + "-{}".format(modality)
            elif modality == "text":
                pass
            else: # text + vision
                # Do nothing special because we want both text + vision
                modality_embedding = np.copy(module_embedding)
                n_embeddings = None  # if None, compute mean embedding over whole sequence
                modality_name = module_name

        # Calculate mean embedding
        mean_embeddings = db_utils.compute_mean_embeddings(
            embeddings=modality_embedding,
            n_embeddings=n_embeddings)

        # Calculate similarities of pairs of images
        module_sim = db_utils.cosine_similarity_numpy(mean_embeddings, mean_embeddings)


        # Assert similarity is symmetric
        assert (module_sim - module_sim.T < EPS).all()
        # assert np.array_equal(module_sim, module_sim.T), "{}".format(np.mean(module_sim - module_sim.T))

        # Select only Upper Triangular Matrix
        n_samples = module_sim.shape[0]
        ut_idxs = np.triu_indices(n_samples, k=1)
        sim_values = module_sim[ut_idxs]
        assert len(sim_values) == n_samples * (n_samples - 1) / 2

        # Add to list
        module_names.append(modality_name)
        module_embeddings.append(mean_embeddings)
        module_sims.append(sim_values)

    return module_names, module_embeddings, module_sims

def compute_image_pair_similarities(database_path,
                                    model,
                                    layer_names=None,
                                    modalities=['visual'],
                                    # Parameters for unwrapping attention
                                    attn_component=None):
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
        if database_path.endswith(".db"):
            input_ids = db_utils.get_embeddings_by_layer(
                db_path=database_path,
                layer_name="input_ids")
            input_ids, _ = db_utils.unwrap_embeddings(input_ids) # empty value is input_ids_same_shapes
        else: # Is a directory hopefully
            filepath = os.path.join(database_path, "{}.npy".format("input_ids"))
            if not os.path.exists(filepath):
                raise ValueError("File at {} does not exist".format(filepath))
            input_ids = utils.read_file(filepath)

        assert model.config.architecture in IMAGE_TOKEN_IDS, \
            "No entry for {} in models.config.IMAGE_TOKEN_IDS".format(model.config.architecture)

    # If layer_names is None, assume all layer names in config are desired
    if layer_names is None:
        layer_names = []
        for name, _ in model.model.named_modules():
            if model.config.matches_module(name):
                layer_names.append(name)

    # Return values
    module_names = []
    module_embeddings = []
    module_similarities = []

    # Get embeddings for each layer and compute image-pair similarity scores
    for layer_name in tqdm(layer_names):
        if "qkv" in layer_name: # TODO: This might need to change beyond Qwen models
            unwrap_fn = model.get_unwrap_qkv_fn()
        else:
            unwrap_fn = None
        modality_names, modality_embeddings, modality_similarities = \
            compute_image_pair_similarity_at_layer(
                model=model,
                layer_name=layer_name,
                database_path=database_path,
                modalities=modalities,
                input_ids=input_ids,
                unwrap_fn=unwrap_fn,
                attn_component=attn_component)

        module_names += modality_names
        module_embeddings += modality_embeddings
        module_similarities += modality_similarities

    return (module_names, module_embeddings, module_similarities)

# For each layer, calculate the mean similarity and the norm
def plot_similarities(module_names,
                      module_embeddings,
                      module_similarities,
                      vision_key,
                      model_name=''):
    '''
    Given list of module names, image embeddings (vision and text + vision), and similarities
    plot mean similarity score for each layer

    Arg(s):
        module_names : list[str]
        module_embeddings : list[2D np.array]
        module_similarities : list[1D np.array]
    '''
    mean_embeddings = {}
    mean_similarities = {}
    for idx, (name, embs, sims) in enumerate(zip(module_names, module_embeddings, module_similarities)):
        # Key is based on vision vs text
        key = name.split(".")[0]
        # If we are isolating vision token embeddings
        if "-" in name:
            key += " ({})".format(name.split("-")[1])
        if key in mean_embeddings:
            mean_embeddings[key].append(np.mean(embs))
        else:
            mean_embeddings[key] = [np.mean(embs)]

        if key in mean_similarities:
            mean_similarities[key].append(np.mean(sims))
        else:
            mean_similarities[key] = [np.mean(sims)]

    # Plot mean image-pair similarity scores by layer
    xs = []
    ys = []
    labels = []

    # Separate data based on modality and layer
    if vision_key in mean_embeddings:
        n_vision_layers = len(mean_embeddings[vision_key])
    else:
        n_vision_layers = 0
    print(mean_embeddings.keys())
    for k, v in mean_similarities.items():
        if vision_key not in k: # text layer
            xs.append([i + n_vision_layers for i in range(len(v))])
        else:
            xs.append([i for i in range(len(v))])
        ys.append(v)
        labels.append("{} blocks".format(k))

    visualizations.plot(
        xs=xs,
        ys=ys,
        labels=labels,
        alpha=0.6,
        xlabel='Layer of {} Model'.format(model_name),
        ylabel='Mean Image Pairwise Similarity (0-1)',
        show=True
    )
    return mean_embeddings, mean_similarities

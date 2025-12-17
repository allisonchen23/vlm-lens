import io
import numpy as np
import os
import sqlite3
import torch
from tqdm import tqdm

import utils

def cosine_similarity_numpy(a,
                            b,
                            elementwise=False):
    """Calculate cosine similarity between two vectors or matrices using numpy with robust error handling.

    Arg(s):
        a : D-dim np.array or B x D np.array
        b : D-dim np.array or B x D np.array
        elementwise : bool
            Only applicable if a and b are 2D arrays
            If True, only compute similarities of elements at same location.
            Otherwise, compute all pairwise cosine similarities

    Returns:
        int (if a and b are 1D) or B x B np.array (if a and b are 2D)
    """
    assert a.shape == b.shape
    # Check for NaN or infinite values
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        print("Warning: NaN or infinite values detected in tensors")
        return 0.0
    if len(a.shape) == 1:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Handle zero vectors or invalid norms
        if norm_a == 0 or norm_b == 0 or not (np.isfinite(norm_a).all() and np.isfinite(norm_b).all()):
            return 0.0
        dot_product = np.dot(a, b.T)

        # Check if dot product is valid
        if not np.isfinite(dot_product).all():
            print("Warning: Invalid dot product")
            return 0.0

        return dot_product / (norm_a * norm_b)
    elif len(a.shape) == 2:
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        if np.sum(norm_a) == 0 or np.sum(norm_b) == 0 or not (np.isfinite(norm_a).all() and np.isfinite(norm_b).all()):
            return 0.0

        normalized_a = a / norm_a
        normalized_b = b / norm_b
        if elementwise:
            # Compute element-wise multiplication then sum over D dimension for
            # element-wise dot product
            return np.sum(normalized_a * normalized_b, axis=1)
        else:
            # Compute pairwise dot product
            return np.dot(normalized_a, normalized_b.T)
    elif len(a.shape) == 3:
        norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=-1, keepdims=True)

        normalized_a = a / norm_a
        normalized_b = b / norm_b

        if elementwise:
            raise ValueError("Elementwise dot product for n_dim == 3 not yet supported")
        else:
            return np.matmul(normalized_a, normalized_b.transpose(0, 2, 1))

    else:
        raise ValueError("Cosine similarity not supported for {}-dimensional matrices".format(len(a.shape)))



def extract_tensor_from_object(tensor_obj):
    """Extract actual tensor data from HuggingFace model outputs or other objects."""
    if hasattr(tensor_obj, 'last_hidden_state'):
        # HuggingFace model output - extract the main tensor
        return tensor_obj.last_hidden_state
    elif hasattr(tensor_obj, 'pooler_output'):
        # Use pooled output if available
        return tensor_obj.pooler_output
    elif hasattr(tensor_obj, 'hidden_states'):
        # Use hidden states
        return tensor_obj.hidden_states
    elif torch.is_tensor(tensor_obj):
        # Already a tensor
        return tensor_obj
    else:
        # Try to find the first tensor attribute
        for attr_name in dir(tensor_obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(tensor_obj, attr_name)
                    if torch.is_tensor(attr_value):
                        print(f"Using attribute '{attr_name}' from {type(tensor_obj).__name__}")
                        return attr_value
                except:
                    continue

        print(f"Could not find tensor data in {type(tensor_obj).__name__}")
        return None

def get_all_embeddings(db_path="output/llava.db", device='cpu'):
    """
    Retrieve all embeddings from the database using PyTorch tensor deserialization.

    Args:
        db_path: Path to the SQLite database
        device: PyTorch device for tensor loading ('cpu', 'cuda', etc.)

    Returns:
        List of tuples: (layer, tensor_data, label)
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Direct SQL query to get layer, tensor, and label
    cursor.execute("SELECT layer, tensor, label FROM tensors")
    results = cursor.fetchall()

    # Close the connection
    connection.close()

    embeddings_data = []

    for row_id, (layer, tensor_bytes, label) in enumerate(tqdm(results)):
        try:
            # Use PyTorch to load tensor from BytesIO with weights_only=False
            tensor_obj = torch.load(io.BytesIO(tensor_bytes), map_location=device, weights_only=False)

            # Extract actual tensor from object
            tensor = extract_tensor_from_object(tensor_obj)
            if tensor is None:
                continue

            # Convert dtype from bfloat16 -> float32 if needed
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            # Convert to numpy for analysis
            if tensor.requires_grad:
                tensor_np = tensor.detach().cpu().numpy()
            else:
                tensor_np = tensor.cpu().numpy()

            embeddings_data.append((layer, tensor_np, label))

        except Exception as e:
            print(f"Warning: Could not deserialize tensor at row {row_id}: {e}")
            continue


    return embeddings_data

def get_embeddings_by_layer(db_path="output/llava.db", layer_name=None, device='cpu'):
    """
    Retrieve embeddings for a specific layer from the database.

    Args:
        db_path: Path to the SQLite database
        layer_name: Name of the layer to filter by
        device: PyTorch device for tensor loading

    Returns:
        List of tuples: (layer, tensor_data, label)
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    if layer_name:
        cursor.execute("SELECT layer, tensor, label FROM tensors WHERE layer = ?", (layer_name,))
    else:
        cursor.execute("SELECT layer, tensor, label FROM tensors")

    results = cursor.fetchall()
    connection.close()

    embeddings_data = []

    for row_id, (layer, tensor_bytes, label) in enumerate(results):
        try:
            tensor_obj = torch.load(io.BytesIO(tensor_bytes), map_location=device, weights_only=False)

            # Extract actual tensor from object
            tensor = extract_tensor_from_object(tensor_obj)
            if tensor is None:
                continue


            # Convert dtype from bfloat16 -> float32 if needed
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            # Convert to numpy for analysis
            if tensor.requires_grad:
                tensor_np = tensor.detach().cpu().numpy()
            else:
                tensor_np = tensor.cpu().numpy()

            embeddings_data.append((layer, tensor_np, label))

        except Exception as e:
            print(f"Warning: Could not deserialize tensor at row {row_id}: {e}")
            continue

    return embeddings_data

def unwrap_embeddings(embeddings,
                      idx=1):
    """
    From output of `get_embeddings_by_layer()` extract only the embeddings
    idx = 0 gives layer; 1 gives embeddings; 2 gives label
    """
    try:
        unwrapped = np.squeeze(np.array(tuple(map(list, zip(*embeddings)))[idx]))
        same_shapes = True
    except Exception as e:
        print("Could not squeeze embeddings into 2D array: {}".format(e))
        unwrapped = tuple(map(list, zip(*embeddings)))[idx]
        same_shapes = False
    return unwrapped, same_shapes

def get_layer_names(db_path="output/llava.db"):
    """
    Get all unique layer names from the database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        List of unique layer names
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("SELECT DISTINCT layer FROM tensors")
    layers = [row[0] for row in cursor.fetchall()]

    connection.close()
    return layers

def analyze_layer_similarity(db_path="output/llava.db", layer_name=None, device='cpu'):
    """
    Analyze cosine similarity between embeddings within a specific layer.
    (albeit a bit inefficiently)
    Args:
        db_path: Path to the SQLite database
        layer_name: Specific layer to analyze (if None, analyzes all layers)
        device: PyTorch device for tensor loading

    Returns:
        Dictionary mapping layer names to similarity results
    """
    if layer_name:
        embeddings = get_embeddings_by_layer(db_path, layer_name, device)
        layer_groups = {layer_name: embeddings}
    else:
        # Get all embeddings and group by layer
        all_embeddings = get_all_embeddings(db_path, device)
        layer_groups = {}
        for layer, tensor, label in all_embeddings:
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append((layer, tensor, label))

    similarity_results = {}

    for layer, embeddings in layer_groups.items():
        if len(embeddings) < 2:
            print(f"Skipping layer '{layer}': only {len(embeddings)} embedding(s)")
            continue

        print(f"\n=== Cosine Similarity Analysis for Layer: {layer} ===")

        # Extract tensors and labels
        tensors = [tensor.flatten() for _, tensor, _ in embeddings]
        labels = [label for _, _, label in embeddings]


        # Debug: Check tensor validity
        print("Tensor analysis:")
        for i, tensor in enumerate(tensors):

            norm = np.linalg.norm(tensor)
            has_nan = np.isnan(tensor).any()
            has_inf = np.isinf(tensor).any()
            min_val = np.min(tensor) if len(tensor) > 0 else 0
            max_val = np.max(tensor) if len(tensor) > 0 else 0

            label_str = labels[i] if labels[i] else "No label"
            print(f"  Tensor {i} ({label_str}): norm = {norm:.6f}")
            print(f"    Shape: {tensor.shape}, Range: [{min_val:.6f}, {max_val:.6f}]")
            print(f"    has_nan: {has_nan}, has_inf: {has_inf}")

            if norm == 0:
                print(f"    WARNING: Tensor {i} is a zero vector!")
            if has_nan:
                print(f"    WARNING: Tensor {i} contains NaN values!")
            if has_inf:
                print(f"    WARNING: Tensor {i} contains infinite values!")

        layer_similarities = []

        # Calculate all pairwise similarities
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                similarity = cosine_similarity_numpy(tensors[i], tensors[j])

                label1 = labels[i] if labels[i] else f"Tensor_{i}"
                label2 = labels[j] if labels[j] else f"Tensor_{j}"

                result = {
                    'tensor1_idx': i,
                    'tensor2_idx': j,
                    'label1': label1,
                    'label2': label2,
                    'similarity': similarity
                }
                layer_similarities.append(result)

                print(f"Tensor {i} vs Tensor {j}: {similarity:.4f}")
                print(f"  {label1} vs {label2}")

        similarity_results[layer] = layer_similarities

    return similarity_results

def get_database_info(db_path="output/llava.db"):
    """
    Get basic information about the database contents.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Dictionary with database statistics
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Get total number of tensors
    cursor.execute("SELECT COUNT(*) FROM tensors")
    total_tensors = cursor.fetchone()[0]

    # Get unique layers
    cursor.execute("SELECT layer, COUNT(*) FROM tensors GROUP BY layer")
    layer_counts = dict(cursor.fetchall())

    # Get unique labels
    cursor.execute("SELECT label, COUNT(*) FROM tensors WHERE label IS NOT NULL GROUP BY label")
    label_counts = dict(cursor.fetchall())

    connection.close()

    return {
        'total_tensors': total_tensors,
        'layer_counts': layer_counts,
        'label_counts': label_counts,
        'unique_layers': list(layer_counts.keys())
    }

def extract_visual_embeddings(input_ids,
                              llm_embeddings,
                              image_token_id,
                              same_shapes=False):
    """
    Given input IDs and LLM embeddings,
        return tokens that are of the image representation; text tokens become 0

    Arg(s):
        input_ids : B x T np.array or B-length list of variable-length np.arrays
        llm_embeddings : B x T x D np. array or B-length list of variable-length x D np.arrays
        image_token_id : int

    Returns:
        visual_tokens, n_visual_tokens : B x T x D np.array, B-dim integer np.array
    """
    if same_shapes:
        # Check that Batch and Token dimensions match
        assert input_ids.shape[0] == llm_embeddings.shape[0]
        assert input_ids.shape[1] == llm_embeddings.shape[1]
        assert len(input_ids.shape) == 2
        assert len(llm_embeddings.shape) == 3

        # Mask where token corresponds to image
        mask = np.where(input_ids == image_token_id, 1, 0)
        n_visual_embs = np.sum(mask, axis=1)
        visual_embs = llm_embeddings * mask[..., None]

    else: # Handle lists of arrays
        visual_embs = []
        n_visual_embs = []
        for input_id, llm_embedding in zip(input_ids, llm_embeddings):
            assert input_id.shape == llm_embedding.shape[:-1] # Assert same shape for first 2 dimensions
            mask = np.where(input_id == image_token_id, 1, 0)
            n_visual_emb = np.sum(mask, axis=1)
            visual_emb = llm_embedding * mask[..., None]
            visual_embs.append(visual_emb)
            n_visual_embs.append(n_visual_emb)
    return visual_embs, n_visual_embs

def compute_mean_embeddings(embeddings,
                            n_embeddings=None):
    """
    Compute mean embedding for each element in the batch

    Arg(s):
        embeddings : B x T x D
        n_embeddings : int or None (if int, divide by n_embeddings instead of T)
    """

    # Get denominator for mean
    assert len(embeddings.shape) == 3
    if n_embeddings is None:
        n_embeddings = np.full((embeddings.shape[0]), embeddings.shape[1]) # Number of tokens

    # Compute mean
    mean_embeddings = np.sum(embeddings, axis=1) / n_embeddings[..., None]

    # Shape assertions
    assert len(mean_embeddings.shape) == 2
    assert mean_embeddings.shape[0] == embeddings.shape[0]
    assert mean_embeddings.shape[1] == embeddings.shape[2]

    return mean_embeddings

def extract_features_and_targets(db_path="output/llava.db", probe_layer=None, device='cpu'):
    """
    Extract features and targets for machine learning, following VLM-Lens pattern.

    Args:
        db_path: Path to the SQLite database
        probe_layer: Specific layer to extract (if None, extracts all)
        device: PyTorch device for tensor loading

    Returns:
        Tuple: (features, targets, label_to_idx)
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("SELECT layer, tensor, label FROM tensors")
    results = cursor.fetchall()
    connection.close()

    # Gather unique class labels
    all_labels = set([result[2] for result in results if result[2] is not None])
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    features, targets = [], []

    for layer, tensor_bytes, label in results:
        if (probe_layer and layer == probe_layer) or (not probe_layer):
            try:
                tensor_obj = torch.load(io.BytesIO(tensor_bytes), map_location=device, weights_only=False)

                # Extract actual tensor from object
                tensor = extract_tensor_from_object(tensor_obj)
                if tensor is None:
                    continue

                # Convert to numpy
                if tensor.requires_grad:
                    tensor_np = tensor.detach().cpu().numpy()
                else:
                    tensor_np = tensor.cpu().numpy()

                # Flatten for ML
                feature_vector = tensor_np.flatten()

                if label is not None:
                    features.append(feature_vector)
                    targets.append(label_to_idx[label])

            except Exception as e:
                print(f"Warning: Could not process tensor for layer {layer}: {e}")
                continue

    return np.array(features), np.array(targets), label_to_idx

def save_embeddings_npy(db_path,
                        layer_names,
                        overwrite=False):
    """
    Extract embeddings from database and save as separate .npy files for each layer. To save time for later retrieval

    Arg(s):
        db_path : str
        layer_names : list[str]
    """
    save_dir, _ = os.path.splitext(db_path)
    utils.ensure_dir(save_dir)

    # Save input IDs
    try:
        save_path = os.path.join(save_dir, "{}.npy".format("input_ids"))
        # Do not overwrite
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("File exists at {} and not overwriting".format(save_path))
        else:
            input_ids = get_embeddings_by_layer(
                    db_path=db_path,
                    layer_name="input_ids")
            try:
                input_ids, _ = unwrap_embeddings(input_ids)
            except Exception as e:
                raise ValueError("{} {} {}".format(e, "input IDs", input_ids))
            assert isinstance(input_ids, np.ndarray)
            utils.write_file(input_ids, save_path)
    except:
        utils.informal_log("Could not save input IDs")

    # Iterate through layers and save embeddings
    for idx, layer_name in enumerate(tqdm(layer_names)):
        save_path = os.path.join(save_dir, "{}.npy".format(layer_name))

        # Do not overwrite
        if os.path.exists(save_path) and not overwrite:
            utils.informal_log("File exists at {} and not overwriting".format(save_path))
            continue

        module_embedding = get_embeddings_by_layer(
            db_path=db_path,
            layer_name=layer_name
        )
        try:
            module_embedding, _ = unwrap_embeddings(module_embedding)
        except Exception as e:
            raise ValueError("{} {} {}".format(e, layer_name, module_embedding))
        assert isinstance(module_embedding, np.ndarray)
        utils.write_file(module_embedding, save_path)

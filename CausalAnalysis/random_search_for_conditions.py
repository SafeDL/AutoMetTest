"""
Ablation experiment: Prepare test conditions by random search
"""
import numpy as np
import pandas as pd
from utils import variable_to_string
import clip
import torch
from search_for_conditions import cosine_similarity


def selected_prompts(datapath=None):
    # For the found optimal test conditions, use similarity_threshold=0.95 to remove duplicates
    df = pd.read_excel(datapath)
    string_vectors = []
    text_features = []
    clip_model, _ = clip.load("ViT-B/32", "cuda")
    # Convert test conditions to string
    for idx, row in df.iterrows():
        variables = row.values
        string_vector, _ = variable_to_string(variables)
        # print(f"String representation of test condition {idx}: {string_vector}")
        string_vectors.append(string_vector)
        with torch.no_grad():
            texts = clip.tokenize(string_vector).to("cuda")
            text_features.append(clip_model.encode_text(texts).cpu().numpy())

    # Set a similarity threshold. The higher the threshold, the higher the required similarity, and the easier it is to remove similar conditions
    similarity_threshold = 0.95

    # Calculate the similarity between found string_vectors
    unique_string_vectors = []  # Used to store deduplicated test conditions
    n_samples = len(string_vectors)

    # Initially, add the first string_vector to the unique list
    unique_string_vectors.append(string_vectors[0])

    # Traverse the subsequent string_vectors and calculate their similarity with the existing unique_string_vectors
    for idx in range(1, n_samples):
        current_vector = string_vectors[idx]
        is_unique = True  # Assume the current string_vector is unique

        for unique_vector in unique_string_vectors:
            # Calculate the cosine similarity between two vectors
            similarity = cosine_similarity(text_features[idx].reshape(-1),
                                           text_features[string_vectors.index(unique_vector)].reshape(-1))

            # If the similarity exceeds the threshold, consider the two string_vectors similar and mark as not unique
            if similarity >= similarity_threshold:
                is_unique = False
                break

        # If the string_vector is unique, add it to unique_string_vectors
        if is_unique:
            unique_string_vectors.append(current_vector)

    # Calculate the diversity between the found unique test conditions: the larger (1 - cosine similarity), the greater the diversity
    diversity_score = []
    for idx in range(0, len(unique_string_vectors)):
        for jdx in range(0, len(unique_string_vectors)):
            if idx != jdx:
                # Calculate diversity (1 - cosine similarity)
                diversity_score.append(
                    1 - cosine_similarity(text_features[string_vectors.index(unique_string_vectors[idx])].reshape(-1),
                                          text_features[string_vectors.index(unique_string_vectors[jdx])].reshape(-1)))

    return unique_string_vectors, diversity_score


def random_search(num_samples=1000, bounds=None):
    """
    Generate test condition combinations using random search.
    :param num_samples: Number of randomly generated samples
    :param bounds: Bounds for each dimension
    :return: Randomly generated condition combinations
    """
    if bounds is None:
        bounds = [(0, 100), (0, 100), (1, 8), (0, 100), (0, 100), (5, 70), (0, 100), (-90, 90)]

    # Randomly generate test conditions
    samples = np.array([
        [np.random.randint(low, high) for low, high in bounds]
        for _ in range(num_samples)
    ])
    return samples


if __name__ == '__main__':
    # Define decision variable names and ranges
    decision_variables = ['cloudiness', 'dust_storm', 'roadtype', 'precipitation', 'fog_density',
                          'traffic_density', 'precipitation_deposits', 'sun_altitude_angle']
    variable_bounds = [(0, 100), (0, 100), (1, 8), (0, 50), (0, 50), (5, 70), (0, 60), (-50, 90)]

    # Generate condition combinations using random search
    num_samples = 32
    random_conditions = random_search(num_samples=num_samples, bounds=variable_bounds)

    # Save the results to an Excel file
    df = pd.DataFrame(random_conditions, columns=decision_variables)
    df.to_excel('./result/random_conditions.xlsx', index=False)

    unique_string_vectors, diversity_score = selected_prompts(datapath='./result/random_conditions.xlsx')
    mean_diversity = np.mean(np.array(diversity_score))
    print(f"Average diversity between found test conditions: {mean_diversity}")

    # Output unique_string_vectors
    print("Deduplicated test conditions:")
    for idx, unique_vector in enumerate(unique_string_vectors):
        print(f"String representation of test condition {idx}: {unique_vector}")

    # Translate random condition combinations into text prompts
    data = pd.read_excel('./result/random_conditions.xlsx')
    for i in range(data.shape[0]):
        variables = data.iloc[i, :].values
        combined_string, paired_results = variable_to_string(variables)
        # Concatenate results into the required format and write to text
        with open('./result/random_prompts.txt', 'a') as f:
            output_line = ', '.join([f"({pair[0]}:{pair[1]})" for pair in paired_results])
            f.write(output_line + '\n')

    print(f"Randomly generated {num_samples} test condition combinations, and saved to 'random_conditions.xlsx' and 'random_conditions_prompts.txt'.")

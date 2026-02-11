"""
By designing a multi-objective optimization strategy, using inconsistency measured by causal counterfactuals / encoded space distance as optimization objectives,
search for the optimal combination of test conditions.
"""
import numpy as np
import torch
import clip
import geatpy as ea
from causal_structural_models import Causal4End2End
import pandas as pd
from utils import variable_to_string
import warnings
warnings.filterwarnings('ignore')


def selected_prompts(datapath='./result/searched_conditions.xlsx'):
    df = pd.read_excel(datapath)
    string_vectors = []
    text_features = []
    clip_model, _ = clip.load("ViT-B/32", "cuda")
    for idx, row in df.iterrows():
        variables = row.values[:-1]
        string_vector, _ = variable_to_string(variables)
        string_vectors.append(string_vector)
        with torch.no_grad():
            texts = clip.tokenize(string_vector).to("cuda")
            text_features.append(clip_model.encode_text(texts).cpu().numpy())

    similarity_threshold = 0.95

    # Calculate the similarity between found string_vectors
    unique_string_vectors = [] 
    n_samples = len(string_vectors)
    unique_string_vectors.append(string_vectors[0])
    for idx in range(1, n_samples):
        current_vector = string_vectors[idx]
        is_unique = True 
        for unique_vector in unique_string_vectors:
            similarity = cosine_similarity(text_features[idx].reshape(-1),
                                           text_features[string_vectors.index(unique_vector)].reshape(-1))
            if similarity >= similarity_threshold:
                is_unique = False
                break

        if is_unique:
            unique_string_vectors.append(current_vector)

    diversity_score = []
    for idx in range(0, len(unique_string_vectors)):
        for jdx in range(0, len(unique_string_vectors)):
            if idx != jdx:
                diversity_score.append(
                    1 - cosine_similarity(text_features[string_vectors.index(unique_string_vectors[idx])].reshape(-1),
                                          text_features[string_vectors.index(unique_string_vectors[jdx])].reshape(-1)))

    return unique_string_vectors, diversity_score


def cosine_similarity(vec1, vec2):
    # Compute the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)

    return similarity


def challenge_metric(causal_model, current_parameter, alpha=0.6):
    rmse_values = causal_model.cal_counterfactural(current_parameter)
    mean_rmse = np.mean(np.array(rmse_values))
    inconsistencies = np.var(rmse_values)
    objective_value = alpha * mean_rmse + (1 - alpha) * inconsistencies

    return objective_value


def calculate_clip_diversity(idx, texts, clip_model):
    """
    Calculate the semantic diversity between samples.
    :param idx: Index of the current test condition
    :param texts: List of text prompts
    :param clip_model: Pretrained CLIP model for encoding text
    :return: Diversity objective value
    """
    with torch.no_grad():
        text_features = clip_model.encode_text(texts).cpu().numpy()
        diversity_score = []
        n_samples = len(texts)
        for jdx in range(0, n_samples):
            if idx != jdx:
                diversity_score.append(1 - cosine_similarity(text_features[idx], text_features[jdx]))

        minimum_diversity = np.min(np.array(diversity_score))
    return minimum_diversity


def calculate_environment_diversity(current_parameter, all_parameters):
    """
    Calculate the numerical diversity of the current test condition compared to other test conditions in environmental variables.
    :param current_parameter: Array of the current test condition
    :param all_parameters: Array of all test conditions
    :return: Diversity objective value
    """
    diversity_scores = []
    for other_parameter in all_parameters:
        diversity_scores.append(np.linalg.norm(current_parameter - other_parameter))

    min_score = np.min(diversity_scores)
    max_score = np.max(diversity_scores)
    scaled_scores = [(score - min_score) / (max_score - min_score) for score in diversity_scores]

    return np.mean(scaled_scores)


# Define the objective function: (1) Calculate the consistency of detection results; (2) Calculate test coverage
def fitness_function(Vars, causal_model):
    # Vars is an ndarray of shape (n, 8), where each row represents a found test condition, and columns represent the values in different dimensions
    f1_list = []
    f2_list = []
    # First, traverse each row of Vars to get its semantic description
    text_prompts = []
    for idx in range(Vars.shape[0]):
        current_parameter = Vars[idx, :]
        current_parameter_string, _ = variable_to_string(current_parameter)
        text_prompts.append(current_parameter_string)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # Encode text
    encoded_texts = clip.tokenize(text_prompts).to(device)

    for idx in range(Vars.shape[0]):
        # (1) Traverse each row of Vars, perform counterfactual inference on its model performance
        current_parameter = Vars[idx, :]
        f1 = challenge_metric(causal_model=causal_model, current_parameter=current_parameter)

        # (2) Traverse each row of the model, calculate diversity (semantic and numerical)
        clip_diversity = calculate_clip_diversity(idx, encoded_texts, clip_model)
        env_diversity = calculate_environment_diversity(current_parameter, Vars)
        f2 = 0.7 * clip_diversity + 0.3 * env_diversity

        # (3) Add to the objective function value list
        f1_list.append(f1)
        f2_list.append(f2)

    return np.array(f1_list).reshape(-1, 1), np.array(f2_list).reshape(-1, 1)


########################################
# Define multi-objective optimization problem
########################################
class MyProblem(ea.Problem): 
    def __init__(self):
        name = 'NSGA-II Algorithm'  
        M = 2  
        maxormins = [-1] * M  
        Dim = 8 
        varTypes = [1, 1, 1, 1, 1, 1, 1, 1] 
        lb = [0, 0, 1, 0, 0, 5, 0, -90] 
        ub = [100, 100, 8, 100, 100, 70, 100, 90] 
        lbin = [1, 1, 1, 1, 1, 1, 1, 1]  
        ubin = [1, 1, 1, 1, 1, 1, 1, 1]  
        # Instantiate structural causal model
        self.build_causal_model()
        self.fitness = fitness_function
        # Call parent class constructor to complete instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def build_causal_model(self):
        # Define causal model
        decision_variables = ['cloudiness', 'dust_storm', 'roadtype', 'precipitation', 'fog_density',
                              'traffic_density', 'precipitation_deposits', 'sun_altitude_angle']

        targets = ['PilotNet', 'cg23', 'cnn_gru', 'stacked_cnn', 'two_stream_cnn']
        self.causal_model = Causal4End2End(causal_factors=decision_variables,
                                           train_data=pd.read_excel('./structures/CausalAnalysis_continuous_demo.xlsx'),
                                           factor_to_predict=targets)

    def evalVars(self, Vars): 
        f1, f2 = self.fitness(Vars, self.causal_model)
        ObjV = np.hstack([f1, f2])  
        CV = -Vars ** 2 + 2.5 * Vars - 1.5 
        self.previous_ObjV = getattr(self, 'previous_ObjV', None)
        self.ObjV = ObjV
        return ObjV, CV

    def has_converged(self, tolerance=1e-4):
        if self.previous_ObjV is None:
            return False
        change = np.abs(self.ObjV - self.previous_ObjV).max()
        self.previous_ObjV = self.ObjV
        return change < tolerance


if __name__ == '__main__':
    problem = MyProblem()
    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='RI', NIND=1000),
                                      MAXGEN=30, 
                                      logTras=1) 

    res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
                      dirName='result', early_stopping=problem.has_converged)  

    data = pd.read_excel('./result/searched_conditions.xlsx')
    for i in range(data.shape[0]):
        variables = data.iloc[i, :-1].values
        combined_string, paired_results = variable_to_string(variables)
        with open('./result/prompts.txt', 'a') as f:
            output_line = ', '.join([f"({pair[0]}:{pair[1]})" for pair in paired_results])
            f.write(output_line + '\n')

    unique_string_vectors, diversity_score = selected_prompts()
    mean_diversity = np.mean(np.array(diversity_score))

    with open('./result/filtered_prompts.txt', 'a') as f:
        for unique_vector in unique_string_vectors:
            f.write(f"{unique_vector}\n")

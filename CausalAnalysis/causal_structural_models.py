"""
It is recommended to use Tetrad to discover causal structures and build structural causal models based on this.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


class Causal4End2End:
    def __init__(self, causal_factors, train_data, regression_model=None, factor_to_predict=None):
        self.scaler = StandardScaler()
        # Specify various causal environmental variables discovered by causal discovery (also used as decision variables for target search)
        self.causal_factors = causal_factors
        # Specify the target variables for counterfactual estimation
        self.factor_to_predict = factor_to_predict
        # Feature scaling, used for comparing nearest neighbor distances
        self.scaled_current_data = self.scaler.fit_transform(train_data.loc[:, self.causal_factors].values)
        self.observational_data = train_data

        # Define the discovered causal graph model
        causal_graph = nx.DiGraph(nx.DiGraph(
            [('cloudiness', 'Brenner'), ('cloudiness', 'TeRES'), ('cloudiness', 'Tenengrad'),
             ('cloudiness', 'lapulaseDetection'),
             ('dust_storm', 'TeRES'),
             ('roadtype', 'Brenner'), ('roadtype', 'TeRES'), ('roadtype', 'PilotNet'), ('roadtype', 'cg23'),
             ('roadtype', 'cnn_gru'),
             ('roadtype', 'inception'), ('roadtype', 'stacked_cnn'), ('roadtype', 'lapulaseDetection'),
             ('roadtype', 'sift'),
             ('roadtype', 'two_stream_cnn'), ('precipitation', 'Brenner'), ('precipitation', 'TeRES'),
             ('precipitation', 'Tenengrad'),
             ('precipitation', 'niqe'), ('precipitation', 'orb'), ('fog_density', 'Brenner'),
             ('fog_density', 'PilotNet'),
             ('fog_density', 'TeRES'), ('fog_density', 'Tenengrad'), ('fog_density', 'Variance'),
             ('fog_density', 'cg23'),
             ('fog_density', 'cnn_gru'), ('fog_density', 'inception'), ('fog_density', 'lapulaseDetection'),
             ('fog_density', 'niqe'),
             ('fog_density', 'orb'), ('fog_density', 'phase'), ('fog_density', 'sift'), ('fog_density', 'stacked_cnn'),
             ('fog_density', 'std'), ('fog_density', 'two_stream_cnn'), ('traffic_density', 'inception'),
             ('traffic_density', 'orb'),
             ('precipitation_deposits', 'Brenner'), ('precipitation_deposits', 'cnn_gru'),
             ('precipitation_deposits', 'lapulaseDetection'),
             ('precipitation_deposits', 'niqe'), ('precipitation_deposits', 'orb'), ('precipitation_deposits', 'phase'),
             ('sun_altitude_angle', 'Brenner'), ('sun_altitude_angle', 'TeRES'), ('sun_altitude_angle', 'Tenengrad'),
             ('sun_altitude_angle', 'Variance'), ('sun_altitude_angle', 'lapulaseDetection'),
             ('sun_altitude_angle', 'niqe'),
             ('sun_altitude_angle', 'orb'), ('sun_altitude_angle', 'phase'), ('sun_altitude_angle', 'std'),
             ('Brenner', 'PilotNet'), ('TeRES', 'PilotNet'), ('TeRES', 'cg23'), ('TeRES', 'cnn_gru'),
             ('TeRES', 'stacked_cnn'),
             ('TeRES', 'two_stream_cnn'),
             ('Tenengrad', 'PilotNet'), ('Tenengrad', 'cg23'), ('Tenengrad', 'cnn_gru'), ('Tenengrad', 'stacked_cnn'),
             ('Variance', 'cg23'), ('Variance', 'cnn_gru'), ('Variance', 'stacked_cnn'), ('Variance', 'two_stream_cnn'),
             ('inception', 'PilotNet'), ('inception', 'cg23'), ('inception', 'cnn_gru'), ('inception', 'stacked_cnn'),
             ('inception', 'two_stream_cnn'), ('lapulaseDetection', 'PilotNet'), ('lapulaseDetection', 'cg23'),
             ('lapulaseDetection', 'cnn_gru'), ('lapulaseDetection', 'stacked_cnn'),
             ('lapulaseDetection', 'two_stream_cnn'),
             ('niqe', 'PilotNet'), ('niqe', 'cg23'), ('niqe', 'two_stream_cnn'),
             ('orb', 'PilotNet'), ('orb', 'cg23'), ('orb', 'cnn_gru'), ('orb', 'stacked_cnn'),
             ('phase', 'PilotNet'), ('phase', 'cg23'), ('phase', 'cnn_gru'), ('phase', 'stacked_cnn'),
             ('phase', 'two_stream_cnn'),
             ('sift', 'PilotNet'), ('sift', 'cg23'), ('sift', 'cnn_gru'), ('sift', 'stacked_cnn'),
             ('sift', 'two_stream_cnn'),
             ('std', 'PilotNet'), ('std', 'cg23'), ('std', 'cnn_gru'), ('std', 'stacked_cnn'), ('std', 'two_stream_cnn')
             ]))

        self.causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

        # Specify the probability distributions related to the causal variables
        self.causal_model.set_causal_mechanism('cloudiness', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('dust_storm', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('roadtype', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('precipitation', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('fog_density', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('traffic_density', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('precipitation_deposits', gcm.EmpiricalDistribution())
        self.causal_model.set_causal_mechanism('sun_altitude_angle', gcm.EmpiricalDistribution())

        # Specify intermediate nodes as linear noise models
        self.causal_model.set_causal_mechanism('Brenner',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('TeRES',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('Tenengrad',
                                               gcm.AdditiveNoiseModel(
                                                   gcm.ml.regression.create_random_forest_regressor()))
        self.causal_model.set_causal_mechanism('Variance',
                                               gcm.AdditiveNoiseModel(
                                                   gcm.ml.regression.create_hist_gradient_boost_regressor()))
        self.causal_model.set_causal_mechanism('inception',
                                               gcm.AdditiveNoiseModel(
                                                   gcm.ml.regression.create_random_forest_regressor()))
        self.causal_model.set_causal_mechanism('lapulaseDetection',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('niqe',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('orb',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('phase',
                                               gcm.AdditiveNoiseModel(
                                                   gcm.ml.regression.create_support_vector_regressor()))
        self.causal_model.set_causal_mechanism('sift',
                                               gcm.AdditiveNoiseModel(
                                                   gcm.ml.regression.create_support_vector_regressor()))
        self.causal_model.set_causal_mechanism('std',
                                               gcm.AdditiveNoiseModel(
                                                   gcm.ml.regression.create_hist_gradient_boost_regressor()))
        self.causal_model.set_causal_mechanism('PilotNet',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('cg23',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('cnn_gru',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('stacked_cnn',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))
        self.causal_model.set_causal_mechanism('two_stream_cnn',
                                               gcm.AdditiveNoiseModel(gcm.ml.regression.create_extra_trees_regressor()))

        gcm.fit(self.causal_model, self.observational_data)
        # Create KNN model to find the most similar observation
        self.knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        self.knn.fit(self.scaled_current_data)

    def find_similar_item(self, current_parameter):
        # Given the parameter combination of current causal variables, find the closest observation
        scaled_current_parameter = self.scaler.transform(np.array([current_parameter]))
        # Find the most similar record to the new data in the observational dataset
        distances, indices = self.knn.kneighbors(scaled_current_parameter)
        # Restore the result after feature scaling
        most_similar_distance = distances[0][0]
        self.most_similar_index = indices[0][0]
        self.most_similar_record = self.observational_data.iloc[self.most_similar_index].to_frame().transpose()

        return most_similar_distance

    def cal_counterfactural(self, current_parameter):
        # First, get the most similar observed_parameters
        new_cloudiness, new_dust_storm, new_roadtype, new_precipitation, new_fog_density, \
            new_traffic_density, new_precipitation_deposits, new_sun_altitude_angle = current_parameter[0:8]
        # Select the corresponding sample in the database and calculate the counterfactual result
        _ = self.find_similar_item(current_parameter)
        counterfactual = gcm.counterfactual_samples(
            self.causal_model,
            {'cloudiness': lambda x: new_cloudiness, 'dust_storm': lambda x: new_dust_storm,
             'roadtype': lambda x: new_roadtype, 'precipitation': lambda x: new_precipitation,
             'fog_density': lambda x: new_fog_density, 'traffic_density': lambda x: new_traffic_density,
             'precipitation_deposits': lambda x: new_precipitation_deposits,
             'sun_altitude_angle': lambda x: new_sun_altitude_angle
             },
            observed_data=self.most_similar_record)

        # Return the counterfactual result
        current_counterfactual = []
        for factor in self.factor_to_predict:
            current_counterfactual.append(counterfactual.loc[0, [factor]].values)

        return current_counterfactual


def inference_counterfactual(observed_parameters, decision_variables, train_data, factor_to_predict):
    causal_model = Causal4End2End(decision_variables, train_data=train_data, factor_to_predict=factor_to_predict)
    # Select the corresponding sample in the database and calculate the counterfactual result
    counterfactuals = []
    for idx in range(len(observed_parameters)):
        current_parameter = observed_parameters[idx, :]
        current_counterfactual = causal_model.cal_counterfactural(current_parameter)
        counterfactuals.append(current_counterfactual)

    return np.array(counterfactuals)


if __name__ == '__main__':
    decision_variables = ['cloudiness', 'dust_storm', 'roadtype', 'precipitation', 'fog_density',
                          'traffic_density', 'precipitation_deposits', 'sun_altitude_angle']

    data = pd.read_excel('./structures/CausalAnalysis_continuous_demo.xlsx')
    target = ['two_stream_cnn']
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_list = []
    r2_list = []

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        counterfactuals = inference_counterfactual(observed_parameters=test_data.loc[:, decision_variables].values,
                                                   decision_variables=decision_variables, train_data=train_data, factor_to_predict=target)
        real_values = test_data.loc[:, 'two_stream_cnn'].values
        mse = np.median((counterfactuals - real_values) ** 2)
        r2 = 1 - mse / np.var(real_values)
        mse_list.append(mse)
        r2_list.append(r2)

    # Calculate the average error
    mean_mse = np.mean(mse_list)
    mean_r2 = np.mean(r2_list)
    print(f"Mean MSE: {mean_mse}")
    print(f"Mean R^2: {mean_r2}")

    # Plot line chart to compare predicted values and real values
    plt.plot(real_values, label='real_values')
    plt.plot(np.squeeze(counterfactuals), label='counterfactuals')
    plt.legend()
    plt.show()

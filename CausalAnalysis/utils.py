def variable_to_string(variables):
    """
    Pass in a numerical vector, convert each value to a string according to its range, and return a string vector.
    :param variable: The value of each node
    :return: String vector
    """
    string_vector = []
    cloudiness_value = variables[0]
    dust_storm_value = variables[1]
    roadtype_value = variables[2]
    precipitation_value = variables[3]
    fog_density_value = variables[4]
    traffic_density_value = variables[5]
    precipitation_deposits_value = variables[6]
    sun_altitude_angle_value = variables[7]

    # According to the estimated causal effect, sort prompt words by importance from high to low and apply corresponding weights
    if cloudiness_value == 0:
        cloudiness = "No clouds"
        cloudiness_causal_effect = 0.0
    elif 0 < cloudiness_value < 5:
        cloudiness = "a few clouds scattered in the sky"
        cloudiness_causal_effect = 0.017794909
    elif 5 <= cloudiness_value < 35:
        cloudiness = "some clouds covering a small portion of the sky"
        cloudiness_causal_effect = 0.015007036
    elif 35 <= cloudiness_value < 60:
        cloudiness = "a moderate amount of clouds"
        cloudiness_causal_effect = 0.030803595
    else:
        cloudiness = "mostly cloudy skies"
        cloudiness_causal_effect = 0.041185199
    string_vector.append(cloudiness)

    if 0 <= dust_storm_value <= 5:
        dust_storm = "clear sky"
        dust_storm_causal_effect = 0.0
    elif 5 < dust_storm_value <= 15:
        dust_storm = "a light dust storm"
        dust_storm_causal_effect = 0.000248429
    elif 15 <= dust_storm_value < 30:
        dust_storm = "a small dust storm"
        dust_storm_causal_effect = 0.001363336
    elif 30 <= dust_storm_value < 90:
        dust_storm = "a moderate dust storm"
        dust_storm_causal_effect = 0.001313716
    else:
        dust_storm = "a big dust storm"
        dust_storm_causal_effect = 0.009379442
    string_vector.append(dust_storm)

    if roadtype_value == 1:
        roadtype = "small town with diverse buildings"
        roadtype_causal_effect = 0.192045411
    elif roadtype_value == 2:
        roadtype = "suburban road with residential and commercial area"
        roadtype_causal_effect = 0.285759229
    elif roadtype_value == 3:
        roadtype = "urban road"
        roadtype_causal_effect = 0.153134908
    elif roadtype_value == 4:
        roadtype = "multi-lane road"
        roadtype_causal_effect = 0.0
    elif roadtype_value == 5:
        roadtype = "urban area with raised highway"
        roadtype_causal_effect = 0.120081787
    elif roadtype_value == 6:
        roadtype = "Long highways with multiple entrances/exits"
        roadtype_causal_effect = 0.017108997
    elif roadtype_value == 7:
        roadtype = "rural area"
        roadtype_causal_effect = 0.271547566
    else:
        roadtype = "downtown area with tall buildings and cityscape"
        roadtype_causal_effect = 0.057879417
    string_vector.append(roadtype)

    if precipitation_value == 0:
        precipitation = "no rain at all"
        precipitation_causal_effect = 0.0
    elif 0 < precipitation_value < 10:
        precipitation = "a light drizzle or a few drops of rain"
        precipitation_causal_effect = 0.0019871
    elif 10 <= precipitation_value < 30:
        precipitation = "a consistent but gentle rain"
        precipitation_causal_effect = 0.00154094
    elif 30 <= precipitation_value < 50:
        precipitation = "a steady and noticeable rain"
        precipitation_causal_effect = 0.00350668
    elif 50 <= precipitation_value < 80:
        precipitation = "a strong rain with large raindrops"
        precipitation_causal_effect = 0.004308
    else:
        precipitation = "torrential rain with heavy downpour"
        precipitation_causal_effect = 0.007421
    string_vector.append(precipitation)

    if fog_density_value == 0:
        fog_density = "no fog"
        fog_causal_effect = 0.0
    elif 0 < fog_density_value < 15:
        fog_density = "a slight mist"
        fog_causal_effect = 0.006206589
    elif 15 <= fog_density_value < 35:
        fog_density = "a thin fog"
        fog_causal_effect = 0.011286806
    elif 35 <= fog_density_value < 65:
        fog_density = "a moderate fog"
        fog_causal_effect = 0.018120066
    else:
        fog_density = "a very dense fog"
        fog_causal_effect = 0.023157324
    string_vector.append(fog_density)

    if 0 < traffic_density_value < 15:
        traffic_density = "few vehicles on the road"
        traffic_causal_effect = 0.0
    elif 15 <= traffic_density_value < 35:
        traffic_density = "a steady flow of vehicles"
        traffic_causal_effect = 0.006771929
    elif 35 <= traffic_density_value < 65:
        traffic_density = "a moderate amount of traffic"
        traffic_causal_effect = 0.011091377
    else:
        traffic_density = "significant congestion on the road"
        traffic_causal_effect = 0.020183336
    string_vector.append(traffic_density)

    if precipitation_deposits_value == 0:
        precipitation_deposits = "no precipitation deposits"
        deposits_causal_effect = 0.0
    elif 0 < precipitation_deposits_value < 15:
        precipitation_deposits = "light precipitation deposits"
        deposits_causal_effect = 0.0
    elif 15 <= precipitation_deposits_value < 30:
        precipitation_deposits = "small precipitation deposits"
        deposits_causal_effect = 0.002561995
    elif 30 <= precipitation_deposits_value < 50:
        precipitation_deposits = "moderate precipitation deposits"
        deposits_causal_effect = 0.001276832
    else:
        precipitation_deposits = "very large precipitation deposit"
        deposits_causal_effect = 0.015960896
    string_vector.append(precipitation_deposits)

    if sun_altitude_angle_value == -90:
        sun_altitude_angle = "midnight"
        sun_causal_effect = 0.040443856
    elif -90 < sun_altitude_angle_value <= 0:
        sun_altitude_angle = "night"
        sun_causal_effect = 0.022562955
    elif 0 < sun_altitude_angle_value < 20:
        sun_altitude_angle = "sunrise"
        sun_causal_effect = 0.004643442
    elif 20 <= sun_altitude_angle_value < 60:
        sun_altitude_angle = "morning"
        sun_causal_effect = 0.012983781
    else:
        sun_altitude_angle = "noon"
        sun_causal_effect = 0.0
    string_vector.append(sun_altitude_angle)

    # Pair causal effect values with their corresponding strings
    effect_string_pairs = list(zip([cloudiness_causal_effect, dust_storm_causal_effect, roadtype_causal_effect, precipitation_causal_effect,
                                    fog_causal_effect, traffic_causal_effect, deposits_causal_effect, sun_causal_effect], string_vector))

    # Sort by causal effect value from high to low
    effect_string_pairs.sort(reverse=True, key=lambda x: x[0])

    # Extract the sorted string list
    sorted_causal_effects = [pair[0] for pair in effect_string_pairs]
    sorted_string_vector = [pair[1] for pair in effect_string_pairs]
    paired_results = calculate_weights(sorted_causal_effects, sorted_string_vector)
    combined_string = ', '.join(sorted_string_vector)

    return combined_string, paired_results


def calculate_weights(sorted_causal_effects, sorted_string_vector):
    """
    :param sorted_causal_effects: List of causal effect values sorted from high to low
    :param sorted_string_vector: List of prompt words sorted by causal effect value from high to low
    :return:
    """
    min_effect = min(sorted_causal_effects)
    max_effect = max(sorted_causal_effects)
    weights = [1 + (effect - min_effect) / (max_effect - min_effect) * (1.1 - 1) for effect in sorted_causal_effects]
    paired_results = list(zip(sorted_string_vector, [round(weight, 3) for weight in weights]))

    return paired_results



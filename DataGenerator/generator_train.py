"""
Generate dataset for end-to-end model training
"""
from DataSave import DataSave
from SynchronyModel import SynchronyModel
import carla
import psutil
import os
import re  


# Get CARLA weather presets
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')  
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))  
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)] 
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]  

def main():
    town_string = "Town02"  # Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10
    weather_preset_list = find_weather_presets()
    for idx in range(0, len(weather_preset_list)):
        weather_preset, weather_preset_str = weather_preset_list[idx]   
        weather_preset_str = weather_preset_str.replace(" ", "_")
        print(f"we have tested the weather preset {weather_preset_str}")

        infodict = {'roadtype': town_string, 'weather': weather_preset, 'traffic_density': 15}

        index = len(os.listdir('../Autopilot/driving_dataset/carla_collect'))
        print(f"we are now in the {index} th environment")
        synmodel = SynchronyModel(infodict)
        dtsave = DataSave(index)
        try:
            synmodel.set_synchrony()
            synmodel.spawn_agent()
            synmodel.spawn_actors()
            synmodel.set_actors_route()
            synmodel.sensor_listen()
            spectator = synmodel.world.get_spectator()
            step = 0
            interval = 10
            while step <= 10000:
                agent = synmodel.actors["agents"][0]
                loc = agent.get_transform().location
                spectator.set_transform(
                    carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))
                if agent.is_at_traffic_light():
                    traffic_light = agent.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)

                if step % interval == 0 and step != 0:
                    ret_data = synmodel.update()
                    dtsave.save_training_files(ret_data)
                    print(step / interval)
                else:
                    synmodel.world.tick()

                existing_files_num = dtsave.existing_data_files()  
                if existing_files_num == 200:
                    break
                step += 1
        finally:
            synmodel.setting_recover()

    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == 'CarlaUE4-Win64-Shipping.exe':
            cmd = 'taskkill /F /IM CarlaUE4-Win64-Shipping.exe'
            os.system(cmd)


if __name__ == '__main__':
    main()

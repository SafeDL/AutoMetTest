"""
(1) Traverse multiple Carla simulation environments configured in the Excel file, collecting images and steering data for each environment.
(2) This script is mainly used for observational testing in causal analysis.
"""
from DataSave import DataSave
from SynchronyModel import SynchronyModel
import carla
import psutil
import os
import pandas as pd


def cfg_from_excel_file(excel_file):
    data = pd.read_excel(excel_file)
    cfg = data.to_dict(orient='records')
    return cfg


def main():
    excel_file = './carlaenv.xlsx'
    cfg = cfg_from_excel_file(excel_file)

    for index in range(0, len(cfg)):
        print(f"\nwe are now in the condition_{index} th environment")
        infodict = cfg[index] 
        synmodel = SynchronyModel(infodict)
        dtsave = DataSave(index)  

        synmodel.set_synchrony()
        synmodel.spawn_agent()
        synmodel.spawn_actors()
        synmodel.set_actors_route()
        synmodel.sensor_listen()
        spectator = synmodel.world.get_spectator()
        step = 0
        interval = 10
        while step <= 20000:
            agent = synmodel.actors["agents"][0]
            loc = agent.get_transform().location
            spectator.set_transform(
                carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))
            if agent.is_at_traffic_light():
                traffic_light = agent.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            if step % interval == 0 and agent.get_velocity().x > 1:
                ret_data = synmodel.update()
                dtsave.save_training_files(ret_data)
            else:
                synmodel.world.tick()

            existing_files_num = dtsave.existing_data_files()  # Returns the current number of images
            if existing_files_num == 200:
                break
            step += 1

    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == 'CarlaUE4-Win64-Shipping.exe':
            cmd = 'taskkill /F /IM CarlaUE4-Win64-Shipping.exe'
            os.system(cmd)


if __name__ == '__main__':
    main()

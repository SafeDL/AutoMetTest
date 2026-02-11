import queue
import random
import carla
import logging
import numpy as np
import math


class SynchronyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)
        town = str(cfg['roadtype'])
        print(f"Set the test environment to {town}")
        self.world = self.client.load_world(town)

        # Import predefined weather here
        # weather = cfg['weather']

        # Customize weather for observational dataset here
        weather = carla.WeatherParameters(cloudiness=cfg['cloudiness'], precipitation=cfg['precipitation'],
                                          precipitation_deposits=cfg['precipitation_deposits'],
                                          wind_intensity=cfg['wind_intensity'], sun_azimuth_angle=cfg['sun_azimuth_angle'],
                                          sun_altitude_angle=cfg['sun_altitude_angle'],
                                          fog_density=cfg['fog_density'], fog_distance=0.0, wetness=cfg['wetness'],
                                          fog_falloff=0.0,
                                          scattering_intensity=cfg['scattering_intensity'], mie_scattering_scale=cfg['mie_scattering_scale'],
                                          rayleigh_scattering_scale=cfg['rayleigh_scattering_scale'], dust_storm=cfg['dust_storm'])


        self.world.set_weather(weather)
        self.traffic_manager = self.client.get_trafficmanager()
        self.init_settings = None
        self.frame = None
        self.actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
        self.data = {"sensor_data": {}}
        self.vehicle = None

    def set_synchrony(self):
        self.init_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.traffic_manager.set_synchronous_mode(True)
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def setting_recover(self):
        # Destroy sensors and agents
        for agent in self.actors["agents"]:
            for sensor in self.actors["sensors"][agent]:
                sensor.destroy()
            agent.destroy()

        batch = []
        for non_agent_actor_id in self.actors["non_agents"]:
            batch.append(carla.command.DestroyActor(non_agent_actor_id))
        for walker_id in self.actors["walkers"]:
            batch.append(carla.command.DestroyActor(walker_id))
        self.client.apply_batch_sync(batch)
        self.world.apply_settings(self.init_settings)

    def spawn_actors(self):
        num_of_vehicles = self.cfg['traffic_density']
        num_of_walkers = 0

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # Get spawn points near main car
        car_spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(car_spawn_points)

        # Generate traffic vehicles, unreasonable vehicle types in small maps may cause traffic jams
        town = str(self.cfg['roadtype'])
        TownWithoutTruck = ["Town01", "Town02", "Town07"]

        if num_of_vehicles < number_of_spawn_points:
            random.shuffle(car_spawn_points)
        elif num_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, num_of_vehicles, number_of_spawn_points)
            num_of_vehicles = number_of_spawn_points

        batch = []
        for n, transform in enumerate(car_spawn_points):
            if n >= num_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if town in TownWithoutTruck and blueprint.id == "vehicle.carlamotors.firetruck":
                continue
            if blueprint.has_attribute('driver_id'):
                driver_id = blueprint.get_attribute('driver_id').recommended_values[0]
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform))
            car_spawn_points.pop(n)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                continue
            else:
                self.actors["non_agents"].append(response.actor_id)

        # Generate pedestrians
        blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        ped_spawn_points = []
        for i in range(num_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                ped_spawn_points.append(spawn_point)

        batch = []
        for spawn_point in ped_spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                continue
            else:
                self.actors["walkers"].append(response.actor_id)
        print("spawn {} vehicles and {} walkers".format(len(self.actors["non_agents"]),
                                                        len(self.actors["walkers"])))
        self.world.tick()

    def set_actors_route(self):
        vehicle_actors = self.world.get_actors(self.actors["non_agents"])
        for vehicle in vehicle_actors:
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        batch = []
        for i in range(len(self.actors["walkers"])):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(),
                                                  self.actors["walkers"][i]))
        controllers_id = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                controllers_id.append(response.actor_id)
        self.world.set_pedestrians_cross_factor(0.2)

        # Control pedestrians to walk
        for con_id in controllers_id:
            self.world.get_actor(con_id).start()
            destination = self.world.get_random_location_from_navigation()
            self.world.get_actor(con_id).go_to_location(destination)

    # Randomly generate ego_vehicle and sensors
    def spawn_agent(self):
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
        ego_transform = random.choice(self.world.get_map().get_spawn_points())
        agent = self.world.spawn_actor(vehicle_bp, ego_transform)
        agent.set_autopilot(True)
        self.actors["agents"].append(agent)
        self.actors["sensors"][agent] = []

        # Configure RGB camera sensor
        cam_rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_rgb_bp.set_attribute('image_size_x', f'{640}')
        cam_rgb_bp.set_attribute('image_size_y', f'{384}')
        cam_rgb_bp.set_attribute('fov', '90')
        cam_rgb_bp.set_attribute('exposure_mode', 'fixed') 
        cam_rgb_bp.set_attribute('exposure_compensation', '0.5')  
        cam_rgb_bp.set_attribute('motion_blur_intensity', '0.1')
        cam_rgb_transform = carla.Transform(carla.Location(x=0.5, z=1.6), carla.Rotation(roll=0.0, pitch=5.0, yaw=0.0))
        cam_rgb_sensor = self.world.spawn_actor(cam_rgb_bp, cam_rgb_transform, attach_to=agent)
        self.actors["sensors"][agent].append(cam_rgb_sensor)

        # Configure semantic segmentation sensor
        # cam_sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # cam_sem_bp.set_attribute("image_size_x", f"{640}")
        # cam_sem_bp.set_attribute("image_size_y", f"{384}")
        # cam_sem_bp.set_attribute("fov", str(90))
        # cam_sem_transform = carla.Transform(carla.Location(x=0.5, z=1.6), carla.Rotation(roll=0.0, pitch=5.0, yaw=0.0))
        # cam_sem_sensor = self.world.spawn_actor(cam_sem_bp, cam_sem_transform, attach_to=agent, attachment_type=carla.AttachmentType.Rigid)
        # self.actors["sensors"][agent].append(cam_sem_sensor)

        self.world.tick()

    # Save data from RGB and semantic sensors
    def sensor_listen(self):
        for agent, sensors in self.actors["sensors"].items():
            self.data["sensor_data"][agent] = []
            for sensor in sensors:
                q = queue.Queue()
                self.data["sensor_data"][agent].append(q)
                sensor.listen(q.put)

    def update(self):
        ret = {"environment_objects": None, "actors": None, "agents_data": {}}
        self.frame = self.world.tick()
        ret["actors"] = self.world.get_actors()

        for agent, dataQue in self.data["sensor_data"].items():
            data = [self._retrieve_data(q) for q in dataQue]
            assert all(x.frame == self.frame for x in data)
            ret["agents_data"][agent] = {}
            ret["agents_data"][agent]["sensor_data"] = data

            # Get the steering angle of the main car in Carla
            control = agent.get_control()
            ret["agents_data"][agent]["control"] = control.steer * 180

            # Get camera parameters
            ret["agents_data"][agent]["intrinsic"] = self._camera_intrinsic(640, 384)
            ret["agents_data"][agent]["extrinsic"] = np.mat(
                self.actors["sensors"][agent][0].get_transform().get_matrix())

        return ret

    def _camera_intrinsic(self, width, height):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k

    def _retrieve_data(self, q):
        while True:
            data = q.get()
            if data.frame == self.frame:
                return data

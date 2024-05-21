from environment.grid.simple_grid import MultiRoomGrid


class GymMultiGrid(MultiRoomGrid):
    def __init__(self, config):
        super().__init__(
            config=config["config"],
            start_rooms=config["start_rooms"],
            goal_rooms=config["goal_rooms"],
            room_size=config["room_size"],
            max_steps=config["max_steps"],
            num_rubble=config["num_rubble"],
            rubble_reward = config["rubble_reward"],
        )
# Reward Signal testing to insure:
# 1. Lander will crash at accelerated velocity (y_velocity < -3);
#    downward velocity greater than 3
# 2. Lander will crash at accelerated horizontal velocity
#    (x_velocity < -10 or 10 < x_velocity)
# 3. Lander will crash at non-zero vetical position, with 5 degree acceptable variance
#    (5 < angle < 355 ; vertical is at 0 degrees, and lander can be between [0, 359])
# 4. Lander will crash if it runs out of fuel (fuel <= 0)
# 5. Least amount of fuel used possible (to save money)
# 6. Lander will crash if not landed within landing zone (x_position != landing_zone)

import environment
from utils import get_landing_zone, get_angle, get_velocity, get_position, get_fuel, tests
get_landing_zone()
# Lunar Lander Environment
class LunarLanderEnvironment(environment.BaseEnvironment):
    def __init__(self):
        self.current_state = None
        self.count = 0

    def env_init(self, env_info):
        # users set this up
        self.state = np.zeros(6) # velocity x, y, angle, distance to ground, landing zone x, y

    def env_start(self):
        land_x, land_y = get_landing_zone() # gets the x, y coordinate of the landing zone
        # At the start we initialize the agent to the top left hand corner (100, 20) with 0 velocity
        # in either any direction. The agent's angle is set to 0 and the landing zone is retrieved and set.
        # The lander starts with fuel of 100.
        # (vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel)
        self.current_state = (0, 0, 0, 100, 20, land_x, land_y, 100)
        return self.current_state

    def env_step(self, action):

        land_x, land_y = get_landing_zone() # gets the x, y coordinate of the landing zone
        vel_x, vel_y = get_velocity(action) # gets the x, y velocity of the lander
        angle = get_angle(action) # gets the angle the lander is positioned in
        pos_x, pos_y = get_position(action) # gets the x, y position of the lander
        fuel = get_fuel(action) # get the amount of fuel remaining for the lander

        terminal = False
        reward = 0.0
        observation = (vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel)

        # use the above observations to decide what the reward will be, and if the
        # agent is in a terminal state.
        # Recall - if the agent crashes or lands terminal needs to be set to True

        # See if lander has crashed
        lander_crashed = False
        landing_success = False
        if fuel <= 0:  # ran out of fuel
            lander_crashed = True
        if pos_y == land_y:  # if lander has touched the groud
            if (vel_y < -3) or (vel_x < -10) or (10 < vel_x):  # lander going too fast
                lander_crashed = True
            elif (5 < angle < 355):  # lander not vertical
                lander_crashed = True
            elif (pos_x != land_x):
                lander_crashed = True
            else:
                landing_success = True

        # calc reward
        if landing_success:
            reward = 10000 -(100 - fuel)
            terminal = True
        elif lander_crashed:
            reward = -10000
            terminal = True
        else:
            reward += -(100 - fuel)  # cost of using fuel
            reward -= 1  # cost of taking a step without a reward

        self.reward_obs_term = (reward, observation, terminal)
        return self.reward_obs_term

    def env_cleanup(self):
        return None

    def env_message(self):
        return None

tests(LunarLanderEnvironment, 1)

tests(LunarLanderEnvironment, 2)

tests(LunarLanderEnvironment, 3)

tests(LunarLanderEnvironment, 4)

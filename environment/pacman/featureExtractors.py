# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import numpy as np

from .definitions import Directions, Actions
from . import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

def modifiedClosestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    backtrace = {}
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)

        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            while (pos_x, pos_y) in backtrace and backtrace[(pos_x, pos_y)] != (pos[0], pos[1]):
                pos_x, pos_y = backtrace[(pos_x, pos_y)]

            # Return the vector direction that the closest food is in
            return pos_x - pos[0], pos_y - pos[1]

        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            if (nbr_x, nbr_y) not in expanded:
                fringe.append((nbr_x, nbr_y, dist+1))
                backtrace[(nbr_x, nbr_y)] = (pos_x, pos_y)

    # no food found
    return None

# def multipleClosestFood(pos, food, walls, store_path=False):
#     backtrace = {}
#     fringe = [(pos[0], pos[1], 0, None)]  # Store initial action as None
#     expanded = set()
#     min_distance = float('inf')
#     closest_foods_actions = []
#     closest_foods_paths = [] # find the path to the closest food

#     while fringe:
#         pos_x, pos_y, dist, initial_action = fringe.pop(0)
#         if (pos_x, pos_y) in expanded:
#             continue

#         expanded.add((pos_x, pos_y))

#         if food[pos_x][pos_y]:
#             if store_path:
#                 path = reconstruct_path(backtrace, (pos_x, pos_y), pos)
#             if dist < min_distance:
#                 min_distance = dist
#                 closest_foods_actions = [initial_action]
#                 if store_path:
#                     closest_foods_paths = [path]
#             elif dist == min_distance:
#                 closest_foods_actions.append(initial_action)
#                 if store_path:
#                     closest_foods_paths.append(path)
#             continue

#         nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
#         for nbr_x, nbr_y in nbrs:
#             if (nbr_x, nbr_y) not in expanded:
#                 next_action = (nbr_x - pos_x, nbr_y - pos_y)
#                 if initial_action is None:  # Store the direction for the first step
#                     new_action = next_action
#                 else:
#                     new_action = initial_action  # Pass along the initial action
#                 fringe.append((nbr_x, nbr_y, dist + 1, new_action))
#                 backtrace[(nbr_x, nbr_y)] = (pos_x, pos_y)

#     return closest_foods_actions, closest_foods_paths



def multipleClosestFood(pos, food, walls, store_food_pos=False):
    fringe = [(pos[0], pos[1], 0, None)]  # (x, y, distance, initial action)
    expanded = set()
    min_distance = float('inf')
    closest_foods_actions = []
    closest_foods_positions = []  # Store positions of the closest food

    while fringe:
        pos_x, pos_y, dist, initial_action = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue

        expanded.add((pos_x, pos_y))

        if food[pos_x][pos_y]:
            if dist < min_distance:
                min_distance = dist
                closest_foods_actions = [initial_action]
                if store_food_pos:
                    closest_foods_positions = [(pos_x, pos_y)]
            elif dist == min_distance:
                closest_foods_actions.append(initial_action)
                if store_food_pos:
                    closest_foods_positions.append((pos_x, pos_y))
            continue

        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            if (nbr_x, nbr_y) not in expanded:
                next_action = (nbr_x - pos_x, nbr_y - pos_y)
                if initial_action is None:  # Determine the direction for the first step
                    new_action = next_action
                else:
                    new_action = initial_action  # Pass along the initial action
                fringe.append((nbr_x, nbr_y, dist + 1, new_action))

    return closest_foods_actions, closest_foods_positions





def reconstruct_path(backtrace, end_pos, start_pos):
    path = []
    current_pos = end_pos
    while current_pos != start_pos:
        prev_pos = backtrace[current_pos]
        move = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
        path.insert(0, move)  # Insert at the beginning to build the path in correct order
        current_pos = prev_pos
    return path








# def closestTargetActions(pos, targets, walls, distance_threshold=20):
#     """
#     Find all initial actions to take toward the closest targets (ghosts or capsules).

#     :param pos: The starting position (x, y) of Pacman.
#     :param targets: A list of target positions [(x1, y1), (x2, y2), ...].
#     :param walls: A matrix representing walls on the grid.
#     :return: A list of initial actions to take towards all closest targets.
#     """
#     backtrace = {}
#     fringe = [(pos[0], pos[1], 0, None)]  # (x, y, distance, initial action)
#     expanded = set()
#     min_distance = float('inf')
#     closest_actions = []
#     closest_target_indices = []

#     while fringe:
#         pos_x, pos_y, dist, initial_action = fringe.pop(0)

#         if (pos_x, pos_y) in expanded:
#             continue

#         expanded.add((pos_x, pos_y))


#         # if (pos_x, pos_y) in targets and dist <= distance_threshold:
#         #     if dist < min_distance:
#         #         min_distance = dist
#         #         closest_actions = [initial_action] if initial_action is not None else []
#         #         closest_target_indices = [targets.index((pos_x, pos_y))]
#         #     elif dist == min_distance:
#         #         if initial_action is not None:
#         #             closest_actions.append(initial_action)
#         #             closest_target_indices.append(targets.index((pos_x, pos_y)))


#         if (pos_x, pos_y) in targets:
#             if dist < min_distance:
#                 min_distance = dist
#                 if initial_action == None:
#                     closest_actions = []
#                 else:
#                     closest_actions = [initial_action]  # Reset with the new closest action
#                 closest_target_indices = [targets.index((pos_x, pos_y))]
#             elif dist == min_distance:
#                 closest_actions.append(initial_action)  # Add equally close action
#                 closest_target_indices.append(targets.index((pos_x, pos_y)))
#             continue  # Keep looking for all closest targets at the minimal distance


#         nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
#         for nbr_x, nbr_y in nbrs:
#             if (nbr_x, nbr_y) not in expanded:
#                 next_action = Actions.vectorToDirection((nbr_x - pos_x, nbr_y - pos_y))
#                 if initial_action is None:  # Store the first step's action
#                     new_action = next_action
#                 else:
#                     new_action = initial_action  # Pass along the initial action
#                 fringe.append((nbr_x, nbr_y, dist + 1, new_action))

#     return closest_actions, closest_target_indices
def closestTargetActions(pos, targets, walls, distance_threshold=20, store_closest_target=False):
    """
    Find all initial actions to take toward the closest targets (ghosts or capsules) and optionally store the closest targets.

    :param pos: The starting position (x, y) of Pacman.
    :param targets: A list of target positions [(x1, y1), (x2, y2), ...].
    :param walls: A matrix representing walls on the grid.
    :param distance_threshold: The maximum distance to consider for a target to be 'closest'.
    :param store_closest_target: Boolean indicating whether to store the locations of the closest targets.
    :return: A tuple containing a list of initial actions, indices of closest targets, and optionally, the positions of the closest targets.
    """
    backtrace = {}
    fringe = [(pos[0], pos[1], 0, None)]  # (x, y, distance, initial action)
    expanded = set()
    min_distance = float('inf')
    closest_actions = []
    closest_target_indices = []
    closest_targets_positions = []  # Store positions of the closest targets if required

    while fringe:
        pos_x, pos_y, dist, initial_action = fringe.pop(0)

        if (pos_x, pos_y) in expanded:
            continue

        expanded.add((pos_x, pos_y))

        if (pos_x, pos_y) in targets:
            target_index = targets.index((pos_x, pos_y))
            if dist < min_distance:
                min_distance = dist
                closest_actions = [initial_action] if initial_action is not None else []
                closest_target_indices = [target_index]
                if store_closest_target:
                    closest_targets_positions = [(pos_x, pos_y)]
            elif dist == min_distance:
                if initial_action is not None:
                    closest_actions.append(initial_action)
                closest_target_indices.append(target_index)
                if store_closest_target:
                    closest_targets_positions.append((pos_x, pos_y))

        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            if (nbr_x, nbr_y) not in expanded:
                next_action = Actions.vectorToDirection((nbr_x - pos_x, nbr_y - pos_y))
                if initial_action is None:
                    new_action = next_action
                else:
                    new_action = initial_action
                fringe.append((nbr_x, nbr_y, dist + 1, new_action))

    # Return the closest targets' positions along with actions and indices if requested
    if store_closest_target:
        return closest_actions, closest_target_indices, closest_targets_positions
    else:
        return closest_actions, closest_target_indices










class VectorFeatureExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """
    def _findGhosts(self, pos, dir_vec, walls, ghosts, ghost_radius=2):
        # Finds ghosts down a particular direction
        x, y = pos
        dx, dy = dir_vec

        next_x = x + dx
        next_y = y + dy

        if (next_x < 0 or next_x >= walls.width) or (next_y < 0 or next_y >= walls.height) or (walls[next_x][next_y]):
            return False

        fringe = [(next_x, next_y, 0)]
        expanded = set()

        # Add the original starting node to the expanded list so we don't re-explore it
        expanded.add((x, y))
        
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)

            # Skip node if we have already explored it, the depth is greater than ghost_radius
            if (pos_x, pos_y) in expanded or dist > ghost_radius:
                continue

            expanded.add((pos_x, pos_y))

            # If we find a ghost here, then return
            for ghost in ghosts:
                if ghost == (pos_x, pos_y):
                    return True

            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)

            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))

        # No ghost found
        return False

    # def getUnconditionedFeatures(self, state):
    #     # extract the grid of food and wall locations and get the ghost locations
    #     food = state.getFood()
    #     walls = state.getWalls()
    #     ghosts = state.getGhostPositions()
    #     x, y = state.getPacmanPosition()
        

    #     # print ('food', food)
    #     # print ('walls', walls)
    #     # print ('ghosts', ghosts)
    #     # print ('x, y', x, y)

    #     closest_food = modifiedClosestFood((x, y), food, walls)
    #     food_features = [0, 0, 0, 0]
    #     if closest_food is not None:
    #         food_features[Actions._actionToInt[Actions.vectorToDirection(closest_food)]] = 1

    #     ghost_features = []
    #     wall_features = []

    #     # Search whether there's a neighboring wall
    #     x_int, y_int = int(x + 0.5), int(y + 0.5)
    #     for dir, vec in Actions._directionsAsList:
    #         dx, dy = vec
    #         next_x = x_int + dx
    #         next_y = y_int + dy
    #         if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
    #             wall_features.append(1)
    #         else:
    #             wall_features.append(0)

    #     # Search whether there's a ghost within ghost_radius tiles
    #     ghost_radius = 3
    #     x_int, y_int = int(x + 0.5), int(y + 0.5)

    #     for dir, vec in Actions._directionsAsList:
    #         if self._findGhosts((x_int, y_int), vec, walls, ghosts, ghost_radius=ghost_radius):
    #             ghost_features.append(1)
    #         else:
    #             ghost_features.append(0)

    #     ghost_mode = 0

    #     # for index in range(1, len(state.data.agentStates)):
    #     #     if state.data.agentStates[index].scaredTimer > 0:
    #     #         ghost_mode = 1
    #     #         break

    #     features = np.concatenate((wall_features, ghost_features, food_features, [ghost_mode])).astype(int)

    #     return features.reshape(13, 1, 1)





    def getUnconditionedFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        x, y = state.getPacmanPosition()
        capsules = state.getCapsules()
        
        # print (str(state))

        # Update in main logic
        closest_foods, _ = multipleClosestFood((x, y), food, walls)
        food_features = [0, 0, 0, 0]
        if closest_foods:
            for food_dir in closest_foods:
                direction = Actions.vectorToDirection(food_dir)
                food_features[Actions._actionToInt[direction]] = 1
        # print ('food features (up, down, right, left)', food_features)

        ghost_features = []
        wall_features = []

        # Search whether there's a neighboring wall
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
                wall_features.append(1)
            else:
                wall_features.append(0)


        # print ('ghosts', ghosts)
        # Search whether there's a ghost within ghost_radius tiles
        ghost_radius = 2 #6 #3
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        for dir, vec in Actions._directionsAsList:
            if self._findGhosts((x_int, y_int), vec, walls, ghosts, ghost_radius=ghost_radius):
                ghost_features.append(1)
            else:
                ghost_features.append(0)
        # state.data.ghost_nearby = ghost_features


        # modify ghost far_features to be the closest ghost
        ghost_far_features = [0, 0, 0, 0]
        int_ghosts = [(int(x), int(y)) for x, y in ghosts]
        closest_actions, closest_ghost_indices = closestTargetActions((x_int, y_int), int_ghosts, walls)
        for closest_action in closest_actions:
            ghost_far_features[Actions._actionToInt[closest_action]] = 1
        # print ('closest ghost (up, down, right, left)', ghost_far_features)


        ghost_mode = [0, 0, 0, 0] # same features except for the ghost mode, depending on timer

        # for index in range(1, len(state.data.agentStates)):
        for index in closest_ghost_indices:
            timer = state.data.agentStates[index].scaredTimer
            if timer > 0:
                ghost_mode[0] = 1
                if timer > 10:
                    ghost_mode[1] = 1
                    if timer > 20:
                        ghost_mode[2] = 1
                        if timer > 30:
                            ghost_mode[3] = 1


        # modify ghost far_features to be the closest ghost
        capsule_features = [0, 0, 0, 0]
        closest_actions, _ = closestTargetActions((x_int, y_int), capsules, walls)
        for closest_action in closest_actions:
            capsule_features[Actions._actionToInt[closest_action]] = 1
        # print ('closest capsule (up, down, right, left)', capsule_features)
                                    
        features = np.concatenate((wall_features, 
                                   ghost_features,
                                   ghost_far_features, 
                                   food_features, 
                                   ghost_mode,
                                   capsule_features)).astype(int)

##################################################################################################
        # need to add if it is currently in a intersection
        state.data.is_intersection = True
        wall_cnt = 0
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
                wall_cnt += 1
                if wall_cnt > 1:
                    state.data.is_intersection = False
                    break
        
##################################################################################################

        return features.reshape(24, 1, 1)
















    # def examine_wall(self, walls, x, y):
    #     wall_features = [] 

    #     # Search whether there's a neighboring wall
    #     x_int, y_int = int(x + 0.5), int(y + 0.5)
    #     empty_neighbors = {}
    #     for dir, vec in Actions._directionsAsList:
    #         dx, dy = vec
    #         next_x = x_int + dx
    #         next_y = y_int + dy
    #         if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
    #             wall_features.append(1)
    #         else:
    #             wall_features.append(0)
    #             empty_neighbors[(dx, dy)] = (next_x, next_y)

    #     return wall_features, empty_neighbors


    # def getUnconditionedFeatures(self, state):
    #     # extract the grid of food and wall locations and get the ghost locations
    #     food = state.getFood()
    #     walls = state.getWalls()
    #     ghosts = state.getGhostPositions()
    #     x, y = state.getPacmanPosition()

    #     closest_food = modifiedClosestFood((x, y), food, walls)
    #     food_features = [0, 0, 0, 0]
    #     if closest_food is not None:
    #         food_features[Actions._actionToInt[Actions.vectorToDirection(closest_food)]] = 1

    #     ghost_features = []
    #     wall_features, empty_neighbors = self.examine_wall(walls, x, y)
    #     intersection_features = [0, 0, 0, 0]
    #     ind = 0
    #     for dir, vec in Actions._directionsAsList:
    #         key_vec = (vec[0], vec[1])
    #         if key_vec in empty_neighbors:
    #             neigh_x, neigh_y = empty_neighbors[key_vec]
    #             neigh_walls, _ = self.examine_wall(walls, neigh_x, neigh_y)
    #             if sum(neigh_walls) < 2:
    #                 intersection_features[ind] = 1
    #                 # print ('close to intersection')
    #         ind += 1
        

    #     # Search whether there's a ghost within ghost_radius tiles
    #     ghost_radius = 3
    #     x_int, y_int = int(x + 0.5), int(y + 0.5)

    #     for dir, vec in Actions._directionsAsList:
    #         if self._findGhosts((x_int, y_int), vec, walls, ghosts, ghost_radius=ghost_radius):
    #             ghost_features.append(1)
    #         else:
    #             ghost_features.append(0)
        
    #     far_ghost_features = []
    #     for dir, vec in Actions._directionsAsList:
    #         if self._findGhosts((x_int, y_int), vec, walls, ghosts, ghost_radius=10 + ghost_radius):
    #             far_ghost_features.append(1)
    #         else:
    #             far_ghost_features.append(0)


    #     ghost_mode = 0

    #     # for index in range(1, len(state.data.agentStates)):
    #     #     if state.data.agentStates[index].scaredTimer > 0:
    #     #         ghost_mode = 1
    #     #         break

    #     features = np.concatenate((wall_features, intersection_features, 
    #                                ghost_features, far_ghost_features, 
    #                                food_features, [ghost_mode])).astype(int)

    #     return features.reshape(21, 1, 1)

class MultiagentFeatureExtractor(FeatureExtractor):
    def getUnconditionedFeatures(self, state):
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        x, y = state.getPacmanPosition()

        wall_features = []

        # Search whether there's a neighboring wall
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
                wall_features.append(1)
            else:
                wall_features.append(0)

        ghost_mode = 0

        for index in range(1, len(state.data.agentStates)):
            if state.data.agentStates[index].scaredTimer > 0:
                ghost_mode = 1
                break

        for ghost in range(len(ghosts)):
            pass
            features = np.concatenate((wall_features, ghost_features, food_features, [ghost_mode])).astype(int)

        return None
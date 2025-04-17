import os
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import pybullet_data
from Helper import Helper
from Dualcontrol import Dualcontrol
from pkg_resources import parse_version

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class Dualenv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, urdf_root=pybullet_data.getDataPath(), action_repeat=1,
                 is_enable_self_collision=True, renders=False,
                 is_discrete=False, max_steps=60000):
        self.in_pos = None
        self.move2pos_initial = None
        self.is_discrete = is_discrete
        self._timeStep = 1. / 240
        self._urdfRoot = urdf_root
        self._actionRepeat = action_repeat
        self._isEnableSelfCollision = is_enable_self_collision
        self._observation = []
        self._renders = renders
        self._maxSteps = max_steps
        self._width = 341
        self._height = 256
        self.terminated = 0
        self.grasp = None
        self.p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            #p.resetDebugVisualizerCamera(1.5, 130, -50, [0.52, -0.2, 0.])
            p.resetDebugVisualizerCamera(2, 45, -50, [0.52, -0.2, 0.])
        else:
            p.connect(p.DIRECT)
        self.hp = Helper()
        action_dim = 8  # action dimension
        self._action_bound = 1.0  # action range
        action_high = np.array([self._action_bound] * action_dim, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        # observationDim = 25
        lower_observation = [-1.0] * 50
        upper_observation = [1.0] * 50
        self.observation_space = spaces.Box(low=np.array(lower_observation, dtype=np.float64),
                                            high=np.array(upper_observation, dtype=np.float64),
                                            dtype=np.float64)
        self.viewer = None
        self.seed()
        self.reset()

    def reset(self, **kwargs):
        p.resetSimulation()
        self.terminated = 0
        self.stage = 0
        self.in_pos = -1
        self.gl_error = 0.015
        self.near_error = 0.03
        self.out_of_range = 0
        self._envStepCounter = 0
        self._graspSuccess = 0
        self.object_slip = 0
        self.move2pos_initial = 0
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._timeStep)
        #p.setPhysicsEngineParameter(numSolverIterations=150)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -0.15])
        self.info = self.hp.loadInfo()  # load object info from pos_all.csv
        self.index = self.info[0]  # object id
        self.grasp = self.info[8]  # grasp topology
        self.affordance = self.info[9]  # object affordance
        self.task_id = self.info[10] #task_id
        self.fail_reason = None
        # relative pos and orn between the hand and object
        self.p_rel, self.q_rel = self.hp.relative_pno(self.hp.p_origin, self.hp.q_origin,
                                                      self.info[1], self.info[2])
        # align link pos to center mass pos
        self.h_p_rel, self.h_q_rel = self.hp.relative_pno(self.hp.p_palm_cm, self.hp.q_origin,
                                                          self.hp.p_palm_lk, self.hp.q_origin)
        self.p_obj = self.info[5]  # object pos
        self.q_obj = self.info[6]  # object orientation
        angle = self.hp.rNum(self.info[3], self.info[4])  # random rotate the object
        # rotate the object round z axis by angle degree
        self.q_obj = self.hp.rotate_object(self.q_obj, angle, "z")
        self.object = p.loadURDF(os.path.join(parent_dir, self.info[7]), self.p_obj, self.q_obj,
                                 useFixedBase=0)  # load object
        self.p_new, self.q_new = self.hp.calculate_rigid_trans(self.p_obj, self.q_obj, self.p_rel,
                                                               self.q_rel)  # calculate new hand pos and orn
        self._dual = Dualcontrol(timeStep=self._timeStep, grasp=self.grasp, object_id=self.object)
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        d_s1 = 0.01
        d_s2 = 0.1
        d_near = 0.003      
        obPos, obOrn = p.getBasePositionAndOrientation(self.object)
        self.p_new, self.q_new = self.hp.calculate_rigid_trans(obPos, obOrn, self.p_rel, self.q_rel)

        if self.inPos(self.gl_error):
            self.stage = 2
            self.in_pos = 1
        elif self.in_pos == 1 and self.inGrasp():
            self.stage = 2
            #self.draw_rays_batch()
        else:
            self.stage = 1
            self.in_pos = 0
        #self.stage = 2
        if self.stage == 2:
            j_88 = action[3] * d_s2  # index
            j_92 = action[4] * d_s2  # mid
            j_105 = action[5] * d_s2  # ring
            j_100 = action[6] * d_s2  # pinky
            j_81 = action[7] * d_s2  # thumb
            if self.grasp is None:
                realAction = [0, 0, 0, 0, 0, 0, 0, 0]
            if self.grasp == "platform":
                realAction = [0, 0, 0, 0, 0, 0, 0, 0]
            if self.grasp == "inSiAd2":
                realAction = [0, 0, 0, 0, 0, 0, 0, j_81]
            if self.grasp == "pPdAb2":
                realAction = [0, 0, 0, j_88, 0, 0, 0, j_81]
            if self.grasp == "pPdAb23":
                realAction = [0, 0, 0, j_88, j_92, 0, 0, j_81]
            if self.grasp == "pPdAb25":
                realAction = [0, 0, 0, j_88, j_92, j_105, j_100, j_81]
            if self.grasp == "poPmAb25":
                realAction = [0, 0, 0, j_88, j_92, j_105, j_100, j_81]

        if self.stage == 1:
            dx = action[0] * d_s1
            dy = action[1] * d_s1
            dz = action[2] * d_s1

            if self.s1_x(self.near_error):
                dx = action[0] * d_near
            if self.s1_y(self.near_error):
                dy = action[1] * d_near
            if self.s1_z(self.near_error):
                dz = action[2] * d_near

            if self.s1_x(self.gl_error):
                dx = action[0] * 0.0005
            if self.s1_y(self.gl_error):
                dy = action[1] * 0.0005
            if self.s1_z(self.gl_error):
                dz = action[2] * 0.0005

            realAction = [dx, dy, dz, 0, 0, 0, 0, 0]
        return self.step1(realAction)

    def step1(self, action):
        # move to initial pos and orn first
        if self.move2pos_initial == 0:
            for i in range(300):
                self._dual.applyAction(action, self.p_new, self.q_new, self.terminated, self.stage,
                                       self.move2pos_initial, self.pickup())
                p.stepSimulation()
            self.move2pos_initial = 1

        for i in range(self._actionRepeat):
            self._dual.applyAction(action, self.p_new, self.q_new, self.terminated, self.stage,
                                   self.move2pos_initial, self.pickup())
            p.stepSimulation()

            if self._termination(action):
                break
            self._envStepCounter += 1
            if self._renders:
                time.sleep(self._timeStep)
        reward = self._reward()

        if self.fail_reason == "time out":
            reward = reward - 10000
        if self.fail_reason == "out of range":
            reward = reward - 20000
        if self.object_slip == 1:
            reward = reward + 40000
        if self._graspSuccess:
            reward = reward + 100000
            print(self.task_id, ": Task Succeeded!!")
            self.fail_reason = "success"
            if self.in_friction_cone():
                reward = reward + 10000
            done = True
        elif not self._graspSuccess and self.terminated == 1:
            done = True
        elif not self._graspSuccess and self.terminated == 2:
            done = True
        else:
            done = self._termination(action)
        self._observation = self.getExtendedObservation()
        return np.array(self._observation), reward, done, {}

    def getExtendedObservation(self):
        # distance between the current pos of the hand and the target
        dist = self.hp.distant(p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex)[4], self.p_new)
        #  6 dims each 36 total, normalized
        thumb_tip = ([float(i) / sum(self.observation_relatives(82)[0]) for i in self.observation_relatives(82)[0]]
                     + [float(i) / sum(self.observation_relatives(82)[1]) for i in self.observation_relatives(82)[1]])
        index_tip = ([float(i) / sum(self.observation_relatives(89)[0]) for i in self.observation_relatives(89)[0]]
                     + [float(i) / sum(self.observation_relatives(89)[1]) for i in self.observation_relatives(89)[1]])
        middle_tip = ([float(i) / sum(self.observation_relatives(93)[0]) for i in self.observation_relatives(93)[0]]
                      + [float(i) / sum(self.observation_relatives(93)[1]) for i in self.observation_relatives(93)[1]])
        ring_tip = ([float(i) / sum(self.observation_relatives(57)[0]) for i in self.observation_relatives(57)[0]]
                    + [float(i) / sum(self.observation_relatives(57)[1]) for i in self.observation_relatives(57)[1]])
        pinky_tip = ([float(i) / sum(self.observation_relatives(53)[0]) for i in self.observation_relatives(53)[0]]
                     + [float(i) / sum(self.observation_relatives(53)[1]) for i in self.observation_relatives(53)[1]])
        palm_relatives = ([float(i) / sum(self.observation_relatives(self.hp.dualEndEffectorIndex)[0])
                           for i in self.observation_relatives(self.hp.dualEndEffectorIndex)[0]]
                          + [float(i) / sum(self.observation_relatives(self.hp.dualEndEffectorIndex)[1])
                             for i in self.observation_relatives(self.hp.dualEndEffectorIndex)[1]])

        in_pos_gl = [self.s1_x(self.gl_error), self.s1_y(self.gl_error), self.s1_z(self.gl_error)]
        in_pos_near = [self.s1_x(self.near_error), self.s1_y(self.near_error), self.s1_z(self.near_error)]
        #in_pos = [self.pickup(), self.inGrasp()]
        o_cone = [self.in_friction_cone()]
        force, torque = self.check_equilibrium()
        self._observation = (palm_relatives + thumb_tip + index_tip + middle_tip + ring_tip + pinky_tip + [dist]
                             + o_cone + in_pos_gl + in_pos_near + force.tolist() + torque.tolist())

        return self._observation

    def _termination(self, action):
        if self._envStepCounter > self._maxSteps:
            self._observation = self.getExtendedObservation()
            print(self.task_id, ": stop due to time out")
            self.fail_reason = "time out"
            time.sleep(1)
            return True

        if not self.object_inPos() and not self.inGrasp():
            self._observation = self.getExtendedObservation()
            print(self.task_id, " Terminated: object out of range")
            self.fail_reason = "out of range"
            self.out_of_range = 1
            time.sleep(1)
            return True

        if self.grasp == "platform" and self.stage == 2:
            self.terminated = 2
            for i in range(300):
                self._dual.applyAction(action, self.p_new, self.q_new, self.terminated, self.stage,
                                       self.move2pos_initial, self.pickup())
                p.stepSimulation()
                contact = self.contactInfo(1)
                if sum(contact) > 0:
                    self._graspSuccess = 1
                    self._observation = self.getExtendedObservation()
                    break
            time.sleep(1)

            if self._graspSuccess == 0:
                print(self.task_id, " Terminated: Press Failed")
                self.fail_reason = "press fail"
                self._observation = self.getExtendedObservation()
                time.sleep(1)
                return True

        if self.inGrasp() and self.pickup() and self.stage == 2:
            self.terminated = 1
            for i in range(1000):
                self._dual.applyAction(action, self.p_new, self.q_new, self.terminated, self.stage,
                                       self.move2pos_initial, self.pickup())
                p.stepSimulation()
                objectPosCurrent = p.getBasePositionAndOrientation(self.object)[0]
                #print(objectPosCurrent[2])
                if objectPosCurrent[2] > self.p_obj[2] + 0.08:
                    self._graspSuccess = 1
                    self._observation = self.getExtendedObservation()
                    time.sleep(1)
                    break
            if self._graspSuccess != 1:
                print(self.task_id, " Terminated: Object slipped")
                self.fail_reason = "slipped"
                self.object_slip = 1
                self._observation = self.getExtendedObservation()
                time.sleep(1)
                return True
        return False

    def _reward(self):
        reward_s1 = self.reward_s1()
        reward_s2 = self.reward_s2()
        if self.inPos(self.gl_error):
            self.stage = 2
            self.in_pos = 1
        elif self.in_pos == 1 and self.inGrasp():
            self.stage = 2
            #self.draw_rays_batch()
        else:
            self.stage = 1
            self.in_pos = 0

        if self.stage == 2:  # stage 2
            reward = 80 + reward_s2
        if self.stage == 1:  # stage 2
            reward = reward_s1
        return reward

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        return self.getExtendedObservation()

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step

    ################################################# helper function #####################################

    def reward_s1(self):
        # convert link pos to center mass pos (palm)
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex)[4]
        # distance reward
        dist = self.hp.distant(p_hand, self.p_new)
        reward_move = 1 / (dist + 1)  # max 1

        # direction reward
        # Initial and current direction vectors
        initial_direction = np.array(self.p_new) - np.array(self.hp.endEffectorPos)
        current_direction = np.array(p_hand) - np.array(self.hp.endEffectorPos)
        # Calculate cosine similarity
        reward_direct = self.hp.calculate_direction(initial_direction, current_direction)  # [-0.5, 0.5]

        # collision penalty
        # get contact information
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        contact_num = len(contact_points)
        reward_collision = contact_num if contact_num > 0 else 0
        # overall stage 1 reward
        reward = reward_move + 50 * reward_direct - reward_collision
        if self.s1_x(self.near_error):
            reward += 2
        if self.s1_y(self.near_error):
            reward += 2
        if self.s1_z(self.near_error):
            reward += 2
        if self.inPos(self.gl_error):
            reward = 80

        return reward

    def reward_s2(self):
        # get contact information
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        force, torque = self.check_equilibrium()

        contact_num = len(contact_points)*2 + 1 / (sum(force) + 1) + 1 / (sum(torque) + 1)
        #print("reward2: ", len(contact_points), contact_num)
        return contact_num

    def inPos(self, error):  # the grasp location is a range
        return self.s1_x(error) and self.s1_y(error) and self.s1_z(error)

    def s1_x(self, error):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex)[4]
        return (p_hand[0] <= self.p_new[0] + error) and (p_hand[0] >= self.p_new[0] - error)

    def s1_y(self, error):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex)[4]
        return (p_hand[1] <= self.p_new[1] + error) and (p_hand[1] >= self.p_new[1] - error)

    def s1_z(self, error):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex)[4]
        return (p_hand[2] <= self.p_new[2] + error) and (p_hand[2] >= self.p_new[2] - error)

    def object_inPos(self):  # the object pos range
        obPos, _ = p.getBasePositionAndOrientation(self.object)
        x = (obPos[0] >= 0.72) and (obPos[0] <= 0.84)
        y = (obPos[1] >= -0.51) and (obPos[1] <= -0.40)
        return x and y

    def sus(self):
        return self._graspSuccess

    def pickup(self):
        contact = self.contactInfo(300)
        #print(contact)
        if self.grasp is None:
            return False
        if self.grasp == "inSiAd2":
            return sum(contact) >= 2
        if self.grasp == "pPdAb2":
            return contact[1] == 1 and contact[2] == 1
        if self.grasp == "pPdAb23":
            return contact[1] == 1 and contact[2] == 1 and sum(contact) >= 3
        if self.grasp == "pPdAb25":
            return sum(contact) >= 4
        if self.grasp == "poPmAb25":
            return sum(contact) >= 4

    def contactInfo(self, threshold=500):
        # boolean value 1 for read to pick up, 0 otherwise
        # thumb and index finger contact_points object
        limitForce = threshold
        contactParts = [0, 0, 0, 0, 0, 0]  # palm, thumb, index, middle, ring, pink
        palmLinks = self.hp.palmLinks
        thumbLinks = self.hp.thumbLinks
        indexLinks = self.hp.indexLinks
        middleLinks = self.hp.middleLinks
        ringLinks = self.hp.ringLinks
        pinkyLinks = self.hp.pinkyLinks
        # get contact information
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        contact_num = len(contact_points)

        # find contact_points point
        # fill force and dist
        if contact_num > 0:
            for i in range(contact_num):
                if contact_points[i][3] in palmLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[0] = 1

                if contact_points[i][3] in thumbLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[1] = 1
                #print("thumb", contact_points[i][9] )

                if contact_points[i][3] in indexLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[2] = 1
                #print("index", contact_points[i][9] )
                if contact_points[i][3] in middleLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[3] = 1

                if contact_points[i][3] in ringLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[4] = 1

                if contact_points[i][3] in pinkyLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[5] = 1
        return contactParts

    def observation_relatives(self, linkId):
        # convert link pos to center mass pos (palm)
        p_link = p.getLinkState(self._dual.dualUid, linkId)[4]
        q_link = p.getLinkState(self._dual.dualUid, linkId)[5]
        # get relative position between the hand and the target
        p_r, q_r = self.hp.relative_pno(self.p_new, self.q_new, p_link, q_link)
        q_r = p.getEulerFromQuaternion(q_r)
        p_rel = [p_r[0], p_r[1], p_r[2]]
        q_rel = [q_r[0], q_r[1], q_r[2]]
        return [p_rel, q_rel]

    def draw_rays_batch(self):
        # take a list of positions and emit collision ray relatively
        # draw rays from ray_from[i] to ray_to[i]
        # ray_to[i] is the collision point
        line_width = 2
        life_time = 1 / 120
        miss_color = [1, 0, 0]  # green
        hit_color = [0, 1, 0]  # blue
        ray_from, ray_to = self.setup_rays_positions_12()
        ray_readings = p.rayTestBatch(ray_from, ray_to)
        for i in range(len(ray_readings)):
            if ray_readings[i][0] != self.object:  # no collision
                p.addUserDebugLine(ray_from[i], list(ray_readings[i][3]), miss_color,
                                   lifeTime=life_time, lineWidth=line_width)
            else:  # collision
                p.addUserDebugLine(ray_from[i], list(ray_readings[i][3]), hit_color,
                                   lifeTime=life_time, lineWidth=line_width)

    def setup_rays_positions_12(self):  # 12 rays
        # pair 2 list without duplication
        ray_from, ray_to = [], []
        # 36 cross hit
        thumb_joints = [80] * 4 + [81] * 4 + [82] * 4
        finger_joints = [87, 88, 89, 91, 92, 93, 99, 100, 101, 104, 105, 106]
        for i in range(len(thumb_joints)):
            ray_from.append(p.getLinkState(self._dual.dualUid, thumb_joints[i])[4])
            ray_to.append(p.getLinkState(self._dual.dualUid, finger_joints[i])[4])
        return ray_from, ray_to

    def inGrasp(self):
        # check if the object is in grasp
        ray_from, ray_to = self.setup_rays_positions_12()
        readings = p.rayTestBatch(ray_from, ray_to)
        object_contact = 0
        if len(readings) > 0:
            for i in range(len(readings)):
                if readings[i][0] == self.object:
                    object_contact += 1
            if object_contact > 0:
                return True
        return False

    def check_equilibrium(self):

        force_total = np.array([0, 0, 0], dtype=np.float64)
        torque_total = np.array([0, 0, 0], dtype=np.float64)
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        if len(contact_points) > 0:
            for contact in contact_points:
                # position vecc for torque
                object_com_position, _ = p.getBasePositionAndOrientation(self.object)
                object_com_position = np.array(object_com_position)
                contact_pos = np.array(contact[6], dtype=np.float64)
                c_com = contact_pos - object_com_position
                # forces and directions
                norm_force = np.array(contact[9], dtype=np.float64)  # 1
                norm_vec = np.array(contact[7], dtype=np.float64)  # 3
                lateral1 = np.array(contact[10])  # 1
                lateral1_vec = np.array(contact[11], dtype=np.float64)  # 3
                lateral2 = np.array(contact[12], dtype=np.float64)  # 1
                lateral2_vec = np.array(contact[13], dtype=np.float64)  # 3
                # forces
                norm = norm_force * norm_vec
                lateral_1 = lateral1 * lateral1_vec
                lateral_2 = lateral2 * lateral2_vec
                # force balance
                force_total += norm
                force_total += lateral_1
                force_total += lateral_2
                # torque balance
                torque_total += np.cross(c_com, norm)
                torque_total += np.cross(c_com, lateral_1)
                torque_total += np.cross(c_com, lateral_2)
            return force_total, torque_total
        else:
            return (np.array([3.3, 3.3, 3.3], dtype=np.float64),
                    np.array([3.3, 3.3, 3.3], dtype=np.float64))

    def in_friction_cone(self, friction_coefficient=0.6):
        """
        Check if all contact forces between two bodies are within the friction cone.
        Parameters:
        friction_coefficient (float): Coefficient of friction at the contact points.
        Returns:
        bool: True if all contact forces are within the friction cone, False otherwise.
        """
        # Get all contact points between the two bodies
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        for contact in contact_points:
            # Extract normal force (force along the contact normal)
            normal_force = contact[9]  # index 9 is normal force in contact information
            # Extract lateral forces
            lateral_friction1 = contact[10]  # Lateral friction force along the first direction
            lateral_friction2 = contact[12]  # Lateral friction force along the second direction

            # Compute the tangential (lateral) force magnitude
            tangential_force = np.sqrt(lateral_friction1 ** 2 + lateral_friction2 ** 2)

            # Check if the tangential force is within the friction cone
            if tangential_force > friction_coefficient * normal_force:
                # If any tangential force exceeds the friction cone limit, return False
                return False
        # If all contact points satisfy the friction cone condition, return True
        return True

    ##################################### Not Used ###########################################################

    def pickup1(self):
        contact = self.contactInfo(30)
        #print(contact)
        if self.grasp is None:
            return False
        if self.grasp == "inSiAd2":
            return contact[1] == 1 and contact[2] == 1
        if self.grasp == "pPdAb2":
            return contact[1] == 1 and contact[2] == 1
        if self.grasp == "pPdAb23":
            return contact[1] == 1 and contact[2] == 1 and contact[3] == 1
        if self.grasp == "pPdAb25":
            return (contact[1] == 1 and contact[2] == 1 and contact[3] == 1
                    and contact[4] == 1 )
        if self.grasp == "poPmAb25":
            # return (contact[0] == 1 and contact[1] == 1 and contact[2] == 1
            #         and contact[3] == 1 and contact[4] == 1 and contact[5] == 1)
            return (contact[0] == 1 and contact[1] == 1 and contact[2] == 1
                    and contact[3] == 1 and contact[4] == 1 and contact[5] == 1)

    def get_joint_pos(self, jointId):
        return p.getJointState(self._dual.dualUid, jointId)[0]

    def draw_rays(self, ray_from, ray_to):
        # same as set_rays_batch but take single position
        miss_color = [0, 0, 1]  # green
        hit_color = [1, 0, 0]  # blue
        ray_readings = p.rayTest(ray_from, ray_to)
        if ray_readings[0][2] > 0.9:  # no collision
            p.addUserDebugLine(ray_from, ray_readings[0][3], miss_color, lifeTime=1 / 240, lineWidth=3,
                               replaceItemUniqueId=p.addUserDebugLine(ray_from, ray_readings[0][3],
                                                                      hit_color, lineWidth=30))
        else:  # collision
            p.addUserDebugLine(ray_from, ray_readings[0][3], hit_color, lifeTime=1 / 240, lineWidth=3,
                               replaceItemUniqueId=p.addUserDebugLine(ray_from, ray_readings[0][3],
                                                                      miss_color, lineWidth=3))

    def get_contactInfo(self, bodyA, bodyB):
        # get all contact info between 2 bodies
        # detailed info about this function can be
        # found in pybullet guide (function getContactPoints)
        return p.getContactPoints(bodyA, bodyB)

    def get_link_contactInfo(self, bodyA, bodyB, linkA, LinkB):
        # get contact info between 2 links
        # detailed info about this function can be
        # found in pybullet guide (function getContactPoints)
        return p.getContactPoints(bodyA, bodyB, linkA, LinkB)

    def closest_point(self, bodyA, bodyB, distance, linkA, LinkB):
        # return the closest point info between 2 links of 2 bodies
        # within a certain distant
        # detailed info about this function can be
        # found in pybullet guide (function getClosestPoints)
        return p.getClosestPoints(bodyA, bodyB, distance, linkA, LinkB)

    def draw_rays_batch1(self):
        # take a list of positions and emit collision ray relatively
        # draw rays from ray_from[i] to ray_to[i]
        # ray_to[i] is the collision point
        miss_color = [0, 0, 1]  # green
        hit_color = [1, 0, 0]  # blue
        ray_from, ray_to = self.setup_rays_positions_allpairs()
        ray_readings = p.rayTestBatch(ray_from, ray_to)
        for i in range(len(ray_readings)):
            if ray_readings[i][0] != self.object:  # no collision
                p.addUserDebugLine(ray_from[i], list(ray_readings[i][3]), miss_color, lifeTime=1 / 120,
                                   lineWidth=3, replaceItemUniqueId=p.addUserDebugLine(ray_from[i],
                                                                                       list(ray_readings[i][3]),
                                                                                       hit_color, lineWidth=3))
            else:  # collision
                p.addUserDebugLine(ray_from[i], list(ray_readings[i][3]), hit_color, lifeTime=1 / 120,
                                   lineWidth=3, replaceItemUniqueId=p.addUserDebugLine(ray_from[i],
                                                                                       list(ray_readings[i][3]),
                                                                                       miss_color, lineWidth=3))

    def setup_rays_positions_allpairs(self):
        # pair 2 list without duplication
        list1 = self.hp.hand_joints
        list2 = self.hp.hand_joints
        paired = set()
        ray_from, ray_to = [], []
        for i, e1 in enumerate(list1):
            for j, e2 in enumerate(list2):
                if e1 != e2 and (i, j) not in paired and (j, i) not in paired:
                    ray_from.append(p.getLinkState(self._dual.dualUid, e1)[4])
                    ray_to.append(p.getLinkState(self._dual.dualUid, e2)[4])
                    paired.add((i, j))

        return ray_from, ray_to

    def setup_rays_positions_36(self):  # 36 rays
        # pair 2 list without duplication
        ray_from, ray_to = [], []
        # 36 cross hit
        thumb_joints = [80] * 12 + [81] * 12 + [82] * 12
        finger_joints = [87, 88, 89, 91, 92, 93, 99, 100, 101, 104, 105, 106] * 3
        for i in range(len(thumb_joints)):
            ray_from.append(p.getLinkState(self._dual.dualUid, thumb_joints[i])[4])
            ray_to.append(p.getLinkState(self._dual.dualUid, finger_joints[i])[4])

        return ray_from, ray_to

    def setup_rays_positions_4(self):
        # pair 2 list without duplication
        ray_from, ray_to = [], []
        # 36 cross hit
        thumb_joints = [82] * 4
        finger_joints = [89, 93, 101, 106]
        for i in range(len(thumb_joints)):
            ray_from.append(p.getLinkState(self._dual.dualUid, thumb_joints[i])[4])
            ray_to.append(p.getLinkState(self._dual.dualUid, finger_joints[i])[4])
        # print(len(ray_from)) 4 rays
        return ray_from, ray_to


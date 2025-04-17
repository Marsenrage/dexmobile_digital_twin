import os
import time

import pybullet as p
from Helper import Helper

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class Dualcontrol:

    def __init__(self, timeStep=0.01, grasp='relax', object_id=-1):

        self.move2pos = 0
        self.table = None
        self.wait_grasp = None
        self.endEffectorPos = None
        self.timeStep = timeStep
        self.object_id = object_id
        ##################### Joint related info  ############################
        self.hp = Helper()
        self.topology = grasp
        self.grasp = self.hp.grasp_pose[grasp]
        self.arm_pos = self.hp.arm_pos
        self.dualEndEffectorIndex = self.hp.dualEndEffectorIndex
        self.max_force = self.hp.max_force
        self.joint_damp = self.hp.joint_damp
        self.r_joint_id = self.hp.r_joint_id
        self.lower_limit = self.hp.lower_limit
        self.upper_limit = self.hp.upper_limit
        self.joint_range = self.hp.joint_range
        self.max_velocity = self.hp.max_velocity
        self.hand_maxForce = self.hp.hand_maxForce
        #self.rest_pos = self.arm_pos + self.hp.grasp_pose["pPdAb23"]
        self.rest_pos = self.arm_pos + self.hp.grasp_pose["relax"]
        #self.rest_pos = self.arm_pos + [0]*19
        self.dualUid = -100
        self.finger_initial = None
        self.final_index = None
        self.final_mid = None
        self.final_ring = None
        self.final_pinky = None
        self.final_thumb = None
        self.pose1 = [0.46447134017944336, -0.5008310079574585, 0.8000117540359497]

        self.reset()

    def reset(self):
        self.wait_grasp = 0
        self.final_index = -1
        self.final_mid = -1
        self.final_ring = -1
        self.final_pinky = -1
        self.final_thumb = -1
        self.endEffectorPos = self.hp.endEffectorPos
        self.finger_initial = [self.grasp[6], self.grasp[9], self.grasp[17], self.grasp[13], self.grasp[2]]
        # self.dualUid = p.loadURDF(os.path.join(parent_dir, "robots/dual_2hand.urdf"), useFixedBase=1,
        #                           flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self.dualUid = p.loadURDF(os.path.join(parent_dir, "robots/dual_2hand.urdf"), useFixedBase=1)
        p.resetBasePositionAndOrientation(self.dualUid, self.hp.p_origin, self.hp.q_origin)
        self.table = p.loadURDF(os.path.join(parent_dir, "robots/table/table.urdf"), self.hp.p_table, self.hp.q_origin,
                                useFixedBase=1)
        for i in range(len(self.r_joint_id)):
            p.resetJointState(self.dualUid, self.r_joint_id[i], self.rest_pos[i])
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.rest_pos[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=self.max_velocity[i], positionGain=0.03, velocityGain=1)

    def applyAction(self, actions, p_new, q_new, terminated, stage, move2pos_initial=None, pickup=0):
        ############################################# actions ###################################################
        positionX = actions[0]
        positionY = actions[1]
        positionZ = actions[2]
        thumb_close = abs(actions[7])
        finger_close_index = abs(actions[3])
        finger_close_mid = abs(actions[4])
        # thumb_close = actions[7]
        # finger_close_index = actions[3]
        # finger_close_mid = actions[4]
        # j_88 index, j_92 mid, j_105 ring, j_100 pinky, j_81 thumb
        if self.topology is None:
            j_88, j_92, j_105, j_100, j_81 = 0, 0, 0, 0, 0
        if self.topology == "platform":
            j_88, j_92, j_105, j_100, j_81 = 0, 0, 0, 0, 0
        if self.topology == "inSiAd2":
            j_88, j_92, j_105, j_100, j_81 = 0, 0, 0, 0, thumb_close
        if self.topology == "pPdAb2":
            j_88, j_92, j_105, j_100, j_81 = finger_close_index, 0, 0, 0, thumb_close
        if self.topology == "pPdAb23":
            j_88, j_92, j_105, j_100, j_81 = finger_close_index, finger_close_mid, 0, 0, thumb_close
        if self.topology == "pPdAb25" or self.topology == "poPmAb25":
            j_88, j_92, j_105, j_100, j_81 = (finger_close_mid, finger_close_mid,
                                              finger_close_mid, finger_close_mid, thumb_close)
        pose2 = p.getLinkState(self.dualUid, 90)[4]
        #print(pose2)
        ############################################# control arm ###################################################
        if move2pos_initial == 0:  # move to initial pos

            self.move_to(self.dualEndEffectorIndex, self.endEffectorPos, q_new)
            #pose1 = p.getLinkState(self.dualUid, self.dualEndEffectorIndex)[4]
            #self.prevPose = pose1

        elif terminated == 1:  # pick up
            current_pos = p.getJointState(self.dualUid, 65)[0]
            self.move_arm_joint(65, current_pos + 0.3)
            if pickup == 1:
                self.final_index = j_88
                self.final_mid = j_92
                self.final_ring = j_105
                self.final_pinky = j_100
                self.final_thumb = j_81

            if self.topology == "inSiAd2":
                self.finger_model_thumb(self.final_thumb)
            if self.topology == "pPdAb2":
                self.finger_model_index(self.final_index)
                self.finger_model_thumb(self.final_thumb)

            if self.topology == "pPdAb23":
                self.finger_model_index(self.final_index)
                self.finger_model_mid(self.final_mid)
                self.finger_model_thumb(self.final_thumb)

            if self.topology == "pPdAb25" or self.topology == "poPmAb25":
                self.finger_model_index(self.final_index)
                self.finger_model_mid(self.final_mid)
                self.finger_model_ring(self.final_ring)
                self.finger_model_pinky(self.final_pinky)
                self.finger_model_thumb(self.final_thumb)

        elif terminated == 2:  # press
            posT = [self.endEffectorPos[0], self.endEffectorPos[1], self.endEffectorPos[2] - 0.5]
            self.move_down(self.dualEndEffectorIndex, posT, q_new)
        else:
            if stage == 1:
                self.move_to(self.dualEndEffectorIndex, self.endEffectorPos, q_new)
                #self.move_to(self.dualEndEffectorIndex, p_new, q_new)
                self.endEffectorPos[0] = self.endEffectorPos[0] + positionX
                # if self.endEffectorPos[0] > 0.8:
                #     self.endEffectorPos[0] = 0.8
                if self.endEffectorPos[0] > p_new[0]+0.01:
                    self.endEffectorPos[0] = p_new[0]+0.01
                if self.endEffectorPos[0] < 0.65:
                    self.endEffectorPos[0] = 0.65

                self.endEffectorPos[1] = self.endEffectorPos[1] + positionY
                if self.endEffectorPos[1] <= -0.7:
                    self.endEffectorPos[1] = -0.7
                # if self.endEffectorPos[1] >= -0.36:
                #     self.endEffectorPos[1] = -0.36
                if self.endEffectorPos[1] >= p_new[1] + 0.01:
                    self.endEffectorPos[1] = p_new[1] + 0.01

                self.endEffectorPos[2] = self.endEffectorPos[2] + positionZ
                if self.endEffectorPos[2] >= 0.6:
                    self.endEffectorPos[2] = 0.6
                # if self.endEffectorPos[2] <= 0.14:
                #     self.endEffectorPos[2] = 0.14
                if self.endEffectorPos[2] <= p_new[2]-0.01:
                    self.endEffectorPos[2] = p_new[2]-0.01
                ############################################# control hand ###############################################
                # joint id included in dual_joint_info.xlsx
                self.move_finger_joint(78, self.grasp[6])
                self.move_finger_joint(80, self.hp.relax[1])
                self.move_finger_joint(81, self.hp.relax[2])
                self.move_finger_joint(82, self.hp.relax[3])
                self.move_finger_joint(86, self.hp.relax[4])
                self.move_finger_joint(87, self.hp.relax[5])
                self.move_finger_joint(88, self.hp.relax[6])
                self.move_finger_joint(89, self.hp.relax[7])
                self.move_finger_joint(91, self.hp.relax[8])
                self.move_finger_joint(92, self.hp.relax[9])
                self.move_finger_joint(93, self.hp.relax[10])
                self.move_finger_joint(98, self.hp.relax[11])
                self.move_finger_joint(99, self.hp.relax[12])
                self.move_finger_joint(100, self.hp.relax[13])
                self.move_finger_joint(101, self.hp.relax[14])
                self.move_finger_joint(103, self.hp.relax[15])
                self.move_finger_joint(104, self.hp.relax[16])
                self.move_finger_joint(105, self.hp.relax[17])
                self.move_finger_joint(106, self.hp.relax[18])

            if stage == 2:
                if self.wait_grasp == 0:
                    time.sleep(1)
                    self.wait_grasp = 1

                # index
                self.move_finger_joint(86, 0)
                self.finger_model_index(j_88)
                # mid
                self.finger_model_mid(j_92)
                # ring
                self.move_finger_joint(103, 0)
                self.finger_model_ring(j_105)
                # pinky
                self.move_finger_joint(98, 0)
                self.finger_model_pinky(j_100)
                # thumb
                self.finger_model_thumb(j_81)
                self.move_finger_joint(78, self.grasp[6])
        #p.addUserDebugLine(self.pose1, pose2, [0, 0, 1], 5, 0)
        self.pose1 = pose2
    ############################################# helper functions ################################################

    def move_finger_joint(self, joint_id, target_position):

        p.setJointMotorControl2(self.dualUid, jointIndex=joint_id, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position, targetVelocity=0, force=self.hand_maxForce,
                                maxVelocity=1, positionGain=0.03, velocityGain=1)

    def move_arm_joint(self, joint_id, target_position):

        p.setJointMotorControl2(self.dualUid, jointIndex=joint_id, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position, targetVelocity=0, force=self.hand_maxForce,
                                maxVelocity=0.3, positionGain=0.03, velocityGain=1)

    def move_to(self, end_effector_id, target_position, target_orientation):

        jointPoses = p.calculateInverseKinematics(self.dualUid, end_effector_id, target_position, target_orientation,
                                                  lowerLimits=self.lower_limit, upperLimits=self.upper_limit,
                                                  jointRanges=self.joint_range, restPoses=self.rest_pos)

        for i in range(len(self.r_joint_id)):
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=1000, positionGain=0.3, velocityGain=0.3)

    def move_up(self, end_effector_id, target_position, target_orientation):
        jointPoses = p.calculateInverseKinematics(self.dualUid, end_effector_id, target_position, target_orientation,
                                                  lowerLimits=self.lower_limit, upperLimits=self.upper_limit,
                                                  jointRanges=self.joint_range, restPoses=self.rest_pos)

        for i in range(len(self.r_joint_id)):
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=0.6, positionGain=0.3, velocityGain=0.3)

    def move_down(self, end_effector_id, target_position, target_orientation):
        jointPoses = p.calculateInverseKinematics(self.dualUid, end_effector_id, target_position, target_orientation,
                                                  lowerLimits=self.lower_limit, upperLimits=self.upper_limit,
                                                  jointRanges=self.joint_range, restPoses=self.rest_pos)

        for i in range(len(self.r_joint_id)):
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=0.1, positionGain=0.3, velocityGain=0.3)

    # θ_TDIP ≈ 0.5⋅θ_TMCP
    # Index: θDIP = 0.77⋅θPIP
    # Middle: θDIP = 0.75⋅θPIP
    # Ring: θDIP = 0.75⋅θPIP
    # Little: θDIP = 0.57⋅θPIP
    # θMCP = (0.53-0.71)⋅θPIP    (θPIP = (1.4 − 1.9)⋅θMCP)
    # thumb flexion and opposition move freely
    # < initial angle = initial angle
    # if MCP contact point: pip, dip keep going (flexing)
    # if pip contact point: no effect
    # if dip contact point: all keep moving
    # if reach force threshold: stop

    def thumb_model(self, pip):
        # pip is angle of PIP
        # thumb joints in [PIP, DIP] format
        if pip <= 0.98506:  # upper limit of pip thumb
            dip = self.hp.thumb_alpha_dip * pip
        else:
            dip = pip
        self.move_finger_joint(self.hp.thumb_joint[1], pip)
        self.move_finger_joint(self.hp.thumb_joint[2], dip)

    def check_contact_points(self, joint_id, threshold=200):
        contacts = p.getContactPoints(self.object_id, self.dualUid, -1, joint_id)
        return any(contact[9] >= threshold for contact in contacts)

    def set_self_collision(self):
        for thumb_link in self.hp.thumb_joint:
            for finger_link in self.hp.finger_joints:
                if thumb_link != finger_link:
                    p.setCollisionFilterPair(self.dualUid, self.dualUid, thumb_link, finger_link,
                                             enableCollision=1)

    def check_finger_collision(self, threshold=400):
        for thumb_link in self.hp.thumb_joint:
            for finger_link in self.hp.finger_joints:
                if thumb_link != finger_link:
                    contacts = p.getContactPoints(self.dualUid, self.dualUid, thumb_link, finger_link)
                    if any(contact[9] >= threshold for contact in contacts):
                        return True
        return False

    def finger_model_index(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [87, 88, 89]
        # joint range [0.79849, 1.334, 1.394]
        # θDIP = 0.77⋅θPIP; max_dip = 1.394/0.77 = 1.81039
        # θMCP = 0.67⋅θPIP; max_mcp = 0.79849/0.67 = 1.192
        # pip is angle of PIP
        max_mcp = self.hp.index_joint_max[0] / self.hp.index_alpha_mcp
        max_dip = self.hp.index_joint_max[2] / self.hp.index_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial[0] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial[0] = max_pip
        if self.finger_initial[0] < 0:
            self.finger_initial[0] = 0
        self.finger_initial[0] += delta_pip
        mcp = min(self.hp.index_joint_max[0], self.hp.index_alpha_mcp * self.finger_initial[0])
        dip = min(self.hp.index_joint_max[2], self.hp.index_alpha_dip * self.finger_initial[0])
        pip = self.finger_initial[0]
        mcp_contact = self.check_contact_points(self.hp.index_joint[0])
        pip_contact = self.check_contact_points(self.hp.index_joint[1])
        dip_contact = self.check_contact_points(self.hp.index_joint[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.index_joint[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.index_joint[1], pip)
                self.move_finger_joint(self.hp.index_joint[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.index_joint[0], mcp)
                self.move_finger_joint(self.hp.index_joint[1], pip)
                self.move_finger_joint(self.hp.index_joint[2], dip)

    def finger_model_mid(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [91, 92, 93]
        # joint range [0.79849, 1.334, 1.334]
        # θDIP = 0.75⋅θPIP; max_dip = 1.334/0.75 = 1.77867
        # θMCP = 0.67⋅θPIP; max_mcp = 0.79849/0.67 = 1.19178
        # pip is angle of PIP
        max_mcp = self.hp.mid_joint_max[0] / self.hp.mid_alpha_mcp
        max_dip = self.hp.mid_joint_max[2] / self.hp.mid_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial[1] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial[1] = max_pip
        if self.finger_initial[1] < 0:
            self.finger_initial[1] = 0
        self.finger_initial[1] += delta_pip
        mcp = min(self.hp.mid_joint_max[0], self.hp.mid_alpha_mcp * self.finger_initial[1])
        dip = min(self.hp.mid_joint_max[2], self.hp.mid_alpha_dip * self.finger_initial[1])
        pip = self.finger_initial[1]
        mcp_contact = self.check_contact_points(self.hp.mid_joint[0])
        pip_contact = self.check_contact_points(self.hp.mid_joint[1])
        dip_contact = self.check_contact_points(self.hp.mid_joint[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.mid_joint[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.mid_joint[1], pip)
                self.move_finger_joint(self.hp.mid_joint[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.mid_joint[0], mcp)
                self.move_finger_joint(self.hp.mid_joint[1], pip)
                self.move_finger_joint(self.hp.mid_joint[2], dip)

    def finger_model_ring(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [104, 105, 106]
        # joint range [0.98175, 1.334, 1.395]
        # θDIP = 0.75⋅θPIP; max_dip = 1.395/0.57 = 2.44737
        # θMCP = 0.67⋅θPIP; max_mcp = 0.98175/0.67 = 1.4653
        # pip is angle of PIP
        max_mcp = self.hp.ring_joint_max[0] / self.hp.ring_alpha_mcp
        max_dip = self.hp.ring_joint_max[2] / self.hp.ring_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial[2] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial[2] = max_pip
        if self.finger_initial[2] < 0:
            self.finger_initial[2] = 0
        self.finger_initial[2] += delta_pip
        mcp = min(self.hp.ring_joint_max[0], self.hp.ring_alpha_mcp * self.finger_initial[2])
        dip = min(self.hp.ring_joint_max[2], self.hp.ring_alpha_dip * self.finger_initial[2])
        pip = self.finger_initial[2]
        mcp_contact = self.check_contact_points(self.hp.ring_joint[0])
        pip_contact = self.check_contact_points(self.hp.ring_joint[1])
        dip_contact = self.check_contact_points(self.hp.ring_joint[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.ring_joint[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.ring_joint[1], pip)
                self.move_finger_joint(self.hp.ring_joint[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.ring_joint[0], mcp)
                self.move_finger_joint(self.hp.ring_joint[1], pip)
                self.move_finger_joint(self.hp.ring_joint[2], dip)

    def finger_model_pinky(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [99, 100, 101]
        # joint range [0.98175, 1.334, 1.3971]
        # θDIP = 0.57⋅θPIP; max_dip = 1.395/0.57 = 2.44737
        # θMCP = 0.67⋅θPIP; max_mcp = 0.98175/0.67 = 1.4653
        # pip is angle of PIP
        max_mcp = self.hp.pinky_joint_max[0] / self.hp.pinky_alpha_mcp
        max_dip = self.hp.pinky_joint_max[2] / self.hp.pinky_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial[3] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial[3] = max_pip
        if self.finger_initial[3] < 0:
            self.finger_initial[3] = 0
        self.finger_initial[3] += delta_pip
        mcp = min(self.hp.pinky_joint_max[0], self.hp.pinky_alpha_mcp * self.finger_initial[3])
        dip = min(self.hp.pinky_joint_max[2], self.hp.pinky_alpha_dip * self.finger_initial[3])
        pip = self.finger_initial[3]
        mcp_contact = self.check_contact_points(self.hp.pinky_joint[0])
        pip_contact = self.check_contact_points(self.hp.pinky_joint[1])
        dip_contact = self.check_contact_points(self.hp.pinky_joint[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.pinky_joint[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.pinky_joint[1], pip)
                self.move_finger_joint(self.hp.pinky_joint[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.pinky_joint[0], mcp)
                self.move_finger_joint(self.hp.pinky_joint[1], pip)
                self.move_finger_joint(self.hp.pinky_joint[2], dip)

    def finger_model_thumb(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [80, 81, 82]
        # joint range [0.9704, 0.98506, 1.406]
        # pip is angle of PIP
        max_mcp = self.hp.thumb_joint_max[0] / self.hp.thumb_alpha_mcp
        max_dip = self.hp.thumb_joint_max[2] / self.hp.thumb_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial[4] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial[4] = max_pip
        if self.finger_initial[4] < 0:
            self.finger_initial[4] = 0
        self.finger_initial[4] += delta_pip
        mcp = min(self.hp.thumb_joint_max[0], self.hp.thumb_alpha_mcp * self.finger_initial[4])
        dip = min(self.hp.thumb_joint_max[2], self.hp.thumb_alpha_dip * self.finger_initial[4])
        pip = self.finger_initial[4]
        mcp_contact = self.check_contact_points(self.hp.thumb_joint[0])
        pip_contact = self.check_contact_points(self.hp.thumb_joint[1])
        dip_contact = self.check_contact_points(self.hp.thumb_joint[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.thumb_joint[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.thumb_joint[1], pip)
                self.move_finger_joint(self.hp.thumb_joint[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.thumb_joint[0], mcp)
                self.move_finger_joint(self.hp.thumb_joint[1], pip)
                self.move_finger_joint(self.hp.thumb_joint[2], dip)

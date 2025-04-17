#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import inspect
import os
import pybullet as p
from Dualenv import Dualenv
from Helper import Helper

current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent = os.path.dirname(os.path.dirname(current))
os.sys.path.insert(0, parent)


def main():
    env = Dualenv(renders=True, is_discrete=False)
    hp = Helper()
    motorsIds = []

    # dv = math.pi
    dv = 3.5
    # Arm
    motorsIds.append(env.p.addUserDebugParameter("posX", -dv, dv, 0))  #
    motorsIds.append(env.p.addUserDebugParameter("posY", -dv, dv, 0))
    motorsIds.append(env.p.addUserDebugParameter("posZ", -dv, dv, 0))
    motorsIds.append(env.p.addUserDebugParameter("j_88", -0.1, 0.1, 0))  # index
    motorsIds.append(env.p.addUserDebugParameter("j_92", -0.1, 0.1, 0))  # mid
    motorsIds.append(env.p.addUserDebugParameter("j_100", -0.1, 0.1, 0))  # ring
    motorsIds.append(env.p.addUserDebugParameter("j_105", -0.1, 0.1, 0))  # pinky
    motorsIds.append(env.p.addUserDebugParameter("j_81", -0.1, 0.1, 0))  # thumb

    done = False
    while not done:
        action = []
        for motorId in motorsIds:
            action.append(env.p.readUserDebugParameter(motorId))

        state, reward, done, info = env.step(action)
        # obs = env.getExtendedObservation()
        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
            env.close()
            break


if __name__ == "__main__":
    main()

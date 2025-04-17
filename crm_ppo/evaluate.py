
from Dualenv import Dualenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Monitor import Monitor

################################################# Define Variables ########################################################################
orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "inSiAd2"
log_dir = "trained_model"
vName = "Current99"  #Current9 5
modelName = log_dir + "/" + vName
envName = log_dir + "/" + vName + ".pkl"
################################################# Testing and Evaluation #################################################################

env = Dualenv(renders=True, is_discrete=False, max_steps=1024)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(envName, env)
env.training = False  # not continue training the model while testing
env.norm_reward = False  # reward normalization is not needed at test time
# load model 
model = PPO.load(modelName, env=env)

test = 1000
for i in range(test):
	obs = env.reset()
	done = False
	rewards = float('-inf')
	while (not done):
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)

# sus = model.get_env().get_attr("successGrasp")
# print("SUCCESS RATE IS: ", str((sus[0]/test)*100) + "%" )
print("Evaluation is Done")
env.close()

############### write to txt #######################################

# fileName = "log/" + str((sus[0]/test)*100) + ".txt"
#
# with open(fileName, 'w') as f:
# 	f.write(str((sus[0]/test)*100))














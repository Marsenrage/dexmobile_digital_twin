import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np
import numpy as np
import pandas as pd

from stable_baselines3.common import base_class, logger  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

class successRateCallBack(BaseCallback):

	def __init__(self, successRates, verbose, check_freq, path, n_eval_episodes):
		super(successRateCallBack, self).__init__(verbose=verbose)
		self.successRates = successRates
		self.check_freq = check_freq
		self.path = path 
		self.eval_episodes = n_eval_episodes
		self.save_path = path + "/best"

	def _on_step(self) -> bool:
		current = 0
		if self.n_calls % self.check_freq == 0:
			success = self.numSuccess(self.path, self.eval_episodes)
			rate = success
			if(rate> current):
				current = rate
				if self.verbose > 0:
					print("Saving current best model at {} rate".format(rate))
				self.model.save(self.path + "/Current" + str(round(current*100)))
				self.model.get_vec_normalize_env().save(self.path + "/Current" + str(round(current*100)) + ".pkl")				
			if (rate >= self.successRates):
				if self.verbose > 0:
					print("Saving new best model at {} rate".format(rate))
					print("Saving new best model to {}.zip".format(self.save_path))
				self.model.save(self.save_path)
				self.model.get_vec_normalize_env().save(self.path+".pkl")
				return False
		return True 

	def numSuccess(self, path, eval_episodes):
		success = 0
		path = path + "/inSiAd2.csv"
		eval_episodes = eval_episodes
		info = pd.read_csv(path)
		rowsNum = info.shape[0]
		sus = []
		for i in range(rowsNum):
			sus.append(info.iloc[i]["s"].astype(int))
		boundary = 0
		if (len(sus) >= eval_episodes):
			boundary = eval_episodes
		else:
			boundary = len(sus)
		for i in range(-boundary, 0):
			success = success + sus[i]
		#print(sus)
		return success/boundary
			


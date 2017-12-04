import user_feature_generation
import tweet_feature_generation
import cPickle
import pandas as pd
from threading import Thread
import os
import gc

#current_dir = os.path.dirname(os.path.realpath(__file__))
#model_path = os.path.join(current_dir, 'models/tweet_models.pkl')
#user_model_path = os.path.join(current_dir, 'models/user_models.pkl')
#retrain_model_path = os.path.join(current_dir, 'models/retrain_model.pkl')

def load_models():
	with open('fake_recognition/models/retrain_model.pkl', 'rb') as f:
		s = f.read()
		gc.disable()
		retrain_model = cPickle.loads(s)
        gc.enable()
	return retrain_model


def main(tweet):
	tweet_features = None
	tweet_predictions = None
	user_predictions = None

	class ThreadWithReturnValue(Thread):
		def __init__(self, group=None, target=None, name=None,
					 args=(), kwargs={}, Verbose=None):
			Thread.__init__(self, group, target, name, args, kwargs, Verbose)
			self._return = None
		def run(self):
			if self._Thread__target is not None:
				self._return = self._Thread__target(*self._Thread__args,
													**self._Thread__kwargs)
		def join(self):
			Thread.join(self)
			return self._return

	t1 = ThreadWithReturnValue(target=tweet_feature_generation.main,args=(tweet, tweet_features, tweet_predictions,))
	t2 = ThreadWithReturnValue(target=user_feature_generation.main,args=(tweet, user_predictions,))
	t1.start()
	t2.start()
	tweet_values = t1.join()
	user_values = t2.join()
	tweet_features = tweet_values[0]
	tweet_predictions = tweet_values[1]
	user_predictions = user_values
	#print t2.join()
	print tweet_predictions
	print user_predictions

	if tweet_predictions == user_predictions:
		return tweet_predictions
	else:
		retrain_model = load_models()
		preds = retrain_model.predict(tweet_features)
	return preds[0]
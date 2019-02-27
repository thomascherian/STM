import numpy as np
import time as t
import string
import re
import sys
import logging
import pickle

class STM:
	def __init__(self,detectors=200,terminals=200,c=625,r=35,decay =0.1,learn_rate=0.2,sat_threshold=10):
		self.detectors = detectors
		self.terminals = terminals
		self.c = c
		self.r = r
		self.decay = decay
		self.learn_rate = learn_rate
		self.sat_threshold = sat_threshold
		#self.decay = np.random.uniform(0,(1/(float(self.r)-1)))	# Decay as per paper
		
		self.det_thr= np.zeros(self.detectors,)		# Determinor thresholds
		initw_denom= (self.r * (1 + self.c))		#To initialize weight matrix w. Ref

		self.w = (1 / (initw_denom + np.random.sample(self.detectors*self.terminals*self.r))).reshape(self.detectors,self.terminals,self.r) #initialization as per papers

		self.vt=np.zeros((self.terminals,self.r),)		#Activity of SR unit at time t
		self.Rij= np.zeros((self.detectors,self.terminals),np.int8)	# Weights from Modulator-Terminal
		self.vp=np.zeros((self.terminals,self.r),)			# Activity of SR unit at time t-1
		self.Ij=np.zeros(self.terminals,np.int8)			# Indicates input. Ij[j]=1, if terminal j is active at time t. All others 0
		self.Oi=np.zeros(self.detectors,np.int8)			# Indicates output of detector. Oi[z]=1 if z is the winner. All others 0

		self.Mi=np.zeros(self.detectors,np.int8)			# Activity of modulator i
		self.det_degree=np.zeros(self.detectors,np.int8)		# Degree of determinors 

		self.Ai=np.ones(self.detectors,)				# Sensitivity of detector
		self.Ei=np.zeros(self.detectors,)				# Activity of detectors.

		self.activity_detector = np.vectorize(self.activity_detector)	# Vectorizing the activity_detector function for parallelly calculating the activities of the detectors.

		self.indtochar = {} 	#Int to Component map
		self.charmap = {}	#Component to Int map
		self.timestamp = str(t.time()).replace('.','')
		self.train_vecs = []

#------------------ For log writing.----------------------------

		logf = "log/"+self.timestamp+".txt"
		logging.basicConfig(filename=logf, format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
	
#---------------------------------------------------------------

	def increase_r(self):			
		'''Method used to adjust model automatically when 'r' is increased dynamically'''

		# new_r=1.5*r
		new_r = self.r+10
		self.vt = np.append(self.vt,np.zeros((self.terminals,new_r-self.r),self.vt.dtype),axis=1)
		self.vp = np.append(self.vp,np.zeros((self.terminals,new_r-self.r),self.vp.dtype),axis=1)

		initw_denom= (new_r * (1 + self.c))		#To initialize weight matrix w. Ref
		extra_w = (1 / (initw_denom + np.random.sample(self.detectors*self.terminals*(new_r-self.r)))).reshape(self.detectors,self.terminals,(new_r-self.r)) #initialization as per papers
		self.w = np.append(self.w,extra_w,axis=2)
		self.r = new_r
		

	def increase_det(self):			
		''' Method for adjusting the model when 'detectors' is increased dynamically'''

		new_det = 2*self.detectors
		self.det_thr.resize(new_det,refcheck=False)
		self.Oi.resize(new_det,refcheck=False)
		self.Mi.resize(new_det,refcheck=False)
		self.det_degree.resize(new_det,refcheck=False)
		self.Ei.resize(new_det,refcheck=False)
		self.Ai = np.append(self.Ai,np.ones((new_det-self.detectors),self.Ai.dtype),axis=0)
		self.Rij= np.append(self.Rij,np.zeros((new_det-self.detectors,self.terminals),np.int8),axis=0)
		initw_denom= (self.r * (1 + self.c))	
		extra_w = (1 / (initw_denom + np.random.sample((new_det-self.detectors)*self.terminals*self.r))).reshape((new_det-self.detectors),self.terminals,self.r) #initialization as per papers
		self.w = np.append(self.w,extra_w,axis=0)
		self.detectors = new_det


#------------------------- Functions for expressions as in paper by D Wong. Refer paper for explanations---------------------

	def activity_detector(self,i):
		'''Method to calculate the activity of a detector i'''
		sr_act = np.copy(self.vt)
		sr_act[np.where( sr_act < self.Ai[i] )] = 0
		return np.sum( self.w[i] * sr_act )
	def activity_sra(self):
		'''Method to calculate the activity of the SR unit at time t'''
		self.vp = np.copy(self.vt)
		for (j,k), value in np.ndenumerate(self.vt):
			if k==0:
				self.vt[j,k] = self.Ij[j]
			else:
				self.vt[j,k] = max(0,((self.vp[j,(k-1)])-self.decay))
	# 		f.write (" SR "+str(j)+","+str(k)+"= "+str(vt[j,k])+"|"+str(vp[j,k-1])+"|"+str(vp[j,k-1]-decay)+"@"+str(max(0,vp[j,k-1]-decay)))
	#	vp=np.copy(vt)

	def update_wt(self,z):
		'''Method to update the weight for the winner detector z'''
		w_cap = np.copy(self.w[z])
		sr_act = np.copy(self.vt)
		sr_act[np.where(sr_act < self.Ai[z])] = 0
		w_cap = w_cap + (self.learn_rate * sr_act)      #removed Oi[z] since it will be 1 always, as we call for winner detector only. 
		denom = (self.learn_rate * self.c ) + np.sum(w_cap)
		self.w[z] = w_cap / denom 

	def update_sensitivity(self,i):
		'''Method to update the sensitivity associated with detector i'''
		if (self.det_degree[i] == 0 or self.det_degree[i]==1):
			self.Ai[i] = 1
		else:
			self.Ai[i] = max(0,(1-(self.decay * (self.det_degree[i]-1))))

	def update_det_threshold(self,z):
		'''Method to update the thredhold of winner detector z. Should be called only after updating the weights and only for the winner detector'''
		self.det_thr[z]= activity_detector(z)

	def activity_modulator(self,z):
		'''Method to calculate the activity of the modulator corresponding to the winner detector z'''			
		self.Mi[z] = np.sum ( self.Rij[z] * self.Ij ) 		# Oi[z] was removed from the eqn since we call for winner det only and hence it will be always 1

	def update_det_degree(self,z):
		'''Method to update the degree of the winner detector. This should be called before updating Oi, but after calculating activity of the modulator; ie with Oi (t-1) and Mi (t)''' 
		if self.Mi[z] == 0:
			self.det_degree[z] = self.det_degree[z] + 1     #Changed Oi[z] to 1; since it will be always 1 for winner z

	def one_shot_learn_Rzj(self,z):
		'''Method to learn the connections in case of prediction mismatch'''
		self.Rij[z] = np.copy( self.Ij )

#------------------------------------------------------------------------------------------

	def clear_vects(self):
		'''Method to reset the vectors used in the model'''
		self.vt.fill(0.0)
		self.vp.fill(0.0)
		self.Ij.fill(0)
		self.Oi.fill(0)
		self.Mi.fill(0)
		self.Ei.fill(0.0)
		logging.info('Vectors cleared')

	def write_model(self):
		'''Saves the model in the specified modelpath'''

		modelpath="model/"+self.timestamp+".dat"	
	
		mdl=open(modelpath,"w")

		params = np.array([self.detectors,self.terminals,self.r,self.c,self.decay,self.learn_rate])
		np.save(mdl,params)
		np.save(mdl,self.w)
		np.save(mdl,self.Rij)
		np.save(mdl,self.det_thr)
		np.save(mdl,self.det_degree)
		np.save(mdl,self.Ai)
		mdl.close()
		return modelpath

	def load_model(self,modelpath):	
		'''Loads the model from file and initialize the parameters and vectors from the saved values'''
		mdl = open(modelpath,"r")
		params = np.copy(np.load(mdl))
		self.detectors = params[0]
		self.terminals = params[1]
		self.r = params[2]
		self.c = params[3]
		self.decay = params[4]
		self.learn_rate = params[5]

		self.w = np.copy(np.load(mdl))
		self.Rij = np.copy(np.load(mdl))
		self.det_thr = np.copy(np.load(mdl))
		self.det_degree = np.copy(np.load(mdl))
		self.Ai = np.copy(np.load(mdl))

		mdl.close()

		self.vt = np.zeros((self.terminals,self.r),)			# Activity of SR unit at time t
		self.vp = np.zeros((self.terminals,self.r),)			# Activity of SR unit at time t-1
		self.Ij = np.zeros(self.terminals,np.int8)			# Indicates input. Ij[j]=1, if terminal j is active at time t. All others 0
		self.Oi = np.zeros(self.detectors,np.int8)			# Indicates output of detector. Oi[z]=1 if z is the winner. All others 0
		self.Mi = np.zeros(self.detectors,np.int8)			# Activity of modulator i
		self.Ei = np.zeros(self.detectors,)				# Activity of detectors.

		commited = np.count_nonzero(self.det_degree)
		logging.info("\nDetectors -> Committed: "+str(commited)+" Free: "+str(len(self.det_degree)-commited)+"\n")
		
		return self

	def fit(self,train,save = 0):
		'''The method does the training of the model. It fits the dataset to the model and returns the trained model'''

#		logging.info("\nDetectors: "+str(self.detectors)+"\nTerminals: "+str(self.terminals)+"\nr: "+str(self.r)+"\nC: "+str(self.c)+"\ndecay: "+str(self.decay)+"\nLearn Rate: "+str(self.learn_rate))

		logging.info("\nDetectors: %s \nTerminals: %s \nr: %s\nC: %s\ndecay: %s\nLearn Rate: %s", str(self.detectors),str(self.terminals),str(self.r),str(self.c),str(self.decay),str(self.learn_rate))

		self.indtochar = train.indtochar 	#Int to Component map
		self.charmap = train.charmap	#Component to Int map

		starttime = t.time()
		print train.train_set
	
		logging.info("\n*** Training Set ***\n"+str(train.train_set))
		logging.info("\nTraining Size (Line Count): "+str(train.dSetSize))
 
		self.sweep_count = np.zeros(train.dSetSize,np.int64)	
		for line in range(train.dSetSize):
			input_seq = train.train_set[line]
			seq_len = len(input_seq)
			input_vector = np.zeros(seq_len,np.int8) 	#Vector to hold the int equivalent for the components
			for j in range(seq_len):
                		input_vector[j] = train.charmap[input_seq[j]]
			self.train_vecs.append(input_vector)
		self.fit_data(train.dSetSize)

		endtime = t.time()
		tsweeps = np.sum(self.sweep_count)
		ttime = (endtime-starttime)/60
		print "\nTraining completed successfully!"
		print "Sweep Count:\n"
		print self.sweep_count
		print "\nTotal: "+str(tsweeps)
		print "\nTime taken for training: "+str(ttime)+" min\n"
		logging.info("\nTotal Sweeps: "+str(tsweeps)+"\nTime for training: "+str(ttime)+"\n")
		if save:
			write_model()
		return self

	def fit_data(self,size):
		t_miss=0
		while(1):
			for line in range(0,size):
				self.clear_vects()
				t_miss += self.sweep(self.train_vecs[line])
				np.add.at(self.sweep_count, [line], 1)
			if t_miss == 0:
				return
			else:
				t_miss=0

#------------------------Function defines a sweep--------------------------

	def sweep(self,input_vector):
		mis=0
		j = input_vector[0]
		self.Ij.fill(0)
		self.Ij[j]=1				# Presenting the input.
		self.activity_sra()			# Find activity of SR assemblies
		for l in range(1,len(input_vector)):

#--------------- Time 't' -----------------
		
			self.Ei = np.arange(self.detectors)
			self.Ei = self.activity_detector(self.Ei)

			self.Ei[ np.where( self.Ei < self.det_thr ) ] = 0.0
			z = np.argmax(self.Ei)			# Find the winner detector
			self.Oi.fill(0)
			self.Oi[z]=1					# Denotes that 'z' is the winner. All others  0 
			self.update_wt(z)				# Updates weights for z 					
			th = self.det_thr[z]
#			update_det_threshold(z)			# Threshold update for 'z'
			self.det_thr[z] = self.activity_detector(z) 	# Threshold update for 'z'

#-------------- Time 't+1' ----------------	

			j = input_vector[l]		
			self.Ij.fill(0)	
			self.Ij[j]=1					# New input is presented
#			activity_modulator(z)			# Calculate the Activity of Modulator of 'z'
			self.Mi[z] = np.sum ( self.Rij[z] * self.Ij )		# Calculate the Activity of Modulator of 'z'
			if self.Mi[z]==0:				# Activity of mod=0; That is, anticipation is wrong. Update the degree for detector 'z'
				dd = self.det_degree[z]
				self.update_det_degree(z)
				att = self.Ai[z]
				self.update_sensitivity(z)		# Lower the sensitivity for det 'z'
#				f.write( "degree updated for Det: "+str(z) +" from "+str(dd)+" to "+str(det_degree[z])+" Sens from: "+str(att)+" to: "+str(Ai[z])+"\n")
				mis= mis+1
				self.one_shot_learn_Rzj(z)
			else:
				ant = np.argmax(self.Rij[z])
#				f.write( "Detector "+ str(z)+" anticipated actv with terminal "+str(ant)+" for "+indtochar[ant]+" succsess!"+"\n" )
			self.activity_sra()				# Calculate Activity of SR assemblies. 
#			f.write( "Detector "+str(z)+" associated with next char "+indtochar[np.argmax(Rij[z])]+" with contxt= "+str(det_degree[z])+"\n")
		return mis


#------------------------Function generates the sequence --------------------------

	def generate(self,start,endtok):
		
		logging.info('Generating with start: %s and End: %s',start,endtok)
		self.clear_vects()
		anticipation = []
		if(len(start)!=1):
			for pcomp in range(len(start)-1):
				j = self.charmap[start[pcomp]]
				self.Ij.fill(0)
				self.Ij[j]=1
				self.activity_sra()
			j = self.charmap[start[pcomp+1]]
		else:
			j = self.charmap[start]
		
		self.Ij.fill(0)
		self.Ij[j]=1				# Presenting the input.
		self.activity_sra()			# Find activity of SR assemblies

		while(self.indtochar[j]!=endtok):

#--------------- Time 't' -----------------	

			self.Ei = np.arange(self.detectors)
			self.Ei = self.activity_detector(self.Ei)
			self.Ei[ np.where( self.Ei < self.det_thr ) ] = 0.0

			z = np.argmax(self.Ei)			# Find the winner detector
			self.Oi.fill(0)
			self.Oi[z] = 1					# Denotes that 'z' is the winner. All others  0 

#-------------- Time 't+1' ----------------	
		
							# New input is presented
			ant = np.argmax(self.Rij[z])
			anticipation.append(ant)
			j = ant
			self.Ij.fill(0)	
			self.Ij[j]=1	
			self.activity_sra()				# Calculate Activity of SR assemblies. 
		
		return (anticipation)

#----------------------------------------------------------------------------
	
	def getindtoCharMap(self):
		return self.indtochar

	def save(self):
		modelpath="model/"+self.timestamp+".dat"
		with open(modelpath,'wb') as modelfile:
			pickle.dump(self,modelfile)
		logging.info('Model saved in the path: %s',modelpath)
		return modelpath

	def load(self,modelpath):
		with open(modelpath,'rb') as modelfile:
			self = pickle.load(modelfile)
		return self

	def __getstate__(self):
        	odict = self.__dict__.copy() # copy the dict since we change it
		odict.pop('activity_detector')
		odict.pop('vt')
		odict.pop('vp')
		odict.pop('Oi')
		odict.pop('Mi')
		odict.pop('Ei')
		odict.pop('Ij')
        	return odict

    	def __setstate__(self, dict):
		self.__init__()
		ndict = self.__dict__.copy()
		ndict.update(dict)
        	self.__dict__.update(ndict)   # update attributes




#How to return the model?
'''
We need to arrange the code into 3 classes - DataSet, STM and Training/Evaluation
'''


class DataSet:
	def __init__(self,filepath):
		self.train_set = {}
		self.dSetSize = 0
		self.indtochar = {} 	#Int to Component map
		self.charmap = {}	#Component to Int map	
		if(filepath):
			self.load(filepath)
	def load(self,filepath):
		train_set_f=open(filepath,"r")
		num=0
		for line in train_set_f:
			self.train_set[num] = line
			num = num + 1
		self.dSetSize = len(self.train_set)

		chars = list(string.printable)

		for j in range(len(chars)):
			self.charmap[chars[j]]=j
			self.indtochar[j]=chars[j]

class Utils:
	def vectostring(self,vectlist,indtochar): 			# Takes a vector list and return string
		seq=[]
		for p in range(len(vectlist)):
			seq.append(indtochar[vectlist[p]])	# Converting out vector to component list
		string = ''.join(seq)				# Join list to get back the string
		return string

	def getTestSet(self,testFile):
		tcases = []
		with open(testFile) as file: 
			for line in file:
				tcases.append(line.rstrip())
		return tcases
	
	def save(self,model):
		modelpath="model/"+model.timestamp+".dat"
		with open(modelpath,'wb') as modelfile:
			pickle.dump(model,modelfile)
		logging.info('Model saved in the path: %s',modelpath)
		return modelpath

	def load(self,modelpath):
		with open(modelpath,'rb') as modelfile:
			model = pickle.load(modelfile)
		return model

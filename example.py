from src.stm import STM,DataSet,Utils
import signal
import pickle
import argparse
import sys

def handle_ctrlZ(signum,frame):
	#model.write_model()
	print('Exiting!')
	if(inp_args.saveFlag == 'yes'):
		print('Model saved in: %s',model.save())
	sys.exit(1)

signal.signal(signal.SIGTSTP, handle_ctrlZ)
signal.signal(signal.SIGINT, handle_ctrlZ)

p = Utils()

parser = argparse.ArgumentParser()
parser.add_argument('--train',help ='input file path',dest = 'trainSet',required = False, default = 'data/orig.txt')
parser.add_argument('--tf',help ='test file path containing start tokens of test instances; each instance in a new line',dest = 'testFile',required = True)
parser.add_argument('--load',help = 'Input the model path from where the model is to be loaded',dest = 'modelPath', required = False)
parser.add_argument('--save',help = 'To save the model. yes to Save. Default is no', default = 'no', dest = 'saveFlag', required = False)

inp_args = parser.parse_args()



model = STM(detectors=200,terminals=200,c=625,r=35,decay =0.1,learn_rate=0.2,sat_threshold=10)

if(inp_args.modelPath):
	path = inp_args.modelPath
	model = model.load(path)
else:
	if inp_args.trainSet:
		trainSet = DataSet(inp_args.trainSet)
		model = model.fit(trainSet)
		print inp_args.trainSet
		if(inp_args.saveFlag == 'yes'):
			print('Model saved in: %s',model.save())
	else:
		"--train argument is required for training a new model."
		sys.exit(1)

test_labels = p.getTestSet(inp_args.testFile)
print test_labels

for lbl in test_labels:
	print "\nGenerating from "+lbl+": "
	anticipate = model.generate(lbl,'#')
	print lbl+"->"+ p.vectostring(anticipate,model.getindtoCharMap())



# stm
"Short Term Memory(STM) Based Anticipation Model" is a neural model proposed by D Wang in 1995 for sequence learning. Instead of the popular gradient descent strategy, this makes use of a one-short hebbian learning method. Check the downloads folder for the base paper and the article by the same author. 

Here we implement the network in Python. The model can be invoked as follows:

`model = STM(detectors=200,terminals=200,c=625,r=35,decay =0.1,learn_rate=0.2,sat_threshold=10)`

[Check the base paper/ related article in the `\downloads` for the details about parameters]

STM class also supports the pickling and unpickling of the models. Logging is also supported for debugging.

`example.y` provides a walkthrough of the basic usage. It takes the following parameters:
* `--train` : Provide the input file path for training a new model
* `--tf` : Test file path containing start tokens of test instances; each instance in a new line.
* `--load` : In case of loading an existing model, provide the model path from where the model is to be loaded.
* `--save` : To save the model. yes to Save. Default is no.

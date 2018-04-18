### TODO

- create a network trained to go somewhere we want, in order to reuse this network to: 
  - go fetch some food
  - go fetch some points

- update the word so we can choose a custom reward function (in order to train whatever network we want to train)

- ###### test the training function of the model and the network

- ###### test if a trained agent doest really can be imported on a second run. Check the action distribution histogram for that

- ###### implement an output_adapter in order to decrease the output layer size for the avoid_ennemy_network

- ###### check another model for the avoid_ennemy_network (just to test the theory that a network cant't manage a problem because it's to small when the loss function behave unpredictably after some time)

- ###### do a agent's save after N episodes instead of just saving at the end to avoid some unpredictable errors that lead to a crash before having saved something. (at the moment, the agent just saves when exiting 😕)  WHATCH OUT FOR THE EXPECTION TRHOWNED WHEN  THE GRAPH IS TOO BIG! (prevent that!)

- ##### investigate the strange change of weights distribution after an export/import.

  - ##### correct this assuring the agent is writing its weights at the very end of its training!

- ##### remove the experience consumed by the agent in order to correct the action distribution problem BUT think about of the case this wouldn't change the homogeneity of the training set!







### Architecture followed

- a network to avoid ennemies

- a network to fetch something

- two networks using the fetch_something trained to: 

  - fetch food

  - fetch points

  These two branch will use the same train fetch_network but with different input_adapter so they can fetch either food or points

- a final network (the output layer) that will take as input: 

  - the avoid_ennemies network

  - the fetch_food network

  - the fetch_points network

  This final network will be trained once the other networks will be trained. The final network will output the action to do, when the world will have the following config:

  ```json
  world_config = {
      'ennemies' : True,
    	'use_food' : True,
    	'use_points' : True
  }
  ```

  ​


### DONE

- it would be a good practice to save EVERYTHING (agent's models save and tensorboard output) in only one folder. So we could have all our results / weights / hyperparameters in one single folder. And add the folder save in every output. (like episode i) **DONE**

- When an agent saves, print an output file with:  

  - all the hyperparameters 
  - the tensorboard output folder
  - the networks parameters (args)

  We can simply output the hyperparameters and the networks parameters (args) in the tensorflow text part? **DONE**

- test if one tensorboard has multilple sessions files, if we can choose which file to use (and not visualise only the most recent updated) **IT DOESNT WORK**


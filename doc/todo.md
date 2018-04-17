### TODO

- When an agent saves, print an output file with:  
  - all the hyperparameters 
  - the tensorboard output folder
  - the networks parameters (args)

  We can simply output the hyperparameters and the networks parameters (args) in the tensorflow text part?

- create a network trained to go somewhere we want, in order to reuse this network to: 
  - go fetch some food
  - go fetch some points

- update the word so we can choose a custom reward function (in order to train whatever network we want to train)

- ###### it would be a good practice to save EVERYTHING (agent's models save and tensorboard output) in only one folder. So we could have all our results / weights / hyperparameters in one single folder. And add the folder save in every output. (like episode i)

- ###### test the training function of the model and the network

- ###### test if one tensorboard has multilple sessions files, if we can choose which file to use (and not visualise only the most recent updated)





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


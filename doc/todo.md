### TODO

- create a network trained to go somewhere we want, in order to reuse this network to: 
  - go fetch some food
  - go fetch some points

- #### implement an output_adapter in order to decrease the output layer size for the avoid_ennemy_network

- ###### check another model for the avoid_ennemy_network (just to test the theory that a network cant't manage a problem because it's to small when the loss function behave unpredictably after some time)

- do a agent's save after N episodes instead of just saving at the end to avoid some unpredictable errors that lead to a crash before having saved something. (at the moment, the agent just saves when exiting 😕)  WHATCH OUT FOR THE EXPECTION TRHOWNED WHEN  THE GRAPH IS TOO BIG! (prevent that!)

- Add some tensorflow debugs on the experience set: the ratio $\frac{experience\_poped}{experience\_pool}$ has to be small. (And how many experience has been removed without training)

- ###### put the world observations with values between 0 and 1 and not between -0.5 and +0.5!

- ##### test with putting the empty spaces of an observation to zero instead to 1 

- ##### change the input model? Just to see if it changes drastically the learning 







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

- investigate the strange change of weights distribution after an export/import.

  - correct this assuring the agent is writing its weights at the very end of its training! $\rightarrow $ considered close on  commit*( eef753f  or eef753f29a08a954a404877764a190adf8b8f9ad)*  

- test if a trained agent doest really can be imported on a second run. Check the action distribution histogram for that  $\rightarrow $ considered close on  commit*( eef753f  or eef753f29a08a954a404877764a190adf8b8f9ad)*  

- test the training function of the model and the network

- make an new branch in order to have only files / commits < 100mo and continue working on it...

- investigate the training of the agent. It seems to promote the single action taken every step

- update the word so we can choose a custom reward function (in order to train whatever network we want to train)


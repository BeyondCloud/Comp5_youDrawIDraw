# Student ID, name of each team member.
    106061514 許鈞棠  106061536 廖學煒

# The models you tried during competition. 
<img src="./Report/Algo.jpg" style="height:300px">


-  RUSH strategy - Algorithm based sketch generator
    - Heedless of opponent's generated strokes, the algorithm we use focuses on how to finish our own target sketch AS QUICK AS POSSIBLE  
    - We use pretrained SRNN to generate target sketches of four object(ice,bulb...). To gaurantee the range of delta x,y falls between 0 and 1, a sigmoid function is add to the output. Because we wish our model to generate the sketch as efficiently as possible, we cherrypick the sketch with length less than 30.
    - In the first turn (assume we act first), we push out 10 stroke from the sketch stake and end the turn
    - After the opponent finishes their draw, we calculate the minimum step N_min required to return the position at which we end last turn (see blue stroke in the figure shown above). These strokes are set to undraw state to minimize the pollution of the artwork. After return that point, we pop another 10-N_min strokes (green stroke in the figure) from the sketch stack. This process will be repeated until the whole sketch finished.
    # 4 target sketches:
<img src="./Report/4.jpg" style="height:300px">
    

# List the experiment you did. For example, how you define your reward, different training strategy, etc.



# Other things worth to mention. For example, Any tricks or techniques you used in this task.

- The average length of generated strokes should be as long as possible 
    - Because we use rush strategy, we hope the generated strokes could cover much area at maximum efficiency. Therefore, the base sketch is designed to be done in 20 stroke. After the base sketch is done, we generate side feature to enhence the probability to pass the evaluator (Ex: generate an extra ice ball on top of the base ice sketch)
    
- Start from those sketch feature that Player1 possess but the opponent doesn't  
    - This is quite important since we don't want to help the opponent to complete their artwork. For example, if Player 1 draw Ice and the opponent draw microphone, Player 1 should start from the cone instead of the ice since the shape of ice is similar to the head of microphone. The evaluation will be done after turn 4, so our base sketch is designed to complete in about 20 strokes.



# Conclusions (interesting findings, pitfalls, takeaway lessons, etc.)

- Why Reinforcement Learning is hard
   - Although RL seems omnipotent according to the previous work (Alphago,Atari games), it has a lot of limitation. The reward function is hard to define and it won't work in the most case. On the other hand, the represention of state is another problem. If a single state include too much information, it could  consume large amount of memory, and it also makes the model hard to train. Plus, the action set is continuous in this competition, which is different from most Atari games with just a few actions to choose from. Last but not least, the sketch game is a sparse reward game. We could defined the reward of each generated strokes. However, this could be a tough work since the opponent's action needs to be considered as well.

- Evaluator weight is not avaliable
    - This is the most weird part of the competition. We're competiting something but some "rule" is private. Due to this limitation, we are not able to obtain the optimal solution and this is also a part of reason why I didn't use gradient descent method to solve this problem.

- Change of rule during the competition
    - Every time the rule of game is changed, the optimal solution will change as well. In this competition we need to train 12 models, which means we have to retrain (or even redesign) the model from scratch whenever the rule is modified. Consequently, RL approach might not be the elegant solution for this case. 
    
- Deep Reinforcement learning isn't mature enough 
    - During the competition, we have tried the Cartpole game and MountainCar game in this repo:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/7_Policy_gradient_softmax
    - DRL perform well on the Cartpole game ( after 82 episode the agent reaches nice score). However, the Mountain car game take a little bit longer to achieve high score. Note that the mountain car game is a simple game (only two actions to choice, intuitive reward and clear state input). If DRL takes a long time on such simple model, how do you except it could perform well on our complex sketch competition task (continuous choice, sparse reward and complex state input)?  


# Result (TA60, TA80)
- Our evaluator have 0.95 accuracy on the training data
<img src="./Report/TA_60_GG.jpg" style="height:300px">
<img src="./Report/TA_80_GG.jpg" style="height:300px">


# LeaderBoard

<img src="./Report/LeaderBoard.jpg" style="height:300px">


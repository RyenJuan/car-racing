# car-racing
OpenAI Gym Car-Racing

A short project to mix classical controls in an environment meant for reinforcement learning. 

The OpenAI Gym provides many examples for developing reinforcement leaning algorithms and most importantly, a car racing environment. I took inspiration from an article by Kishan Kartha who had applied PID control to the car racing environment. Starting from his example, I made a couple of changes to try and get better results. Better, in this context, refers to completing the track faster and not going off-road.

The car itself has two inputs, steering and acceleration, and it drives around a randomly generated track each time the simulation is run. The original algorithm that Kishan used PID to stay in the center of the road at all times and to drive at more or less constant speed. Since the simulation places the 'camera' on top of the car, the car is effectively 'stationary' while the road/world moves around the car. The 'sensors' of the car can only detect the edges of the road which is how the controller detects a turn.

Since the car can recognize turns and straights, I added a couple small changes. 

<ul>
  <li>Accelerate to full throttle when the road is straight</li>
  <li>Measure the angle of the turn before deciding to turn</li> 
</ul>

Since it is always faster to go straight, I added a rule to keep the car straight if the turn angle is low enough. Sharper turns deserve more steering input and lower throttle while smaller turns could be driven straight. Fine-tuning these parameters achieved faster results and higher scores. Surprisingly, this worked very well and some simulations were reminiscent of optimal racing lines.

Source: https://medium.com/@kartha.kishan/solving-openai-carracing-v0-using-image-processing-5e1005ee0cb

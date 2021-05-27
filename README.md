# thermal_beam

The simulation serves to estimate useful atom flux (trappable atoms/s) at the MOT based on TC/ZS optical setup, vacuum system geometry, and relevant oven parameters (nozzle type, temperature. etc). 


The code is broken down into a couple of major sections, in part due to our choice of collimation in the oven. Erbium atoms exiting the effusion cell will pass through a 3D-printed array of titanium micronozzles, acting as small apertures to ensure low divergence from the vacuum system axis. To faithfully simulate the behavior of the atoms interacting with these microtubes, the first code section is the micronozzle simulation. 

The micronozzle simulation generates atoms with random positions in the oven, and samples their velocities from a Maxwell-Boltzmann distribution at the oven temperature. Atoms are then dynamically evolved by a forward-Euler integration scheme (1ns timestep) of their equations of motion, given by Newton's 2nd law. In this simple picture the only force present is gravity, since atom-atom interactions can be neglected in the molecular flow regime. The other ingredient for their motion is rethermalization inside the micronozzles. If an atom collides with the wall of a nozzle, it "sticks" and is re-emitted with a velocity randomly chosen from a Maxwell-Boltzmann distribution at a temperature hotter than that of the effusion cell called the "hot lip temperature". This favorably pushes atoms toward the exit. 

With the starting kinematics for the atoms determined, they can be propagated through the rest of the vacuum system toward the MOT chamber. 

 

The first obstacle is the pre-TC aperture. This removes atoms with high divergence before they interact the cooling light, so as to reduce the amount of light wasted on atoms that are already lost causes. The choice of aperture diameter will be discussed later on. 

Next is the transverse cooling (TC) section. The light's detuning from the 401nm transition, intensity, and waist are inputs here. Atoms feel an impulse due to the scattering of light and their trajectory is updated accordingly. 

The atoms then pass through the Lithium oven flythrough, which lets through the 2/3 of the Erbium atoms in the upper angular section.

Next is the Zeeman slower. Here atoms are slowed to near the MOT capture velocity. However, due to the design not being finalized and direct simulation of it being complicated, the ZS dynamics are left out in the end. This is justified as a well-tuned ZS will not significantly change the divergence angle of the atoms, allowing naïve ray-tracing and a MOT capture condition based on atoms having less than the capture velocity of the ZS. 

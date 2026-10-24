# Team SM2DH2C

## Team Members
1. Sofia Monge - sofiamonge5
2. Helena Bianco Oliveira - biancooh
3. Diya Deepu - diyadeepuOSU
4. Mari Tokuda - tokudam-dot
5. Heeseo Kim - heeseoKOSU
6. Chase Vanderlip - vanderlc-OSU
7. Mylan Bui - MylanBui2

## Product Goal
Our product goal is to build an efficient and reliable human in the loop autonomous drone system that can effectively locate a missing person. The drone should handle tasks independently, like navigation, avoiding obstacles, and deciding search areas. The human operator can focus solely on the higher level in this system, simulating a search and rescue operation in a 3D environment. This system should take in live inputs from a human operator and seamlessly integrate it into the visual display. 

## Primary Users
The primary users for this system are search and rescue team/first responder operators. They would operate the drone during missions to effectively find missing people, constantly monitoring mission status, and sending groups accordingly.

## User Scenario
For context, we are in an urgent situation where there is a missing person that needs to be located within a certain environment. 
1. The user (operator) initiates the search mission by interacting with the START button in the simulation interface.
2. As the mission starts, the drone autonomously navigates through the search environment.
3. The operator is able to give instructions to the drone during the mission, based on incoming relevant information about the target (e.g. sighting reports, dangerous zones, etc.).
4. The drone is able to adapt to the given operator instructions by replanning its navigation path.
5. During the mission, the drone will send its current mission status to the operator, notifying them about its current location, planned destination, detected obstacles, and any reroutes it does (whether by obstacle avoidance or operator guidance).
6. When the drone locates the missing person, it reports back to the operator, stating that it has successfully found the target. 
7. The search ends when the drone finds the target, and the simulation terminates.

## Functional Requirements
- Environment Set-Up: The system must prompt users to take in geographical input before the SAR simulation begins.
- Environment Visualization: The system must be able to take geographical input from the user and represent realistic terrain and object features based the given information.
- Mission Initiation: The system must allow a search mission to be started by the operator. 
- Autonomous Search: The system must search independently without continuous human operator’s guidance. 
- Operator Guidance: The system must accept information, guidance, or commands from a human operator during the mission.
- Adaptive Drone Behavior: The system must be able to adjust drone behavior or search routes based on operator input. 
- Mission Monitoring: The system must keep the human operator on update with drone location, mission status, or search progress.
- Obstacle Detection/Avoidance: The drone must be able to detect obstacles and re-route based on the updated information. 
- Target Detection: The system must be capable of finding the missing person. 

## Non-Functional Requirements
- Efficient Search Performance: The system should be able to decide the efficient route to complete the mission.
- Clear User Interface: The operator interface should be in an intuitive and professional manner. 
- Obstacle Handling: The drone should recognize the obstacles and re-route for an efficient search path. 
- Fast System Response: The system should respond promptly to human operator’s inputs. 

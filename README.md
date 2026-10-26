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
- A hiker has gone missing in a forest, and the search-and-rescue team is tasked with locating the missing person as quickly as possible. A human operator affiliated with the rescue team uses the drone search-and-rescue simulation system to conduct a search mission. The operator receives the report about the missing person, containing information about their Last Known Position (LKP) and other relevant details. 
- To prepare the mission, the user enters the geographical information of the missing person’s LKP, along with any other known information required for the search. The operator initiates the search mission by clicking the START button in the simulation interface. The system displays a 3D environment representing the real-world geographic characteristics of the missing person’s LKP once the mission starts. 
- During the mission, the operator monitors the drone through the simulation interface, which displays the drone’s current location, mission status, planned destination, detected obstacles, and any rerouting. For example, if the drone encounters an obstacle, the system adjusts the navigation path and informs the operator about the reroute.
- Midway through the search, the operator can guide the mission by entering new information into the system, which updates the drone's search data and recalculates its route. For example, the rescue team receives a report that someone was seen near a trail. The operator marks the the location as a possible sighting and the system confirms the update and reroutes the drone to search that area. 
- When the drone detects the missing person, the system updates the mission status on the operator dashboard that the target has been found and the location where the target was found. The user can communicate this information to the rescue team to locate the missing person. The mission status is updated as completed, and the simulation terminates. 
 
## User Requirements
- Mission Preparation: The operator needs to enter the missing person’s Last Known Position (LKP) and other known details before the mission starts.
- Mission Start: The operator needs to start the search mission from the simulation interface.
- Search Area Display: The operator needs to see a 3D environment that represents the real-world area around the LKP.
- Autonomous Search: The operator needs the drone to search on its own without manually controlling it.
- Mission Monitoring: The operator needs to see the drone’s current location, mission status, planned destination, and detected obstacles during the mission.
- Reroute Notification: The operator needs to be informed when the drone changes its route.
- Mission Guidance: The operator needs to enter new information during the mission, such as a possible sighting, so the drone can update its search and route.
- Guidance Confirmation: The operator needs confirmation that the system received their input and updated the mission.
- Target Found Notification: The operator needs to be notified when the missing person is found, along with their location, so it can be passed to the rescue team.

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

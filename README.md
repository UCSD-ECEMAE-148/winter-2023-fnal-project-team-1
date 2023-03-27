<div id="top"></div>

<h1 align="center">Stoplight, Streetsign and Person Detection </h1>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ECE148-WI-23-Team-1/CV-Sign-and-Person-Detection">
    <img src="images\UCSDLogo_JSOE_BlueGold.png" alt="Logo" width="400" height="100">
  </a>
<h3>MAE-ECE 148 Final Project</h3>
<p>
Team 1 Winter 2023
</p>
</div>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#team-members">Team Members</a>
    </li>
    <li><a href="#final-project">Final Project</a></li>
    <li><a href="#early-quarter">Early Quarter</a></li>
      <ul>
        <li><a href="#mechanical-design">Mechanical Design</a></li>
        <li><a href="#electronic-hardware">Electronic Hardware</a></li>
        <li><a href="autonomous-laps">Autonomous Laps</a></li>
      </ul>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- TEAM MEMBERS -->
## Team Members

<div align="center">
    <img src="images\Team.jpg" alt="Logo" width="500" height=400">
    <p align = "center">Arturo Amaya (Left), Arjun Naageshwaran (Middle), Hariz Megat Zariman (Right)</p>
</div>

<h4>Team Member Major and Class </h4>
<ul>
  <li>Arturo Amaya - Computer Engineering (EC26) - Class of 2023</li>
  <li>Arjun Naageshwaran - MAE Ctrls & Robotics (MC34) - Class of 2024</li>
  <li>Hariz Megat Zariman - Computer Engineering (EC26) - Class of 2024</li>
</ul>

<!-- Final Project -->
## Final Project

The goal of this project was a three-fold application of computer vision. Using the OAKD Lite camera, we aimed to recognize stoplights, traffic signs and people and use those visual inputs to change the state of our car.

### Primary goals:
1) Red, yellow and green stoplights would make the car stop, slow down, and go
2) (Stretch goal) Stop signs, speed limit signs, and right turn signs would be recognized and make the car perform their designated function
3) Individuals could be recognized and followed by the car with dynamic throttle and steering, while the car simultaneously recognized and performed stoplight and streetsign functions

#### Goal : Red, Yellow and Green stoplights
[<img src="images\video1.png" width="50%">](https://drive.google.com/file/d/14FbdGz32Ei6No-sINCMqQw-Jw_kPOdvK/view?resourcekey)

The car was able to detect red and green images succesfully. When shown red, the car completely stopped, and would only move forward when a green light was shown. Due to time constraints and inconsistent lighting conditions, the car was not able to detect a yellow light consistently but would be able to reduce throttle on detection.

#### Goal : Person Following
[<img src="images\video2.png" width="50%">](https://drive.google.com/file/d/1g3es-dhcG1CYK04iPISBUkA-pttMvrR2/view?usp=sharing)

The car was able to identify person objects using the OAKD camera. By doing so, it was then capable of following a person by adjusting steering values based on how far the person strayed from the center of the camera's FOV. Furthermore, the car adjusted its throttle such that the further away a person is, the faster it goes and stops when approaching to near.

#### Stretch Goal: Street Sign Detection
Our team was able to obtain obtain the [Mapillary Dataset](https://www.mapillary.com/datasets) (containing many real-world street signs) and extract the relevant files and labels which were useful to our project (such as the U-turn and stop sign). Unfortunately, due to time constraints, unlabeled images and issues with training the dataset (slow locally, incorrect format for GPU cluster and many others), we were unable to reach this goal on time. However, we were able to implement the movement methods for the car object if it were to identify these signs,

See `car.py` for these methods.

<!-- Early Quarter -->
## Early Quarter

#### Mechanical Design
Prefabricated parts of the car were given to us, but parts like the base plate, dual camera mount, LIDAR stand, and cases for sensitive electronics (VESC, GNSS, Servo PWM, etc.) were either 3D printer or laser cut. These are a few of the models compared to their real-life, 3D printed counterparts.

| Top Camera Mount           |  Bottom Camera Mount | LIDAR Mount |
:-------------------------:|:-------------------------:|:-------------------------:
[<img src="images\Top_camera_mount.PNG">](https://github.com/ECE148-WI-23-Team-1/CV-Sign-and-Person-Detection/blob/main/images/Top_camera_mount.PNG)  |  [<img src="images\Bottom_camera_mount.PNG">](https://github.com/ECE148-WI-23-Team-1/CV-Sign-and-Person-Detection/blob/main/images/Bottom_camera_mount.PNG) | [<img src="images\LIDAR_Mount.PNG">](https://github.com/ECE148-WI-23-Team-1/CV-Sign-and-Person-Detection/blob/main/images/LIDAR_Mount.PNG)


Camera Mount (Physical)           |  LIDAR Mount (Physical)
:-------------------------:|:-------------------------:
[<img src="images\20230324_163531.jpg" width = "300" height="500">](https://github.com/ECE148-WI-23-Team-1/CV-Sign-and-Person-Detection/blob/main/images/20230324_163531.jpg)  |  [<img src="images\20230324_163548.jpg" width = "300" height="500">](https://github.com/ECE148-WI-23-Team-1/CV-Sign-and-Person-Detection/blob/main/images/20230324_163548.jpg)|

#### Electronic Hardware
Our team used only the electronic components given to us. In particular, we focused primarily on the OAK-D camera, Jetson NANO and the GNSS board (only used for the 3 GPS Autonomous Laps). When assembling the circuit, we used the following circuit diagram (given by the TAs of the class):
<div align="center">
    <img src="images\Circuit Diagram.png">
</div>

#### Autonomous Laps
Below is a youtube playlist of the car completing 3 autonomous laps using the DonkeyCar framework under different conditions. 

[<img src="images\playlist1.png">](https://www.youtube.com/watch?v=IvP6Bl-0CmE&list=PLhr_F_bR8N4bZ2Y7_KSgR6yGpgRoOXpH-)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

**Credited Code Examples:**
* [Traffic light example Git](https://github.com/HevLfreis/TrafficLight-Detector/blob/master/src/main.py)
* [DepthAI Git](https://github.com/luxonis/depthai)
* [VESC Object Drive Folder](https://drive.google.com/drive/folders/1SBzChXK2ebzPHgZBP_AIhVXJOekVc0r3)
* [DonkeyCar Framework](https://docs.donkeycar.com/guide/host_pc/setup_windows/)

*Special thanks to Professor Jack Silberman, and TAs Kishore Nukala and Moises Lopez (WI23)*

<!-- CONTACT -->
## Contact

* Hariz Megat Zariman - hzariman@gmail.com | mqmegatz@ucsd.edu
* Arjun Naageshwaran - arjnaagesh@gmail.com | anaagesh@ucsd.edu
* Arturo Amaya - 

<!-- MARKDOWN TEMPLATE INFORMATION -->
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

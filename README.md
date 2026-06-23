# OVERDRIVE
> 2026 SKKU Autonomous Driving Competition
<br/>

## ✨ Quick Start
> The environment is based on `Ubuntu 22.04 LTS`.
<br/>

### 1️⃣ Install ROS 2 Humble
Set Locale
```bash
sudo apt update && sudo apt install locales
sudo locale-gen ko_KR ko_KR.UTF-8 en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```
<br/>

Enable Ubuntu Repositories
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```
<br/>

Add ROS 2 GPG Key and Repository
```bash
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
<br/>

Install ROS 2 Humble Packages
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install ros-humble-desktop -y
```
<br/>

Install Development and Build Tools
```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-argcomplete -y
```
<br/>

Set Up Environment Variables
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
<br/>

Verify the Installation (Talker-Listener Test)  
* First Terminal Tab: `ros2 run demo_nodes_cpp talker`  
* Second Terminal Tab: `ros2 run demo_nodes_py listener`

Successful communication between the two nodes confirms that the ROS 2 Humble installation is working correctly.
<br/>

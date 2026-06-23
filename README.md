# OVERDRIVE
> 2026 SKKU Autonomous Driving Competition
<br/>

## ✨ Installation Guide
> The environment is based on `Ubuntu 22.04 LTS`.
<br/>

### ⚠️ Attach USB device from Windows to WSL2 using `usbipd`
Install `usbipd` via PowerShell as Administrator
```bash
winget install --id dorssel.usbipd-win --exact
```
<br/>

Reopen PowerShell as Administrator and List USB Devices
```bash
usbipd list
```
<br/>

Identify LiDAR Bus ID and Attach to WSL2 (Keep this terminal open)
```bash
usbipd bind --busid 5-1
usbipd attach --busid 5-1 --wsl --auto-attach
```
<br/>
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
<br/>

### 2️⃣ Install LiDAR Driver Package
Install LiDAR Driver Package
```bash
sudo apt update
sudo apt install ros-humble-rplidar-ros -y
```
<br/>

Configure LiDAR USB Port Permissions
```bash
sudo chmod 666 /dev/ttyUSB0
```
<br/>

Run the LiDAR Node
```bash
ros2 launch rplidar_ros rplidar_a1_launch.py
```
<br/>

Verify LiDAR Data Reception in new terminal
```bash
ros2 topic echo /scan
```
<br/>
<br/>

### 3️⃣ Install Arduino CLI
Install Arduino CLI
```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
<br/>

Initialize Arduino CLI and Update Core Package Index
```bash
arduino-cli config init
arduino-cli core update-index
```
<br/>

Install Arduino Board Cores
```bash
arduino-cli core install arduino:avr
```
<br/>

Verify the Arduino Board
```bash
arduino-cli board list
```
<br/>

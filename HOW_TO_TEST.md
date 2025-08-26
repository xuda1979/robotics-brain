# How to Test the Robot Brain Virtually

This project includes a complete virtual testing environment using the **Webots** robotics simulator. This document provides a step-by-step guide to get it running.

Webots is a powerful, open-source, and industry-standard simulator that allows for the development and testing of robotics algorithms without the need for physical hardware. This project is already deeply integrated with it.

## 1. Installation

You will need to install the Webots simulator and the required Python libraries.

### a) Install Webots

Download and install the latest version of Webots from the official website:

- **[Webots Download Page](https://cyberbotics.com/#download)**

Follow the installation instructions for your operating system (Windows, macOS, or Linux).

### b) Install Python Dependencies

The Python script requires `torch` and `matplotlib`. These can be installed using `pip` and the `requirements.txt` file.

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/robotics-brain.git
    cd robotics-brain
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `controller` library needed to communicate with Webots is included with the Webots installation and does not need to be installed via pip.*

## 2. Running the Simulation

Testing is a two-step process:
1.  Start the simulation in the Webots application.
2.  Run the Python controller script to connect to the simulation and control the robot.

### Step 1: Open the World in Webots

1.  Launch the Webots application that you installed.
2.  Go to `File > Open World...`.
3.  Navigate to the `worlds` directory within this project's folder and select `default.wbt`.
4.  The simulation environment, containing a robot, obstacles, and a target, will load.
5.  **Press the "play" button** (the triangle icon) in the Webots toolbar to start the simulation. The simulation will now be running and waiting for a controller to connect.

![Webots Play Button](https://www.cyberbotics.com/doc/images/webots/play_buttons.png)

### Step 2: Run the Controller Script

1.  Open a terminal or command prompt.
2.  Make sure you are in the root directory of this project.
3.  Run the `main.py` script with the `webots` environment flag:
    ```bash
    python main.py --environment webots
    ```

## 3. Expected Outcome

When you run the Python script, you will see output in your terminal indicating that the planner is running. The script will:
1.  Connect to the running Webots simulation.
2.  Get the robot's current position and the target's position from the simulator.
3.  Run the GPU-accelerated planner to calculate an optimal path.
4.  Send the first action of the plan to the robot's motors.
5.  You will see the robot in the Webots window move from its starting position.

The terminal output will look something like this:
```
正在使用设备: cuda
正在设置Webots环境...
Webots环境已创建。
正在初始化模型...
规划器已初始化。

--- 开始规划 ---

正在执行最终计划（动作序列）:
tensor([[ 0.1000,  0.0500],
        [ 0.1000,  0.0500],
        ...
        [ 0.1000, -0.0500]], device='cuda:0')

--- Webots仿真循环 ---

--- 规划结束 ---
```

This completes a full cycle of virtual testing: the Python-based "brain" perceives the virtual world, plans a path, and acts in the simulated environment.

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

## 2. Running the Virtual Tests

This project supports two main types of virtual testing: a lightweight 2D environment for rapid prototyping and a high-fidelity Webots simulation for more realistic testing.

### a) 2D Environment (with Domain Randomization)

The 2D environment is excellent for quickly testing the planner's logic.

To run with the default, fixed environment:
```bash
python main.py --environment 2d
```

A key feature for building robust algorithms is **Domain Randomization**. This creates a new, randomized environment every time you run it. To use this, add the `--random-env` flag:
```bash
python main.py --environment 2d --random-env
```
This will generate a random number of obstacles with random shapes and sizes, along with random start/goal positions. This is a powerful way to ensure the planner is not overtuned to a single scenario.

### b) Automated Webots Simulation

The testing workflow with Webots is now fully automated. You no longer need to open the simulator manually.

Simply run the main script with the `webots` environment flag:
```bash
python main.py --environment webots
```
This command will automatically:
1.  Launch the Webots simulator in a minimized window.
2.  Load the `worlds/default.wbt` file.
3.  Start the simulation.
4.  Run the Python controller to connect to it and control the robot.
5.  Shut down the Webots application when the script finishes.

#### Headless Mode

For automated testing or running on a server without a graphical interface, you can use headless mode. This runs the simulation without rendering the 3D view, which saves resources.
```bash
python main.py --environment webots --headless
```

## 3. Running the Unit Tests

The project now includes a test suite using `pytest` to ensure the core components are working correctly.

1.  **Install test dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the tests:**
    It is recommended to run `pytest` through the Python module flag to ensure it uses the correct environment:
    ```bash
    python -m pytest
    ```

You should see output indicating that all tests have passed. This is a great way to verify your setup and ensure that recent changes have not introduced any bugs.

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

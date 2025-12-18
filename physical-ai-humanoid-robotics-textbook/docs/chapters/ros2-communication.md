---
sidebar_position: 3
title: ROS 2 Communication Patterns - Connecting the Robotic Body
description: Understanding communication patterns in Robot Operating System 2
keywords: [ros2, communication, publisher, subscriber, service, action]
---

# ROS 2 Communication Patterns - Connecting the Robotic Body

## Overview of Communication Patterns

In humanoid robotics, effective communication between different components is crucial for coordinated behavior. Just as the human nervous system uses different types of signals for different purposes, ROS 2 provides multiple communication patterns to suit various needs in robotic systems.

## Publisher-Subscriber Pattern (Topics)

The publisher-subscriber pattern is the most common communication mechanism in ROS 2. It enables asynchronous, decoupled communication between nodes, making it ideal for sensor data distribution and status updates.

### Basic Implementation

```python
# Publisher node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

    def publish_sensor_data(self):
        msg = String()
        msg.data = f"Sensor reading: {self.get_clock().now().nanoseconds}"
        self.publisher.publish(msg)

# Subscriber node
class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received sensor data: {msg.data}')
```

### Quality of Service (QoS) Considerations

For humanoid robotics applications, QoS settings are critical:

- **Reliability**: Use RELIABLE for critical data like safety information
- **Durability**: Use TRANSIENT_LOCAL for data that new subscribers should receive immediately
- **History**: Use KEEP_LAST for real-time data, KEEP_ALL for historical analysis
- **Depth**: Balance between memory usage and data retention

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For critical sensor data
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)
```

## Service Pattern

Services provide synchronous request-response communication, suitable for operations that require a specific result before proceeding.

### Service Implementation

```python
# Service definition (in srv/CalculateIK.srv)
float64[] joint_angles
---
float64[] target_pose
bool success

# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class IKService(Node):
    def __init__(self):
        super().__init__('ik_service')
        self.srv = self.create_service(
            CalculateIK,
            'calculate_inverse_kinematics',
            self.calculate_ik_callback
        )

    def calculate_ik_callback(self, request, response):
        # Calculate inverse kinematics
        response.joint_angles = self.compute_ik(request.target_pose)
        response.success = True
        return response

# Service client
class IKClient(Node):
    def __init__(self):
        super().__init__('ik_client')
        self.cli = self.create_client(CalculateIK, 'calculate_inverse_kinematics')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, target_pose):
        request = CalculateIK.Request()
        request.target_pose = target_pose
        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Action Pattern

Actions are designed for long-running tasks that require feedback and the ability to cancel. They're ideal for navigation, manipulation, and other complex behaviors in humanoid robots.

### Action Implementation

```python
# Action definition (in action/MoveArm.action)
# Goal definition
float64[] target_joint_positions
float64 max_time
---
# Result definition
bool success
string message
---
# Feedback definition
float64[] current_joint_positions
float64[] joint_velocities
float64 progress_percentage

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class MoveArmActionServer(Node):
    def __init__(self):
        super().__init__('move_arm_action_server')
        self._action_server = ActionServer(
            self,
            MoveArm,
            'move_arm',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        # Accept or reject a goal
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept or reject a cancel request
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = MoveArm.Feedback()
        feedback_msg.current_joint_positions = [0.0] * 7
        feedback_msg.progress_percentage = 0.0

        # Simulate arm movement
        for i in range(0, 100, 10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return MoveArm.Result()

            # Simulate progress
            feedback_msg.progress_percentage = float(i)
            goal_handle.publish_feedback(feedback_msg)

            # Sleep to simulate work
            await asyncio.sleep(0.5)

        goal_handle.succeed()
        result = MoveArm.Result()
        result.success = True
        result.message = 'Arm movement completed successfully'
        return result
```

## Communication in Humanoid Robot Architecture

### Sensor Integration
Humanoid robots typically have numerous sensors (IMU, cameras, force/torque sensors, joint encoders). The publisher-subscriber pattern is ideal for distributing this data:

- Joint encoders publish to `/joint_states`
- IMU publishes to `/imu/data`
- Cameras publish to `/camera/image_raw`
- Force/torque sensors publish to `/ft_sensor/wrench`

### Control Systems
Control systems often use a combination of patterns:

- High-frequency control: Publisher-subscriber for real-time commands
- Trajectory planning: Services for path computation
- Complex behaviors: Actions for coordinated movements

### Inter-Module Communication
Different software modules communicate through well-defined interfaces:

```yaml
# Example of communication architecture
Perception Module:
  - Publishes: /perception/objects, /perception/obstacles
  - Subscribes: /camera/rgb/image_raw, /camera/depth/image_raw

Planning Module:
  - Publishes: /planned_trajectory, /navigation/goal
  - Subscribes: /perception/obstacles, /tf, /odom
  - Services: /plan_path, /get_robot_pose

Control Module:
  - Publishes: /joint_commands, /base_velocity
  - Subscribes: /planned_trajectory, /joint_states
  - Actions: /move_arm, /walk
```

## Performance Considerations

### Message Size
- Keep messages small for high-frequency topics
- Use compression for large data like images
- Consider subsampling for high-bandwidth sensors

### Network Communication
- Use reliable transport for critical data
- Optimize for bandwidth when communicating with remote nodes
- Consider security implications for networked robots

### Real-time Requirements
- Use real-time capable DDS implementations for critical systems
- Consider CPU usage of serialization/deserialization
- Monitor communication latencies for time-sensitive operations

## Best Practices for Humanoid Robotics

1. **Consistent Message Types**: Use standard message types where possible
2. **Clear Naming Conventions**: Follow ROS conventions (e.g., `/robot_name/sensor_type/data`)
3. **Error Handling**: Implement robust error handling for communication failures
4. **Monitoring**: Use tools like `ros2 topic hz` to monitor message rates
5. **Documentation**: Document all published topics and subscribed services

## Summary

ROS 2 communication patterns provide the essential infrastructure for coordinating the complex systems in humanoid robots. The publisher-subscriber pattern handles real-time data distribution, services manage request-response interactions, and actions coordinate long-running behaviors. Understanding these patterns and their appropriate use cases is crucial for building effective humanoid robotic systems.

## References

1. ROS 2 Communication: https://docs.ros.org/en/humble/Concepts/About-Topic-Concepts.html
2. QoS Implementation: https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html
3. Actions Design: https://design.ros2.org/articles/actions.html

## Exercises

1. Implement a publisher-subscriber pair for IMU data
2. Create a service for calculating joint angles from Cartesian coordinates
3. Design an action for walking behavior with feedback on step progress
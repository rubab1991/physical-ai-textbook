---
sidebar_position: 4
title: ROS 2 Navigation - Moving with Purpose
description: Understanding navigation systems in Robot Operating System 2
keywords: [ros2, navigation, navigation2, path planning, localization]
---

# ROS 2 Navigation - Moving with Purpose

## Introduction to Navigation in ROS 2

Navigation is a fundamental capability for humanoid robots, enabling them to move purposefully through environments while avoiding obstacles. The Navigation2 stack in ROS 2 provides a comprehensive framework for robot navigation, building upon the Navigation stack from ROS 1 with improved architecture, performance, and features.

## Navigation2 Architecture

Navigation2 follows a behavior tree architecture that allows for complex, reactive navigation behaviors. The system is composed of several key components:

### Core Components

1. **Navigator**: The main controller that coordinates navigation tasks
2. **Planner Server**: Handles global and local path planning
3. **Controller Server**: Manages local path following and obstacle avoidance
4. **Recovery Server**: Provides recovery behaviors when navigation fails
5. **BT Navigator**: Uses behavior trees to orchestrate navigation tasks

### Global and Local Planning

Navigation2 distinguishes between global planning (finding a path from start to goal) and local planning (executing that path while avoiding obstacles):

```python
# Example of using Navigation2 in a custom node
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

    def send_goal(self, x, y, theta):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = theta

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')
```

## Localization Systems

Localization is the process of determining the robot's position in a known map. Navigation2 supports multiple localization approaches:

### AMCL (Adaptive Monte Carlo Localization)
AMCL is the standard localization approach in ROS 2, using particle filters to estimate robot pose based on sensor data and a known map.

```yaml
# Example AMCL configuration
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
```

## Path Planning Algorithms

Navigation2 supports multiple path planning algorithms through its plugin architecture:

### Global Planners
- **NavFn**: Fast-marching method for global path planning
- **GlobalPlanner**: Implementation of Dijkstra and A* algorithms
- **CarrotPlanner**: Finds a valid point near the goal if exact goal is not reachable

### Local Planners
- **DWB (Dynamic Window Approach)**: Local planner for velocity-based control
- **TEB (Timed Elastic Band)**: Trajectory optimization for smooth paths
- **SBC (Simple Band Checker)**: Collision checking for trajectories

```python
# Example of TEB local planner configuration
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: true
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
```

## Behavior Trees in Navigation

Navigation2 uses behavior trees to create complex navigation behaviors. This allows for sophisticated decision-making and reactive behaviors:

```xml
<!-- Example behavior tree for navigation -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="NavigateWithRecovery">
      <GoalUpdated/>
      <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      <FollowPath path="{path}" controller_id="FollowPath"/>
    </Sequence>
  </BehaviorTree>
</root>
```

Common behavior tree nodes include:
- **Sequences**: Execute children in order until one fails
- **Fallbacks**: Try children in order until one succeeds
- **Decorators**: Modify behavior of child nodes
- **Conditions**: Check for specific conditions
- **Actions**: Execute specific navigation tasks

## Navigation for Humanoid Robots

Humanoid robots present unique navigation challenges compared to wheeled robots:

### Balance and Stability
- Center of mass considerations during movement
- Dynamic balance during walking
- Step planning for bipedal locomotion

### Multi-modal Navigation
- Walking on flat surfaces
- Climbing stairs
- Navigating uneven terrain
- Transitioning between different movement modes

### Human-aware Navigation
- Social navigation rules
- Maintaining appropriate distances from humans
- Yielding to human traffic patterns

## Recovery Behaviors

Navigation2 includes various recovery behaviors to handle common navigation failures:

- **Spin**: Rotate in place to clear local minima
- **Backup**: Move backward to escape obstacles
- **Wait**: Pause briefly before retrying navigation

## Performance Optimization

### Map Resolution
Balance map resolution between accuracy and computational cost:
- Higher resolution: Better obstacle detection, slower processing
- Lower resolution: Faster processing, potential for missed obstacles

### Costmap Configuration
Optimize costmap parameters for humanoid robot requirements:
- Robot radius: Account for full robot dimensions including limbs
- Inflation radius: Ensure safe passage around obstacles
- Update rates: Balance responsiveness with computational load

### Sensor Integration
Integrate multiple sensor types for robust navigation:
- LIDAR: Primary obstacle detection
- Cameras: Visual landmarks and dynamic obstacle detection
- IMU: Balance and orientation information
- Force/torque sensors: Ground contact detection

## Safety Considerations

Navigation in humanoid robots must prioritize safety:

- **Emergency stops**: Immediate halt on critical safety events
- **Speed limits**: Reduce speed in uncertain or crowded environments
- **Safe zones**: Maintain safety buffers around humans and obstacles
- **Fallback behaviors**: Safe recovery when navigation fails

## Integration with Control Systems

Navigation systems must integrate seamlessly with the robot's control architecture:

```python
# Example integration with humanoid control
class HumanoidNavigator:
    def __init__(self):
        # Navigation components
        self.navigation_client = NavigationClient()

        # Balance control
        self.balance_controller = BalanceController()

        # Step planner for bipedal locomotion
        self.step_planner = StepPlanner()

    def navigate_with_balance(self, goal_pose):
        # Plan path considering balance constraints
        balanced_path = self.plan_balanced_path(goal_pose)

        # Execute navigation with balance monitoring
        for step in balanced_path:
            self.balance_controller.update_target(step.balance_pose)
            self.step_planner.execute_step(step)

            # Monitor balance during execution
            if not self.balance_controller.is_stable():
                self.execute_recovery_behavior()
                return False

        return True
```

## Summary

Navigation in ROS 2 provides sophisticated capabilities for humanoid robots to move purposefully through complex environments. The Navigation2 stack offers a flexible architecture that can be adapted to the unique requirements of humanoid robots, including balance considerations, multi-modal locomotion, and human-aware navigation. Understanding these systems is essential for developing autonomous humanoid robots capable of operating in human environments.

## References

1. Navigation2 Documentation: https://navigation.ros.org/
2. Behavior Trees in ROS 2: https://github.com/BehaviorTree/BehaviorTree.CPP
3. ROS 2 Navigation Tutorials: https://navigation.ros.org/tutorials/

## Exercises

1. Set up a simple navigation system with a pre-built map
2. Configure a costmap for a humanoid robot's dimensions
3. Implement a custom recovery behavior for navigation failures
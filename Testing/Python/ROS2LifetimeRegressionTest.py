import gc
import traceback

import slicer


def run():
    scene = slicer.mrmlScene

    ros_node = scene.AddNewNodeByClass("vtkMRMLROS2NodeNode")
    if ros_node is None:
        raise RuntimeError("ROS2 node class is unavailable")
    ros_node.Create("lifetime_regression")

    subscriber = ros_node.CreateAndAddSubscriberNode("String", "/lifetime_subscriber")
    publisher = ros_node.CreateAndAddPublisherNode("String", "/lifetime_publisher")
    if subscriber is None or publisher is None:
        raise RuntimeError("Failed to create lifetime pub/sub nodes")
    if not ros_node.RemoveAndDeleteSubscriberNode("/lifetime_subscriber"):
        raise RuntimeError("Failed to remove lifetime subscriber")
    if not ros_node.RemoveAndDeletePublisherNode("/lifetime_publisher"):
        raise RuntimeError("Failed to remove lifetime publisher")
    subscriber = None
    publisher = None
    gc.collect()
    ros_node.Spin()

    parameter_node = ros_node.CreateAndAddParameterNode("/unavailable_parameter_server")
    if parameter_node is None:
        raise RuntimeError("Failed to create parameter node")

    # Removing a parameter node directly from the scene previously left a
    # dangling raw pointer in vtkMRMLROS2NodeNode. Its next Spin() crashed.
    scene.RemoveNode(parameter_node)
    parameter_node = None
    gc.collect()
    ros_node.Spin()

    # Loading a robot node without the default "ros2:node:slicer" node
    # previously dereferenced nullptr in UpdateScene().
    robot_node = scene.AddNewNodeByClass("vtkMRMLROS2RobotNode")
    if robot_node is None:
        raise RuntimeError("ROS2 robot node class is unavailable")
    robot_node.UpdateScene(scene)

    scene.RemoveNode(robot_node)
    scene.RemoveNode(ros_node)
    robot_node = None
    ros_node = None
    scene.Clear(0)
    gc.collect()


try:
    run()
except Exception:
    traceback.print_exc()
    slicer.app.exit(1)
else:
    print("ROS2_LIFETIME_REGRESSION_PASS")
    slicer.app.exit(0)

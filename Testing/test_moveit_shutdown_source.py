"""Source contract for safe MoveIt teardown and reconnect."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "MRML/vtkMRMLROS2RobotNode.h").read_text()
SOURCE = (ROOT / "MRML/vtkMRMLROS2RobotNode.cxx").read_text()
LOGIC_HEADER = (ROOT / "Logic/vtkSlicerROS2Logic.h").read_text()
LOGIC_SOURCE = (ROOT / "Logic/vtkSlicerROS2Logic.cxx").read_text()
MODULE_SOURCE = (ROOT / "qSlicerROS2Module.cxx").read_text()


def braced_body(source, opening):
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError("unterminated body")


def method_body(qualified_name, source=SOURCE):
    match = re.search(rf"\b{re.escape(qualified_name)}\s*\(", source)
    assert match, f"missing {qualified_name}"
    return braced_body(source, source.index("{", match.end()))


def reset_position(body, field):
    match = re.search(rf"{re.escape(field)}\s*(?:\.reset\s*\(\)|=\s*nullptr\s*;)", body)
    assert match, f"{field} is not cleared"
    return match.start()


def test_moveit_resources_are_released_before_node_teardown():
    assert len(re.findall(r"\bvoid\s+ReleaseMoveItResources\s*\(\s*\)\s*;", HEADER)) == 1
    assert len(re.findall(r"(?m)^void\s+vtkMRMLROS2RobotNode::ReleaseMoveItResources\s*\(\s*\)", SOURCE)) == 1

    destructor = method_body("vtkMRMLROS2RobotNode::~vtkMRMLROS2RobotNode")
    assert destructor.index("this->ReleaseMoveItResources()") < destructor.index("this->RemoveRobotVisualization()")

    remove = method_body("vtkMRMLROS2RobotNode::RemoveFromROS2Node")
    release = remove.index("this->ReleaseMoveItResources()")
    assert release < remove.index('this->SetNodeReferenceID("node", nullptr)')
    assert release < remove.index("mMRMLROS2Node = nullptr")

    teardown = method_body("vtkMRMLROS2RobotNode::ReleaseMoveItResources")
    monitor_reset = reset_position(teardown, "mInternals->PlanningSceneMonitorPtr")
    for monitor in ("stopStateMonitor", "stopSceneMonitor", "stopWorldGeometryMonitor"):
        stop = re.search(rf"mInternals->PlanningSceneMonitorPtr->\s*{monitor}\s*\(", teardown)
        assert stop and stop.start() < monitor_reset
    reset_position(teardown, "mInternals->JointModelGroupPtr")
    assert re.search(r"mInternals->IKGroupName\s*(?:\.clear\s*\(\)|=\s*(?:\"\"|\{\}))", teardown)
    assert reset_position(teardown, "mInternals->RobotModelPtr") < reset_position(teardown, "mInternals->RobotModelLoaderPtr")

    assert re.search(r"\bbool\s+SetupIKMoveIt\s*\(", HEADER)
    method_body("vtkMRMLROS2RobotNode::SetupIKMoveIt")


def test_logic_disconnects_robot_nodes_before_ros_shutdown():
    declaration = r"\bvoid\s+DisconnectRobots\s*\(\s*(?:void)?\s*\)\s*;"
    definition = r"(?m)^void\s+vtkSlicerROS2Logic::DisconnectRobots\s*\(\s*(?:void)?\s*\)"
    assert len(re.findall(declaration, LOGIC_HEADER)) == 1
    assert len(re.findall(definition, LOGIC_SOURCE)) == 1

    disconnect = method_body("vtkSlicerROS2Logic::DisconnectRobots", LOGIC_SOURCE)
    managed_nodes = re.search(r"for\s*\([^)]*:\s*mROS2Nodes\s*\)\s*\{", disconnect)
    assert managed_nodes
    node_body = braced_body(disconnect, managed_nodes.end() - 1)
    copied_names = re.search(
        r"(?:const\s+)?(?:auto|std::vector\s*<\s*std::string\s*>)\s+(\w+)\s*=\s*\w+->mRobotNames\s*;",
        node_body,
    )
    assert copied_names, "copy the managed node's robot names before deleting them"
    names = re.escape(copied_names.group(1))
    robot_loop = re.search(rf"for\s*\(([^)]*):\s*{names}\s*\)\s*\{{", node_body[copied_names.end():])
    assert robot_loop, "iterate over the copied robot names"
    robot_name = re.findall(r"[A-Za-z_]\w*", robot_loop.group(1))[-1]
    assert re.search(rf"->RemoveAndDeleteRobotNode\s*\(\s*{re.escape(robot_name)}\s*\)", node_body)

    destructor = method_body("vtkSlicerROS2Logic::~vtkSlicerROS2Logic", LOGIC_SOURCE)
    cleanup = (
        "this->DisconnectRobots()",
        "this->DisconnectSubscriptions()",
        "this->DisconnectPublishers()",
        "node->Destroy()",
        "vtkMRMLROS2::ROSShutdown()",
    )
    assert [destructor.index(call) for call in cleanup] == sorted(destructor.index(call) for call in cleanup)

    setup = method_body("qSlicerROS2Module::setup", MODULE_SOURCE)
    quit_handler = re.search(r"&QCoreApplication::aboutToQuit\s*,\s*this\s*,\s*\[this\]\s*\(\)\s*\{", setup)
    assert quit_handler
    quit_body = braced_body(setup, quit_handler.end() - 1)
    assert quit_body.index("rosLogic->DisconnectRobots()") < quit_body.index("rosLogic->DisconnectSubscriptions()")
    assert quit_body.index("rosLogic->DisconnectRobots()") < quit_body.index("rosLogic->DisconnectPublishers()")

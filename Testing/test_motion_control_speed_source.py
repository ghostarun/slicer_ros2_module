"""Source contract for the R2c speed changes (cannot compile on the host)."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CXX = (ROOT / "MRML/vtkMRMLROS2MotionControlNode.cxx").read_text()
HEADER = (ROOT / "MRML/vtkMRMLROS2MotionControlNode.h").read_text()
ROBOT = (ROOT / "MRML/vtkMRMLROS2RobotNode.cxx").read_text()
PY = (ROOT / "ROS2MotionControl/ROS2MotionControl.py").read_text()


def body(source, name):
    match = re.search(rf"\b{re.escape(name)}\s*\(", source)
    assert match, name
    start = source.index("{", match.end())
    depth = 0
    for index in range(start, len(source)):
        depth += {"{": 1, "}": -1}.get(source[index], 0)
        if depth == 0:
            return source[start:index]
    raise AssertionError(name)


def test_compute_ik_refresh_opt_out_defaults_to_refreshing():
    assert re.search(r"refreshPlanningScene=True,\s*\):", PY)
    start = PY.index("def computeIKWithMoveIt")
    ik = PY[start:PY.index("def computeIKWithKDL", start)]
    assert "if refreshPlanningScene and now -" in ik


def test_no_move_group_is_built_on_the_stack_in_the_main_thread_paths():
    for method in (
        "vtkMRMLROS2MotionControlNode::PlanMoveItTrajectory",
        "vtkMRMLROS2MotionControlNode::PlanMoveItTrajectoryFromState",
        "vtkMRMLROS2MotionControlNode::BeginMoveItTrajectoryFromState",
        "vtkMRMLROS2MotionControlNode::ComputeMoveItForwardKinematics",
        "vtkMRMLROS2MotionControlNode::ExecuteMoveItTrajectory",
    ):
        text = body(CXX, method)
        assert "AcquireMoveGroup" in text, method
        assert not re.search(r"MoveGroupInterface\s+\w+\s*\(", text), method
        assert not re.search(r"\bMoveGroup\s+\w+\s*\(", text), method
    # Planner id is reset on every request: empty selects the default.
    reset = body(CXX, "void ResetMoveGroupRequest")
    assert "setPlannerId(plannerId)" in reset and "setStartStateToCurrentState" in reset
    assert "if (!plannerId.empty())" not in CXX
    assert "JointPlanGroup" not in CXX


def test_fk_prefers_robot_node_model():
    fk = body(CXX, "vtkMRMLROS2MotionControlNode::ComputeMoveItForwardKinematics")
    assert fk.index("GetCachedMoveItRobotModel") < fk.index("AcquireMoveGroup")


def test_batch_apply_is_one_diff_with_cached_client():
    assert re.search(r"bool\s+ApplyMoveItCollisionObjects\s*\(\s*vtkCollection", HEADER)
    batch = body(CXX, "vtkMRMLROS2MotionControlNode::ApplyMoveItCollisionObjects")
    assert batch.count("async_send_request") == 1
    assert "scene.is_diff = true" in batch and "robot_state.is_diff = true" in batch
    assert "AcquireApplySceneClient" in batch and "create_client" not in batch
    assert "kSecondsPerPoint" in batch


def test_cached_interfaces_are_released_before_ros_shutdown():
    assert "ReleaseMoveItResources" in body(
        ROBOT, "vtkMRMLROS2RobotNode::RemoveFromROS2Node"
    )
    assert "ReleaseCaches" in body(CXX, "vtkMRMLROS2MotionControlNode::~vtkMRMLROS2MotionControlNode")

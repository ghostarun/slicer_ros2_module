#include <vtkMRMLROS2MotionControlNode.h>

#include <vtkMRMLROS2NodeNode.h>
#include <vtkMRMLROS2NodeInternals.h>

#include <vtkMoveitMsgsRobotTrajectory.h>
#include <vtkROS2ToSlicer.h>
#include <vtkSlicerToROS2.h>

#include <vtkCollection.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_cartesian_path.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>

#include <algorithm>
#include <chrono>
#include <future>
#include <sstream>
#include <thread>

namespace
{
const char* MoveItErrorCodeName(int code)
{
  // moveit_msgs/msg/MoveItErrorCodes values. Keep the numeric code in every
  // caller-facing message as the authoritative transport value.
  switch (code) {
    case 1: return "SUCCESS";
    case 99999: return "FAILURE";
    case -1: return "PLANNING_FAILED";
    case -2: return "INVALID_MOTION_PLAN";
    case -3: return "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE";
    case -4: return "CONTROL_FAILED";
    case -6: return "TIMED_OUT";
    case -7: return "PREEMPTED";
    case -10: return "START_STATE_IN_COLLISION";
    case -11: return "START_STATE_VIOLATES_PATH_CONSTRAINTS";
    case -12: return "GOAL_IN_COLLISION";
    case -13: return "GOAL_VIOLATES_PATH_CONSTRAINTS";
    case -14: return "GOAL_CONSTRAINTS_VIOLATED";
    case -15: return "INVALID_GROUP_NAME";
    case -16: return "INVALID_GOAL_CONSTRAINTS";
    case -17: return "INVALID_ROBOT_STATE";
    case -18: return "INVALID_LINK_NAME";
    case -19: return "INVALID_OBJECT_NAME";
    case -21: return "FRAME_TRANSFORM_FAILURE";
    case -22: return "COLLISION_CHECKING_UNAVAILABLE";
    case -23: return "ROBOT_STATE_STALE";
    case -24: return "SENSOR_INFO_STALE";
    case -31: return "NO_IK_SOLUTION";
    default: return "UNKNOWN_MOVEIT_ERROR";
  }
}
}

// ── Internals ────────────────────────────────────────────────────────────────

struct vtkMRMLROS2MotionControlNodeInternals
{
  moveit_msgs::msg::RobotTrajectory CachedTrajectory;
  double LastCartesianPathFraction = 0.0;
  std::string LastStateValidityMessage;
  std::string LastForwardKinematicsMessage;
  std::string LastJointPlanMessage;
};

// ── vtkStandardNewMacro ──────────────────────────────────────────────────────

vtkStandardNewMacro(vtkMRMLROS2MotionControlNode);

// ── Constructor / destructor ─────────────────────────────────────────────────

vtkMRMLROS2MotionControlNode::vtkMRMLROS2MotionControlNode()
  : mInternals(std::make_unique<vtkMRMLROS2MotionControlNodeInternals>())
{}

vtkMRMLROS2MotionControlNode::~vtkMRMLROS2MotionControlNode() = default;

// ── vtkMRMLNode boilerplate ──────────────────────────────────────────────────

vtkMRMLNode * vtkMRMLROS2MotionControlNode::CreateNodeInstance(void)
{
  return SelfType::New();
}

const char * vtkMRMLROS2MotionControlNode::GetNodeTagName(void)
{
  return "ROS2MotionControl";
}

void vtkMRMLROS2MotionControlNode::PrintSelf(std::ostream & os, vtkIndent indent)
{
  Superclass::PrintSelf(os, indent);
}

void vtkMRMLROS2MotionControlNode::ReadXMLAttributes(const char** atts)
{
  Superclass::ReadXMLAttributes(atts);
}

void vtkMRMLROS2MotionControlNode::WriteXML(std::ostream & of, int indent)
{
  Superclass::WriteXML(of, indent);
}

// ── Private helper ───────────────────────────────────────────────────────────

std::shared_ptr<rclcpp::Node> vtkMRMLROS2MotionControlNode::GetROSNodePointer()
{
  auto * nodeRef = vtkMRMLROS2NodeNode::SafeDownCast(GetNodeReference("node"));
  if (!nodeRef || !nodeRef->mInternals || !nodeRef->mInternals->mNodePointer) {
    vtkErrorMacro(<< "GetROSNodePointer: \"node\" reference is not set or not initialized");
    return nullptr;
  }
  return nodeRef->mInternals->mNodePointer;
}

// ── Planning / execution ─────────────────────────────────────────────────────

vtkMoveitMsgsRobotTrajectory* vtkMRMLROS2MotionControlNode::PlanMoveItTrajectory(
    const std::string & groupName,
    const std::vector<double> & goalJointValues,
    double velocityScaling,
    double accelerationScaling,
    double planningTimeSec)
{
  mInternals->LastJointPlanMessage.clear();
  vtkMoveitMsgsRobotTrajectory* traj = vtkMoveitMsgsRobotTrajectory::New();

  auto node = GetROSNodePointer();
  if (!node) { return traj; }

  if (groupName.empty()) {
    vtkErrorMacro(<< "PlanMoveItTrajectory: groupName is empty");
    return traj;
  }

  moveit::planning_interface::MoveGroupInterface moveGroup(node, groupName);

  const auto jointNames = moveGroup.getJointNames();
  if (jointNames.size() != goalJointValues.size()) {
    vtkErrorMacro(<< "PlanMoveItTrajectory: expected " << jointNames.size()
                  << " joint values for group '" << groupName
                  << "' but got " << goalJointValues.size());
    return traj;
  }

  const double velScale = std::clamp(velocityScaling,     0.0, 1.0);
  const double accScale = std::clamp(accelerationScaling, 0.0, 1.0);
  moveGroup.setMaxVelocityScalingFactor(velScale);
  moveGroup.setMaxAccelerationScalingFactor(accScale);
  moveGroup.setPlanningTime(planningTimeSec > 0.0 ? planningTimeSec : 5.0);
  moveGroup.setStartStateToCurrentState();

  std::map<std::string, double> targets;
  for (size_t i = 0; i < jointNames.size(); ++i) {
    targets[jointNames[i]] = goalJointValues[i];
  }
  moveGroup.setJointValueTarget(targets);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto result = moveGroup.plan(plan);
  if (result == moveit::core::MoveItErrorCode::SUCCESS) {
    // Cache the ROS trajectory for ExecuteCachedMoveItTrajectory
    mInternals->CachedTrajectory = plan.trajectory;
    vtkROS2ToSlicer(plan.trajectory, vtkSmartPointer<vtkMoveitMsgsRobotTrajectory>(traj));
    mInternals->LastJointPlanMessage = "MoveIt joint planning succeeded";
  } else {
    mInternals->LastJointPlanMessage =
        "MoveIt joint planning failed with MoveItErrorCode=" +
        std::to_string(result.val);
    vtkErrorMacro(<< "PlanMoveItTrajectory: planning failed for group '" << groupName
                  << "' with MoveItErrorCode=" << result.val);
  }

  return traj;
}

vtkMoveitMsgsRobotTrajectory* vtkMRMLROS2MotionControlNode::PlanMoveItTrajectoryFromState(
    const std::string & groupName,
    const std::vector<std::string> & startJointNames,
    const std::vector<double> & startJointValues,
    const std::vector<double> & goalJointValues,
    double velocityScaling,
    double accelerationScaling,
    double planningTimeSec)
{
  mInternals->LastJointPlanMessage.clear();
  vtkMoveitMsgsRobotTrajectory* traj = vtkMoveitMsgsRobotTrajectory::New();

  auto node = GetROSNodePointer();
  if (!node) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: ROS node is unavailable";
    return traj;
  }

  if (groupName.empty()) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: planning group is empty";
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: groupName is empty");
    return traj;
  }
  if (startJointNames.empty() || startJointNames.size() != startJointValues.size()) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: start joint names/values are empty or mismatched";
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: startJointNames has "
                  << startJointNames.size() << " entries but startJointValues has "
                  << startJointValues.size());
    return traj;
  }

  moveit::planning_interface::MoveGroupInterface moveGroup(node, groupName);
  const auto groupJointNames = moveGroup.getJointNames();
  if (groupJointNames.size() != goalJointValues.size()) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: goal vector does not match the planning group";
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: expected "
                  << groupJointNames.size() << " goal joint values for group '"
                  << groupName << "' but got " << goalJointValues.size());
    return traj;
  }

  std::map<std::string, double> submittedStart;
  for (size_t i = 0; i < startJointNames.size(); ++i) {
    submittedStart[startJointNames[i]] = startJointValues[i];
  }
  for (const auto & jointName : groupJointNames) {
    if (submittedStart.find(jointName) == submittedStart.end()) {
      mInternals->LastJointPlanMessage =
          "Explicit-start planning failed: submitted start omitted group joint '" +
          jointName + "'";
      vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: explicit start state omitted group joint '"
                    << jointName << "'");
      return traj;
    }
  }

  auto robotModel = moveGroup.getRobotModel();
  if (!robotModel) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: MoveIt robot model is unavailable";
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: MoveIt robot model is unavailable");
    return traj;
  }
  moveit::core::RobotState startState(robotModel);
  startState.setToDefaultValues();
  for (const auto & jointName : groupJointNames) {
    startState.setVariablePosition(jointName, submittedStart[jointName]);
  }
  startState.update();
  const auto * jointModelGroup = startState.getJointModelGroup(groupName);
  if (!jointModelGroup || !startState.satisfiesBounds(jointModelGroup)) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: submitted start violates group bounds";
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: explicit start state violates group bounds");
    return traj;
  }

  const double velScale = std::clamp(velocityScaling,     0.0, 1.0);
  const double accScale = std::clamp(accelerationScaling, 0.0, 1.0);
  moveGroup.setMaxVelocityScalingFactor(velScale);
  moveGroup.setMaxAccelerationScalingFactor(accScale);
  moveGroup.setPlanningTime(planningTimeSec > 0.0 ? planningTimeSec : 5.0);
  moveGroup.setStartState(startState);

  std::map<std::string, double> targets;
  for (size_t i = 0; i < groupJointNames.size(); ++i) {
    targets[groupJointNames[i]] = goalJointValues[i];
  }
  if (!moveGroup.setJointValueTarget(targets)) {
    mInternals->LastJointPlanMessage =
        "Explicit-start planning failed: submitted goal violates the planning group's joint constraints";
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: goal target was rejected for group '"
                  << groupName << "'");
    return traj;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto result = moveGroup.plan(plan);
  if (result == moveit::core::MoveItErrorCode::SUCCESS) {
    mInternals->CachedTrajectory = plan.trajectory;
    vtkROS2ToSlicer(plan.trajectory, vtkSmartPointer<vtkMoveitMsgsRobotTrajectory>(traj));
    mInternals->LastJointPlanMessage =
        "MoveIt explicit-start joint planning succeeded";
  } else {
    mInternals->LastJointPlanMessage =
        "MoveIt explicit-start joint planning failed: code=" +
        std::to_string(result.val) + " (" + MoveItErrorCodeName(result.val) +
        "), group='" + groupName + "', planningTimeSec=" +
        std::to_string(planningTimeSec > 0.0 ? planningTimeSec : 5.0);
    vtkErrorMacro(<< "PlanMoveItTrajectoryFromState: planning failed for group '"
                  << groupName << "' with MoveItErrorCode=" << result.val);
  }

  return traj;
}

std::string vtkMRMLROS2MotionControlNode::GetLastJointPlanMessage() const
{
  return mInternals->LastJointPlanMessage;
}

bool vtkMRMLROS2MotionControlNode::CheckMoveItStateValidity(
    const std::string & groupName,
    const std::vector<std::string> & jointNames,
    const std::vector<double> & jointValues,
    double timeoutSec)
{
  mInternals->LastStateValidityMessage.clear();
  auto node = GetROSNodePointer();
  if (!node) {
    mInternals->LastStateValidityMessage = "ROS node is unavailable";
    return false;
  }
  if (groupName.empty()) {
    mInternals->LastStateValidityMessage = "planning group is empty";
    return false;
  }
  if (jointNames.empty() || jointNames.size() != jointValues.size()) {
    mInternals->LastStateValidityMessage = "joint names/values are empty or mismatched";
    return false;
  }

  auto request = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
  request->group_name = groupName;
  request->robot_state.joint_state.name = jointNames;
  request->robot_state.joint_state.position = jointValues;
  request->robot_state.is_diff = false;

  auto client = node->create_client<moveit_msgs::srv::GetStateValidity>(
      "/check_state_validity");
  const double boundedTimeoutSec = timeoutSec > 0.0 ? timeoutSec : 2.0;
  const auto timeout = std::chrono::duration<double>(boundedTimeoutSec);
  if (!client->wait_for_service(timeout)) {
    mInternals->LastStateValidityMessage =
        "/check_state_validity is unavailable after the bounded timeout";
    return false;
  }

  auto future = client->async_send_request(request);
  const auto startTime = std::chrono::steady_clock::now();
  while (future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
    rclcpp::spin_some(node);
    if (std::chrono::steady_clock::now() - startTime > timeout) {
      mInternals->LastStateValidityMessage =
          "/check_state_validity request timed out";
      return false;
    }
  }
  auto response = future.get();
  if (!response) {
    mInternals->LastStateValidityMessage =
        "/check_state_validity returned a null response";
    return false;
  }
  if (response->valid) {
    mInternals->LastStateValidityMessage =
        "MoveIt accepted the explicit static joint state";
    return true;
  }

  std::ostringstream reason;
  reason << "MoveIt rejected the explicit static joint state";
  if (!response->contacts.empty()) {
    reason << "; contacts=";
    const size_t limit = std::min<size_t>(response->contacts.size(), 4);
    for (size_t i = 0; i < limit; ++i) {
      if (i > 0) { reason << ", "; }
      reason << response->contacts[i].contact_body_1
             << "<->" << response->contacts[i].contact_body_2;
    }
    if (response->contacts.size() > limit) {
      reason << " (and " << response->contacts.size() - limit << " more)";
    }
  }
  mInternals->LastStateValidityMessage = reason.str();
  return false;
}

std::string vtkMRMLROS2MotionControlNode::GetLastStateValidityMessage() const
{
  return mInternals->LastStateValidityMessage;
}

vtkMatrix4x4* vtkMRMLROS2MotionControlNode::ComputeMoveItForwardKinematics(
    const std::string & groupName,
    const std::vector<std::string> & jointNames,
    const std::vector<double> & jointValues,
    const std::string & linkName,
    double timeoutSec)
{
  mInternals->LastForwardKinematicsMessage.clear();
  vtkMatrix4x4* result = vtkMatrix4x4::New();
  result->Identity();
  auto node = GetROSNodePointer();
  if (!node) {
    mInternals->LastForwardKinematicsMessage = "ROS node is unavailable";
    return result;
  }
  if (groupName.empty() || linkName.empty()) {
    mInternals->LastForwardKinematicsMessage = "planning group or link is empty";
    vtkErrorMacro(<< "ComputeMoveItForwardKinematics: groupName/linkName is empty");
    return result;
  }
  if (jointNames.empty() || jointNames.size() != jointValues.size()) {
    mInternals->LastForwardKinematicsMessage = "joint names/values are empty or mismatched";
    vtkErrorMacro(<< "ComputeMoveItForwardKinematics: joint names/values are empty or mismatched");
    return result;
  }

  moveit::planning_interface::MoveGroupInterface moveGroup(node, groupName);
  static_cast<void>(timeoutSec);
  auto robotModel = moveGroup.getRobotModel();
  if (!robotModel) {
    mInternals->LastForwardKinematicsMessage = "MoveIt robot model is unavailable";
    vtkErrorMacro(<< "ComputeMoveItForwardKinematics: MoveIt robot model is unavailable");
    return result;
  }
  moveit::core::RobotState state(robotModel);
  state.setToDefaultValues();
  const auto & modelVariableNames = state.getVariableNames();
  for (size_t i = 0; i < jointNames.size(); ++i) {
    if (std::find(modelVariableNames.begin(), modelVariableNames.end(),
                  jointNames[i]) == modelVariableNames.end()) {
      mInternals->LastForwardKinematicsMessage =
          "explicit state contains an unknown joint variable";
      vtkErrorMacro(<< "ComputeMoveItForwardKinematics: unknown joint variable '"
                    << jointNames[i] << "'");
      return result;
    }
    state.setVariablePosition(jointNames[i], jointValues[i]);
  }
  state.update();
  const auto * jointModelGroup = state.getJointModelGroup(groupName);
  if (!jointModelGroup || !state.satisfiesBounds(jointModelGroup)) {
    mInternals->LastForwardKinematicsMessage = "explicit state violates group bounds";
    vtkErrorMacro(<< "ComputeMoveItForwardKinematics: explicit state violates group bounds");
    return result;
  }
  if (!state.knowsFrameTransform(linkName)) {
    mInternals->LastForwardKinematicsMessage = "requested TCP link is unknown";
    vtkErrorMacro(<< "ComputeMoveItForwardKinematics: unknown link '" << linkName << "'");
    return result;
  }

  const Eigen::Isometry3d & linkTransform = state.getGlobalLinkTransform(linkName);
  const Eigen::Quaterniond orientation(linkTransform.rotation());
  geometry_msgs::msg::Pose pose;
  pose.position.x = linkTransform.translation().x();
  pose.position.y = linkTransform.translation().y();
  pose.position.z = linkTransform.translation().z();
  pose.orientation.x = orientation.x();
  pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z();
  pose.orientation.w = orientation.w();
  vtkROS2ToSlicer(pose, vtkSmartPointer<vtkMatrix4x4>(result));
  mInternals->LastForwardKinematicsMessage =
      "MoveIt FK returned the explicit-state TCP pose";
  return result;
}

std::string vtkMRMLROS2MotionControlNode::GetLastForwardKinematicsMessage() const
{
  return mInternals->LastForwardKinematicsMessage;
}

vtkMoveitMsgsRobotTrajectory* vtkMRMLROS2MotionControlNode::PlanMoveItCartesianTrajectory(
    const std::string & groupName,
    vtkCollection* targetPoses,
    const std::vector<std::string> & startJointNames,
    const std::vector<double> & startJointValues,
    double eefStepMeters,
    double jumpThreshold,
    bool avoidCollisions,
    double velocityScaling,
    double accelerationScaling,
    double planningTimeSec,
    const std::string & linkName)
{
  vtkMoveitMsgsRobotTrajectory* traj = vtkMoveitMsgsRobotTrajectory::New();
  mInternals->LastCartesianPathFraction = 0.0;

  auto node = GetROSNodePointer();
  if (!node) { return traj; }

  if (groupName.empty()) {
    vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: groupName is empty");
    return traj;
  }

  if (!targetPoses || targetPoses->GetNumberOfItems() == 0) {
    vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: no target poses provided");
    return traj;
  }

  const double velScale = std::clamp(velocityScaling,     0.0, 1.0);
  const double accScale = std::clamp(accelerationScaling, 0.0, 1.0);
  if (!startJointNames.empty() || !startJointValues.empty()) {
    if (startJointNames.size() != startJointValues.size()) {
      vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: startJointNames has "
                    << startJointNames.size() << " entries but startJointValues has "
                    << startJointValues.size());
      return traj;
    }
  }

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.reserve(targetPoses->GetNumberOfItems());
  for (int i = 0; i < targetPoses->GetNumberOfItems(); ++i) {
    vtkMatrix4x4* targetPose = vtkMatrix4x4::SafeDownCast(targetPoses->GetItemAsObject(i));
    if (!targetPose) {
      vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: item " << i
                    << " is not a vtkMatrix4x4");
      return traj;
    }

    geometry_msgs::msg::Pose pose;
    vtkSlicerToROS2(targetPose, pose, node);
    waypoints.push_back(pose);
  }

  auto request = std::make_shared<moveit_msgs::srv::GetCartesianPath::Request>();
  request->group_name = groupName;
  request->waypoints = waypoints;
  request->max_step = eefStepMeters > 0.0 ? eefStepMeters : 0.01;
  request->jump_threshold = jumpThreshold > 0.0 ? jumpThreshold : 0.0;
  request->avoid_collisions = avoidCollisions;
  request->max_velocity_scaling_factor = velScale > 0.0 ? velScale : 1.0;
  request->max_acceleration_scaling_factor = accScale > 0.0 ? accScale : 1.0;
  if (!linkName.empty()) {
    request->link_name = linkName;
  }

  if (!startJointValues.empty()) {
    request->start_state.joint_state.name = startJointNames;
    request->start_state.joint_state.position = startJointValues;
    request->start_state.is_diff = false;
  } else {
    request->start_state.is_diff = true;
  }

  auto client = node->create_client<moveit_msgs::srv::GetCartesianPath>("/compute_cartesian_path");
  const double timeoutSec = planningTimeSec > 0.0 ? planningTimeSec : 5.0;
  const auto timeout = std::chrono::duration<double>(timeoutSec);
  if (!client->wait_for_service(timeout)) {
    vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: /compute_cartesian_path service "
                  << "is not available after " << timeoutSec << " seconds");
    return traj;
  }

  auto future = client->async_send_request(request);
  const auto startTime = std::chrono::steady_clock::now();
  while (future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
    rclcpp::spin_some(node);
    if (std::chrono::steady_clock::now() - startTime > timeout) {
      vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: /compute_cartesian_path "
                    << "request timed out after " << timeoutSec << " seconds");
      return traj;
    }
  }

  auto response = future.get();
  if (!response) {
    vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: /compute_cartesian_path "
                  << "returned a null response");
    return traj;
  }

  auto setFallbackTiming = [](moveit_msgs::msg::RobotTrajectory& trajectory) {
    constexpr double stepSec = 0.1;
    auto& points = trajectory.joint_trajectory.points;
    for (size_t i = 0; i < points.size(); ++i) {
      const double t = static_cast<double>(i) * stepSec;
      const int32_t sec = static_cast<int32_t>(t);
      points[i].time_from_start.sec = sec;
      points[i].time_from_start.nanosec = static_cast<uint32_t>((t - sec) * 1e9);
    }
  };

  mInternals->LastCartesianPathFraction = response->fraction;
  moveit_msgs::msg::RobotTrajectory rosTrajectory = response->solution;
  if (response->fraction <= 0.0 || rosTrajectory.joint_trajectory.points.empty()) {
    vtkErrorMacro(<< "PlanMoveItCartesianTrajectory: Cartesian planning failed for group '"
                  << groupName << "' with fraction=" << response->fraction
                  << " and MoveItErrorCode=" << response->error_code.val);
    return traj;
  }

  const auto& points = rosTrajectory.joint_trajectory.points;
  const bool allTimesZero = std::all_of(
    points.begin(), points.end(),
    [](const auto& point) {
      return point.time_from_start.sec == 0 && point.time_from_start.nanosec == 0;
    });
  if (allTimesZero && points.size() > 1) {
    setFallbackTiming(rosTrajectory);
  }

  mInternals->CachedTrajectory = rosTrajectory;
  vtkROS2ToSlicer(rosTrajectory, vtkSmartPointer<vtkMoveitMsgsRobotTrajectory>(traj));
  return traj;
}

double vtkMRMLROS2MotionControlNode::GetLastCartesianPathFraction() const
{
  return mInternals->LastCartesianPathFraction;
}

bool vtkMRMLROS2MotionControlNode::ExecuteMoveItTrajectory(
    const std::string & groupName,
    vtkMoveitMsgsRobotTrajectory* trajectory)
{
  auto node = GetROSNodePointer();
  if (!node) { return false; }

  if (groupName.empty()) {
    vtkErrorMacro(<< "ExecuteMoveItTrajectory: groupName is empty");
    return false;
  }
  if (!trajectory) {
    vtkErrorMacro(<< "ExecuteMoveItTrajectory: trajectory is null");
    return false;
  }

  try {
    moveit_msgs::msg::RobotTrajectory ros_traj;
    vtkSlicerToROS2(trajectory, ros_traj, node);

    if (ros_traj.joint_trajectory.points.empty()) {
      vtkErrorMacro(<< "ExecuteMoveItTrajectory: trajectory is empty");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface moveGroup(node, groupName);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory = ros_traj;

    auto result = moveGroup.execute(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      vtkInfoMacro(<< "ExecuteMoveItTrajectory: success for group '" << groupName << "'");
      return true;
    }
    vtkErrorMacro(<< "ExecuteMoveItTrajectory: failed for group '" << groupName
                  << "' with MoveItErrorCode=" << result.val);
    return false;
  }
  catch (const std::exception & e) {
    vtkErrorMacro(<< "ExecuteMoveItTrajectory: exception - " << e.what());
    return false;
  }
}

bool vtkMRMLROS2MotionControlNode::ExecuteCachedMoveItTrajectory(const std::string & groupName)
{
  if (mInternals->CachedTrajectory.joint_trajectory.points.empty()) {
    vtkErrorMacro(<< "ExecuteCachedMoveItTrajectory: no cached trajectory. "
                     "Call PlanMoveItTrajectory first.");
    return false;
  }

  vtkSmartPointer<vtkMoveitMsgsRobotTrajectory> vtk_traj =
    vtkSmartPointer<vtkMoveitMsgsRobotTrajectory>::New();
  vtkROS2ToSlicer(mInternals->CachedTrajectory, vtk_traj);
  return ExecuteMoveItTrajectoryAsync(groupName, vtk_traj.Get());
}

bool vtkMRMLROS2MotionControlNode::PlanAndExecuteMoveItTrajectory(
    const std::string & groupName,
    const std::vector<double> & goalJointValues,
    double velocityScaling,
    double accelerationScaling,
    double planningTimeSec)
{
  auto * trajectory = PlanMoveItTrajectory(groupName, goalJointValues,
                                            velocityScaling, accelerationScaling,
                                            planningTimeSec);
  auto node = GetROSNodePointer();
  if (!node) { trajectory->Delete(); return false; }

  moveit_msgs::msg::RobotTrajectory ros_traj;
  vtkSlicerToROS2(trajectory, ros_traj, node);

  if (ros_traj.joint_trajectory.points.empty()) {
    vtkErrorMacro(<< "PlanAndExecuteMoveItTrajectory: planning failed, cannot execute");
    trajectory->Delete();
    return false;
  }

  bool result = ExecuteMoveItTrajectory(groupName, trajectory);
  trajectory->Delete();
  return result;
}

bool vtkMRMLROS2MotionControlNode::ExecuteMoveItTrajectoryAsync(
    const std::string & groupName,
    vtkMoveitMsgsRobotTrajectory* trajectory)
{
  auto node = GetROSNodePointer();
  if (!node) { return false; }

  if (!trajectory) {
    vtkErrorMacro(<< "ExecuteMoveItTrajectoryAsync: trajectory is null");
    return false;
  }

  moveit_msgs::msg::RobotTrajectory ros_traj;
  vtkSlicerToROS2(trajectory, ros_traj, node);

  if (ros_traj.joint_trajectory.points.empty()) {
    vtkErrorMacro(<< "ExecuteMoveItTrajectoryAsync: trajectory is empty");
    return false;
  }

  std::thread([node, groupName, ros_traj]() {
    try {
      moveit::planning_interface::MoveGroupInterface moveGroup(node, groupName);
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory = ros_traj;
      moveGroup.execute(plan);
    }
    catch (const std::exception & e) {
      // Cannot call vtkErrorMacro from a detached thread safely
      (void)e;
    }
  }).detach();

  return true;
}

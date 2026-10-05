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
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/srv/get_cartesian_path.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <vtkMRMLModelNode.h>
#include <moveit_msgs/msg/planning_scene_components.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <map>
#include <stdexcept>
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

namespace
{
using JointPlanAction = moveit_msgs::action::MoveGroup;
using JointPlanHandle = rclcpp_action::ClientGoalHandle<JointPlanAction>;
using MoveGroup = moveit::planning_interface::MoveGroupInterface;

// Callbacks own no MRML objects, MoveGroupInterface, or VTK state.
struct JointPlanReply
{
  std::mutex Mutex;
  std::string Status = "pending";
  JointPlanHandle::SharedPtr Handle;
  std::shared_ptr<JointPlanAction::Result> Result;
  std::string Message;
};
std::atomic<unsigned long long> JointPlanSequence{0};

void ConfigureExplicitJointPlan(MoveGroup& group,
    const std::string& groupName,
    const std::vector<std::string>& startNames,
    const std::vector<double>& startValues,
    const std::vector<double>& goalValues,
    double velocity, double acceleration, double planningTime,
    const std::string& plannerId)
{
  if (!std::isfinite(velocity) || !std::isfinite(acceleration) || !std::isfinite(planningTime)) {
    throw std::invalid_argument("Planning parameters must be finite");
  }
  if (startNames.empty() || startNames.size() != startValues.size()) {
    throw std::invalid_argument("Explicit start names/values are empty or mismatched");
  }
  if (!plannerId.empty()) { group.setPlannerId(plannerId); }
  const auto names = group.getJointNames();
  if (names.size() != goalValues.size()) {
    throw std::invalid_argument("Goal vector does not match the planning group");
  }
  std::map<std::string, double> start;
  for (size_t i = 0; i < startNames.size(); ++i) {
    if (!std::isfinite(startValues[i]) || !start.emplace(startNames[i], startValues[i]).second) {
      throw std::invalid_argument("Explicit start has nonfinite or duplicate values");
    }
  }
  auto model = group.getRobotModel();
  if (!model) { throw std::runtime_error("MoveIt robot model is unavailable"); }
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  std::map<std::string, double> targets;
  for (size_t i = 0; i < names.size(); ++i) {
    if (!start.count(names[i]) || !std::isfinite(goalValues[i])) {
      throw std::invalid_argument("Explicit state omitted a group joint or contains a nonfinite goal");
    }
    state.setVariablePosition(names[i], start.at(names[i]));
    targets[names[i]] = goalValues[i];
  }
  state.update();
  const auto* jointGroup = state.getJointModelGroup(groupName);
  if (!jointGroup || !state.satisfiesBounds(jointGroup)) {
    throw std::invalid_argument("Explicit start violates group bounds");
  }
  group.setMaxVelocityScalingFactor(std::clamp(velocity, 0.0, 1.0));
  group.setMaxAccelerationScalingFactor(std::clamp(acceleration, 0.0, 1.0));
  group.setPlanningTime(planningTime > 0.0 ? planningTime : 5.0);
  group.setStartState(state);
  if (!group.setJointValueTarget(targets)) {
    throw std::invalid_argument("Explicit goal violates group constraints");
  }
}
}

// ── Internals ────────────────────────────────────────────────────────────────

struct vtkMRMLROS2MotionControlNodeInternals
{
  moveit_msgs::msg::RobotTrajectory CachedTrajectory;
  double LastCartesianPathFraction = 0.0;
  std::string LastStateValidityMessage;
  std::string LastPlanningSceneMessage;
  std::string LastForwardKinematicsMessage;
  std::string LastJointPlanMessage;
  std::string LastJointPlannerId;
  std::string JointPlanToken;
  std::weak_ptr<rclcpp::Node> JointPlanNode;
  std::shared_ptr<JointPlanReply> JointPlan;
  // Destroyed only by the main-thread API, never by a result callback.
  std::unique_ptr<MoveGroup> JointPlanGroup;
};

// ── vtkStandardNewMacro ──────────────────────────────────────────────────────

vtkStandardNewMacro(vtkMRMLROS2MotionControlNode);

// ── Constructor / destructor ─────────────────────────────────────────────────

vtkMRMLROS2MotionControlNode::vtkMRMLROS2MotionControlNode()
  : mInternals(std::make_unique<vtkMRMLROS2MotionControlNodeInternals>())
{}

vtkMRMLROS2MotionControlNode::~vtkMRMLROS2MotionControlNode()
{
  try { CancelJointPlan(mInternals->JointPlanToken); } catch (...) {}
}

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
    double planningTimeSec,
    const std::string & plannerId)
{
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or cancellation is unresolved";
    return vtkMoveitMsgsRobotTrajectory::New();
  }

  mInternals->LastJointPlanMessage.clear();
  mInternals->LastJointPlannerId.clear();
  vtkMoveitMsgsRobotTrajectory* traj = vtkMoveitMsgsRobotTrajectory::New();

  auto node = GetROSNodePointer();
  if (!node) { return traj; }

  if (groupName.empty()) {
    vtkErrorMacro(<< "PlanMoveItTrajectory: groupName is empty");
    return traj;
  }

  moveit::planning_interface::MoveGroupInterface moveGroup(node, groupName);
  if (!plannerId.empty()) {
    moveGroup.setPlannerId(plannerId);
  }
  mInternals->LastJointPlannerId = moveGroup.getPlannerId();

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
    double planningTimeSec,
    const std::string & plannerId)
{
  auto* trajectory = vtkMoveitMsgsRobotTrajectory::New();
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "An asynchronous request is pending or cancellation is unresolved";
    return trajectory;
  }
  mInternals->LastJointPlanMessage.clear();
  mInternals->LastJointPlannerId.clear();
  auto node = GetROSNodePointer();
  if (!node || groupName.empty()) {
    mInternals->LastJointPlanMessage = "Explicit-start planning requires a ROS node and planning group";
    return trajectory;
  }
  try {
    MoveGroup group(node, groupName);
    ConfigureExplicitJointPlan(group, groupName, startJointNames, startJointValues,
        goalJointValues, velocityScaling, accelerationScaling, planningTimeSec, plannerId);
    mInternals->LastJointPlannerId = group.getPlannerId();
    MoveGroup::Plan plan;
    const auto result = group.plan(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      mInternals->CachedTrajectory = plan.trajectory;
      vtkROS2ToSlicer(plan.trajectory, vtkSmartPointer<vtkMoveitMsgsRobotTrajectory>(trajectory));
      mInternals->LastJointPlanMessage = "MoveIt explicit-start joint planning succeeded";
    } else {
      mInternals->LastJointPlanMessage = "MoveIt explicit-start joint planning failed: code=" +
          std::to_string(result.val) + " (" + MoveItErrorCodeName(result.val) + ")";
    }
  } catch (const std::exception& error) {
    mInternals->LastJointPlanMessage = error.what();
  }
  return trajectory;
}

std::string vtkMRMLROS2MotionControlNode::BeginMoveItTrajectoryFromState(
    const std::string& groupName,
    const std::vector<std::string>& startJointNames,
    const std::vector<double>& startJointValues,
    const std::vector<double>& goalJointValues,
    double velocityScaling, double accelerationScaling,
    double planningTimeSec, const std::string& plannerId)
{
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or backend cancellation is unconfirmed; recreate the motion-control node before another request";
    return "";
  }
  mInternals->CachedTrajectory = {};
  mInternals->LastJointPlanMessage.clear();
  mInternals->LastJointPlannerId.clear();
  auto node = GetROSNodePointer();
  if (!node || groupName.empty()) {
    mInternals->LastJointPlanMessage = "Explicit-start planning requires a ROS node and planning group";
    return "";
  }
  try {
    auto group = std::make_unique<MoveGroup>(node, groupName,
        std::shared_ptr<tf2_ros::Buffer>(), rclcpp::Duration::from_seconds(1.0));
    ConfigureExplicitJointPlan(*group, groupName, startJointNames, startJointValues,
        goalJointValues, velocityScaling, accelerationScaling, planningTimeSec, plannerId);
    auto& client = group->getMoveGroupClient();
    if (!client.action_server_is_ready()) {
      throw std::runtime_error("MoveGroup action server is unavailable");
    }
    JointPlanAction::Goal goal;
    group->constructMotionPlanRequest(goal.request);
    goal.planning_options.plan_only = true;
    goal.planning_options.look_around = false;
    goal.planning_options.replan = false;
    goal.planning_options.planning_scene_diff.is_diff = true;
    goal.planning_options.planning_scene_diff.robot_state.is_diff = true;
    auto reply = std::make_shared<JointPlanReply>();
    std::weak_ptr<JointPlanReply> weakReply = reply;
    rclcpp_action::Client<JointPlanAction>::SendGoalOptions options;
    options.goal_response_callback = [weakReply](JointPlanHandle::SharedPtr handle) {
      if (auto reply = weakReply.lock()) {
        std::lock_guard<std::mutex> lock(reply->Mutex);
        reply->Handle = handle;
        if (!handle && reply->Status == "pending") {
          reply->Status = "error";
          reply->Message = "MoveGroup rejected the plan-only request";
        }
      }
    };
    options.result_callback = [weakReply](const JointPlanHandle::WrappedResult& result) {
      if (auto reply = weakReply.lock()) {
        std::lock_guard<std::mutex> lock(reply->Mutex);
        if (reply->Status != "pending") { return; }
        reply->Result = result.result;
        const bool success = result.code == rclcpp_action::ResultCode::SUCCEEDED &&
            result.result && result.result->error_code.val == 1;
        reply->Status = success ? "ready" : "error";
        reply->Message = result.result ?
            "MoveIt explicit-start joint planning code=" + std::to_string(result.result->error_code.val) +
                " (" + MoveItErrorCodeName(result.result->error_code.val) + ")" :
            "MoveGroup returned no result";
      }
    };
    mInternals->LastJointPlannerId = group->getPlannerId();
    mInternals->JointPlanToken = "joint-plan-" + std::to_string(++JointPlanSequence);
    mInternals->JointPlanNode = node;
    mInternals->JointPlan = reply;
    mInternals->JointPlanGroup = std::move(group);
    client.async_send_goal(goal, options);
    return mInternals->JointPlanToken;
  } catch (const std::exception& error) {
    mInternals->LastJointPlanMessage = error.what();
    if (mInternals->JointPlan) {
      CancelJointPlan(mInternals->JointPlanToken);
      mInternals->LastJointPlanMessage = std::string(error.what()) + "; " + mInternals->LastJointPlanMessage;
    }
    return "";
  }
}

std::string vtkMRMLROS2MotionControlNode::GetJointPlanStatus(const std::string& token)
{
  if (!mInternals->JointPlan || token != mInternals->JointPlanToken) { return "unknown"; }
  auto submittedNode = mInternals->JointPlanNode.lock();
  if (!submittedNode || !rclcpp::ok(submittedNode->get_node_base_interface()->get_context()) ||
      submittedNode != GetROSNodePointer() || !this->GetScene()) {
    CancelJointPlan(token);
  }
  std::lock_guard<std::mutex> lock(mInternals->JointPlan->Mutex);
  return mInternals->JointPlan->Status;
}

vtkMoveitMsgsRobotTrajectory* vtkMRMLROS2MotionControlNode::TakeJointPlanResult(const std::string& token)
{
  auto* trajectory = vtkMoveitMsgsRobotTrajectory::New();
  const auto status = GetJointPlanStatus(token);
  if (status != "ready" && status != "error") { return trajectory; }
  std::shared_ptr<JointPlanAction::Result> result;
  {
    std::lock_guard<std::mutex> lock(mInternals->JointPlan->Mutex);
    result = mInternals->JointPlan->Result;
    mInternals->LastJointPlanMessage = mInternals->JointPlan->Message;
  }
  if (status == "ready" && result) {
    mInternals->CachedTrajectory = result->planned_trajectory;
    vtkROS2ToSlicer(result->planned_trajectory, vtkSmartPointer<vtkMoveitMsgsRobotTrajectory>(trajectory));
  }
  mInternals->JointPlanGroup.reset();
  mInternals->JointPlan.reset();
  mInternals->JointPlanNode.reset();
  mInternals->JointPlanToken.clear();
  return trajectory;
}

bool vtkMRMLROS2MotionControlNode::CancelJointPlan(const std::string& token)
{
  if (!mInternals->JointPlan || token != mInternals->JointPlanToken) { return false; }
  JointPlanHandle::SharedPtr handle;
  {
    std::lock_guard<std::mutex> lock(mInternals->JointPlan->Mutex);
    mInternals->JointPlan->Status = "cancelled";
    mInternals->JointPlan->Result.reset();
    handle = mInternals->JointPlan->Handle;
    mInternals->JointPlan->Handle.reset();
  }
  mInternals->CachedTrajectory = {};
  if (handle && mInternals->JointPlanGroup) {
    try { mInternals->JointPlanGroup->getMoveGroupClient().async_cancel_goal(handle); }
    catch (const std::exception&) {} // Local authority is revoked even if ROS is gone.
  }
  // No backend ACK wait. Retain only a tombstone: cancellation is unconfirmed,
  // and no further plan/execution is allowed until this MRML node is recreated.
  mInternals->JointPlanGroup.reset();
  mInternals->LastJointPlanMessage = "Local result revoked; backend cancellation unconfirmed; recreate the motion-control node before another request";
  return true;
}

std::string vtkMRMLROS2MotionControlNode::GetLastJointPlanMessage() const
{
  return mInternals->LastJointPlanMessage;
}

std::string vtkMRMLROS2MotionControlNode::GetLastJointPlannerId() const
{
  return mInternals->LastJointPlannerId;
}

namespace {
void DentoQuatPoseToMatrix(const geometry_msgs::msg::Pose & p, double m[3][4])
{
  const double x = p.orientation.x, y = p.orientation.y, z = p.orientation.z, w = p.orientation.w;
  m[0][0] = 1 - 2 * (y * y + z * z); m[0][1] = 2 * (x * y - z * w); m[0][2] = 2 * (x * z + y * w);
  m[1][0] = 2 * (x * y + z * w); m[1][1] = 1 - 2 * (x * x + z * z); m[1][2] = 2 * (y * z - x * w);
  m[2][0] = 2 * (x * z - y * w); m[2][1] = 2 * (y * z + x * w); m[2][2] = 1 - 2 * (x * x + y * y);
  m[0][3] = p.position.x; m[1][3] = p.position.y; m[2][3] = p.position.z;
}

void DentoComposePose(const double a[3][4], const double b[3][4], double out[3][4])
{
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      double v = (c == 3) ? a[r][3] : 0.0;
      for (int k = 0; k < 3; ++k) { v += a[r][k] * b[k][c]; }
      out[r][c] = v;
    }
  }
}

std::string DentoJsonEscape(const std::string & text)
{
  std::string out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') { out += '\\'; }
    out += ch;
  }
  return out;
}
}  // namespace

bool vtkMRMLROS2MotionControlNode::ApplyMoveItCollisionObject(
    vtkMRMLModelNode * modelNode, const std::string & frameId, double timeoutSec)
{
  mInternals->LastPlanningSceneMessage.clear();
  auto node = GetROSNodePointer();
  if (!node) {
    mInternals->LastPlanningSceneMessage = "ROS node is unavailable";
    return false;
  }
  if (!modelNode) {
    mInternals->LastPlanningSceneMessage = "model node is null";
    return false;
  }
  moveit_msgs::msg::CollisionObject object;
  vtkSlicerToROS2(modelNode, object, node);
  object.header.frame_id = frameId;
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
  request->scene.is_diff = true;
  request->scene.robot_state.is_diff = true;
  request->scene.world.collision_objects.push_back(object);
  auto client = node->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
  const auto timeout = std::chrono::duration<double>(timeoutSec > 0.0 ? timeoutSec : 5.0);
  if (!client->wait_for_service(timeout)) {
    mInternals->LastPlanningSceneMessage = "/apply_planning_scene is unavailable after the bounded timeout";
    return false;
  }
  auto future = client->async_send_request(request);
  const auto startTime = std::chrono::steady_clock::now();
  while (future.wait_for(std::chrono::milliseconds(5)) != std::future_status::ready) {
    rclcpp::spin_some(node);
    if (std::chrono::steady_clock::now() - startTime > timeout) {
      mInternals->LastPlanningSceneMessage = "/apply_planning_scene request timed out for " + object.id;
      return false;
    }
  }
  auto response = future.get();
  if (!response || !response->success) {
    mInternals->LastPlanningSceneMessage = "MoveGroup rejected the planning-scene diff for " + object.id;
    return false;
  }
  mInternals->LastPlanningSceneMessage = "MoveGroup applied " + object.id;
  return true;
}

std::string vtkMRMLROS2MotionControlNode::GetLastPlanningSceneMessage() const
{
  return mInternals->LastPlanningSceneMessage;
}

std::string vtkMRMLROS2MotionControlNode::GetMoveItWorldObjectBounds(double timeoutSec)
{
  mInternals->LastPlanningSceneMessage.clear();
  auto node = GetROSNodePointer();
  if (!node) {
    mInternals->LastPlanningSceneMessage = "ROS node is unavailable";
    return "";
  }
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  const auto timeout = std::chrono::duration<double>(timeoutSec > 0.0 ? timeoutSec : 3.0);
  if (!client->wait_for_service(timeout)) {
    mInternals->LastPlanningSceneMessage = "/get_planning_scene is unavailable after the bounded timeout";
    return "";
  }
  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components =
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY;
  auto future = client->async_send_request(request);
  const auto startTime = std::chrono::steady_clock::now();
  while (future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
    rclcpp::spin_some(node);
    if (std::chrono::steady_clock::now() - startTime > timeout) {
      mInternals->LastPlanningSceneMessage = "/get_planning_scene request timed out";
      return "";
    }
  }
  auto response = future.get();
  if (!response) {
    mInternals->LastPlanningSceneMessage = "/get_planning_scene returned a null response";
    return "";
  }
  std::ostringstream json;
  json.precision(9);
  json << "[";
  bool first = true;
  for (const auto & object : response->scene.world.collision_objects) {
    double objectPose[3][4];
    DentoQuatPoseToMatrix(object.pose, objectPose);
    double lo[3] = {1e300, 1e300, 1e300};
    double hi[3] = {-1e300, -1e300, -1e300};
    size_t vertexCount = 0;
    for (size_t i = 0; i < object.meshes.size(); ++i) {
      double meshPose[3][4];
      double full[3][4];
      DentoQuatPoseToMatrix(i < object.mesh_poses.size() ? object.mesh_poses[i] : geometry_msgs::msg::Pose(), meshPose);
      DentoComposePose(objectPose, meshPose, full);
      for (const auto & v : object.meshes[i].vertices) {
        for (int r = 0; r < 3; ++r) {
          const double value = full[r][0] * v.x + full[r][1] * v.y + full[r][2] * v.z + full[r][3];
          lo[r] = std::min(lo[r], value);
          hi[r] = std::max(hi[r], value);
        }
        ++vertexCount;
      }
    }
    if (!first) { json << ","; }
    first = false;
    json << "{\"id\":\"" << DentoJsonEscape(object.id) << "\",\"frame\":\""
         << DentoJsonEscape(object.header.frame_id) << "\",\"vertices\":" << vertexCount
         << ",\"primitives\":" << object.primitives.size() << ",\"bounds_mm\":";
    if (vertexCount == 0) {
      json << "null";
    } else {
      json << "[" << lo[0] * 1000.0 << "," << hi[0] * 1000.0 << "," << lo[1] * 1000.0 << ","
           << hi[1] * 1000.0 << "," << lo[2] * 1000.0 << "," << hi[2] * 1000.0 << "]";
    }
    json << "}";
  }
  json << "]";
  std::ostringstream message;
  message << "MoveGroup PlanningScene has " << response->scene.world.collision_objects.size()
          << " world collision object(s)";
  mInternals->LastPlanningSceneMessage = message.str();
  return json.str();
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
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or cancellation is unresolved";
    return vtkMoveitMsgsRobotTrajectory::New();
  }

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
    // No Cartesian path is a normal answer (e.g. one rejected IK candidate); the
    // caller reads LastCartesianPathFraction and reports real step failures. An
    // error here turned Slicer's error log red on every passing plan (DentoBot
    // r22: 117 per run).
    vtkDebugMacro(<< "PlanMoveItCartesianTrajectory: Cartesian planning failed for group '"
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
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or cancellation is unresolved";
    return false;
  }

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
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or cancellation is unresolved";
    return false;
  }

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
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or cancellation is unresolved";
    return false;
  }

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
  if (mInternals->JointPlan) {
    mInternals->LastJointPlanMessage = "A joint request is pending or cancellation is unresolved";
    return false;
  }

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

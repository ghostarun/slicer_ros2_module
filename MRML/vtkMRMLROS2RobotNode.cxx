#include <vtkMRMLROS2RobotNode.h>
#include <vtkMRMLROS2RobotNodeInternals.h>

#include <vtkEventBroker.h>
#include <vtkTransform.h>
#include <vtkPointSet.h>
#include <vtkSTLReader.h>
#include <vtkOBJReader.h>
#include <vtkSphereSource.h>
#include <vtkTransformFilter.h>
#include <vtksys/SystemTools.hxx>
#include "vtkAssImpConversion.h"

#include <vtkMRMLScene.h>
#include <vtkMRMLTransformNode.h>
#include <vtkMRMLLinearTransformNode.h>
#include <vtkMRMLModelNode.h>
#include <vtkMRMLModelDisplayNode.h>

#include <vtkMRMLROS2Utils.h>
#include <vtkMRMLROS2NodeNode.h>
#include <vtkMRMLROS2ParameterNode.h>
#include <vtkMRMLROS2Tf2LookupNode.h>

#include <regex>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <vtkMRMLROS2NodeInternals.h>
#include <eigen3/Eigen/Geometry>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <thread>
#include <queue>
#include <cmath>
#include <limits>
#include <sstream>

#include <vtkMoveitMsgsRobotTrajectory.h>
#include <vtkROS2ToSlicer.h>
#include <vtkSlicerToROS2.h>

#include <QTimer>

// MoveIt kinematics and planning includes
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
// ROS2 parameter client for reading remote node parameters
#include <rclcpp/parameter_client.hpp>
#include <chrono>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/SVD>

// KDL includes
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

vtkStandardNewMacro(vtkMRMLROS2RobotNode);

vtkMRMLNode * vtkMRMLROS2RobotNode::CreateNodeInstance(void)
{
  return SelfType::New();
}


const char * vtkMRMLROS2RobotNode::GetNodeTagName(void)
{
  return "ROS2RobotNode";
}


vtkMRMLROS2RobotNode::vtkMRMLROS2RobotNode()
{
  mInternals = std::make_unique<vtkMRMLROS2RobotNodeInternals>();
}


vtkMRMLROS2RobotNode::~vtkMRMLROS2RobotNode()
{
  this->ReleaseMoveItResources();
  this->RemoveRobotVisualization();
}

void vtkMRMLROS2RobotNode::ReleaseMoveItResources()
{
  if (mInternals->PlanningSceneMonitorPtr) {
    mInternals->PlanningSceneMonitorPtr->stopStateMonitor();
    mInternals->PlanningSceneMonitorPtr->stopSceneMonitor();
    mInternals->PlanningSceneMonitorPtr->stopWorldGeometryMonitor();
  }
  mInternals->PlanningSceneMonitorPtr.reset();
  mInternals->JointModelGroupPtr = nullptr;
  mInternals->IKGroupName.clear();
  mInternals->RobotModelPtr.reset();
  mInternals->RobotModelLoaderPtr.reset();
}


void vtkMRMLROS2RobotNode::RemoveRobotVisualization()
{
  if (!this->GetScene()) {
    return;
  }

  // Remove nodes referenced as "model"
  int nbModelRefs = this->GetNumberOfNodeReferences("model");
  for (int i = nbModelRefs - 1; i >= 0; --i) {
    vtkMRMLNode* node = this->GetNthNodeReference("model", i);
    if (node) {
      vtkMRMLModelNode* modelNode = vtkMRMLModelNode::SafeDownCast(node);
      if (modelNode) {
        // 1. Remove the display node(s) associated with this model
        std::vector<const char*> displayNodeIDs;
        int numDisplayNodes = modelNode->GetNumberOfDisplayNodes();
        for (int d = 0; d < numDisplayNodes; ++d) {
          displayNodeIDs.push_back(modelNode->GetNthDisplayNodeID(d));
        }
        for (const char* displayNodeID : displayNodeIDs) {
          vtkMRMLNode* displayNode = this->GetScene()->GetNodeByID(displayNodeID);
          if (displayNode) {
            this->GetScene()->RemoveNode(displayNode);
          }
        }
        modelNode->RemoveAllDisplayNodeIDs();
      }
      // 2. Remove the model node itself
      this->GetScene()->RemoveNode(node);
    }
    this->RemoveNthNodeReferenceID("model", i);
  }

  // Remove nodes referenced as "lookup"
  int nbLookupRefs = this->GetNumberOfNodeReferences("lookup");
  for (int i = nbLookupRefs - 1; i >= 0; --i) {
    vtkMRMLNode* node = this->GetNthNodeReference("lookup", i);
    if (node) {
      vtkMRMLROS2Tf2LookupNode* lookupNode = vtkMRMLROS2Tf2LookupNode::SafeDownCast(node);
      if (lookupNode) {
        const char* ros2NodeID = this->GetNodeReferenceID("node");
        if (ros2NodeID) {
          lookupNode->RemoveFromROS2Node(ros2NodeID);
        }
      }
      // 2. Remove the lookup node
      this->GetScene()->RemoveNode(node);
    }
    this->RemoveNthNodeReferenceID("lookup", i);
  }

  // Clear transient vectors in internals
  mInternals->mVisualVector.clear();
  mInternals->mMaterialsMap.clear();
  mInternals->mLinkMaterials.clear();
  mInternals->mMaterialVector.clear();
  mInternals->mParentLinkPointer.reset();
  mInternals->mChildLinkPointer.clear();
  mInternals->mLinkOrigins.clear();
  mInternals->mLinkNames.clear();
  mInternals->mLinkParentNames.clear();
  mInternals->mLinkModelFiles.clear();
  mInternals->mLinkModels.clear();
  mInternals->mLookupNodes.clear();

  mNumberOfLinks = 0;
}


bool vtkMRMLROS2RobotNode::AddToROS2Node(const char * nodeId,
                                         const std::string & robotName,
                                         const std::string & parameterNodeName,
                                         const std::string & parameterName,
                                         const std::string & fixedFrame,
                                         const std::string & tfPrefix)
{
  this->SetName(mMRMLNodeName.c_str());
  std::string errorMessage;
  vtkMRMLROS2NodeNode * mrmlROSNodePtr = vtkMRMLROS2::CheckROS2NodeExists(this, nodeId, errorMessage);
  if (!mrmlROSNodePtr) {
    vtkErrorMacro(<< "AddToROS2Node: " << errorMessage);
    return false;
  }
  // Add the robot to the ros2 node
  mrmlROSNodePtr->SetNthNodeReferenceID("robot",
                                        mrmlROSNodePtr->GetNumberOfNodeReferences("robot"),
                                        this->GetID());
  this->SetNodeReferenceID("node", nodeId);
  mMRMLROS2Node = mrmlROSNodePtr;
  mInternals->mParameterNodeName = parameterNodeName;
  mInternals->mParameterName = parameterName;
  mInternals->mFixedFrame = fixedFrame;
  if ((tfPrefix == "") || ((*(tfPrefix.crend()) == '/'))) {
    mInternals->mTfPrefix = tfPrefix;
  } else {
    mInternals->mTfPrefix = tfPrefix + '/';
  }
  SetRobotDescriptionParameterNode();
  SetRobotName(robotName);
  return true;
}


bool vtkMRMLROS2RobotNode::RemoveFromROS2Node(const char * nodeId)
{
  // Check that the robot is in the scene
  std::string errorMessage;
  vtkMRMLROS2NodeNode * mrmlROSNodePtr = vtkMRMLROS2::CheckROS2NodeExists(this, nodeId, errorMessage);
  if (!mrmlROSNodePtr) {
    vtkErrorMacro(<< "RemoveFromROS2Node: " << errorMessage);
    return false;
  }

  this->ReleaseMoveItResources();

  // Remove the robot visualization
  this->RemoveRobotVisualization();

  // Remove the goal state robot if it exists
  this->RemoveGoalStateRobot();

  // Remove the parameter node
  if (mRobotDescriptionParameterNode) {
    mRobotDescriptionParameterNode->RemoveFromROS2Node(nodeId);
    this->GetScene()->RemoveNode(mRobotDescriptionParameterNode);
    mRobotDescriptionParameterNode = nullptr;
  }

  // Find and remove the reference by index in the ROS2 node
  int index = -1;
  int numRobots = mrmlROSNodePtr->GetNumberOfNodeReferences("robot");
  for (int i = 0; i < numRobots; ++i) {
    const char* refID = mrmlROSNodePtr->GetNthNodeReferenceID("robot", i);
    if (refID && std::string(refID) == this->GetID()) {
      index = i;
      break;
    }
  }

  if (index != -1) {
    mrmlROSNodePtr->RemoveNthNodeReferenceID("robot", index);
    this->SetNodeReferenceID("node", nullptr);
    mMRMLROS2Node = nullptr;
    return true;
  }

  vtkErrorMacro(<< "RemoveFromROS2Node: this robot has not been added to the ROS2 node.");
  return false;
}


bool vtkMRMLROS2RobotNode::SetRobotDescriptionParameterNode(void)
{
  // Check if the node is in the scene
  if (!this->GetScene()) {
    vtkErrorMacro(<< "SetRobotDescriptionParameterNode: robot node needs to be added to the scene first");
    return false;
  }
  // Create a new parameter node
  mRobotDescriptionParameterNode = vtkSmartPointer<vtkMRMLROS2ParameterNode>::New();
  this->GetScene()->AddNode(mRobotDescriptionParameterNode);
  mRobotDescriptionParameterNode->SetName((mRobotName + "_parameter").c_str());
  mRobotDescriptionParameterNode->AddToROS2Node(mMRMLROS2Node->GetID(), mInternals->mParameterNodeName);
  mRobotDescriptionParameterNode->AddParameter(mInternals->mParameterName);
  ObserveParameterNode(mRobotDescriptionParameterNode);
  return true;
}


void vtkMRMLROS2RobotNode::ObserveParameterNode(vtkMRMLROS2ParameterNode * node)
{
  // Set up the observer for the robot state publisher
  if (!this->GetScene()->GetNodeByID(node->GetID())) {
    vtkErrorMacro(<< "ObserveParameterNode: robot node is not in the scene.");
    return;
  }
  node->AddObserver(vtkMRMLROS2ParameterNode::ParameterModifiedEvent, this, &vtkMRMLROS2RobotNode::ObserveParameterNodeCallback);
  this->SetAndObserveNodeReferenceID("parameter", node->GetID());
}


void vtkMRMLROS2RobotNode::ObserveParameterNodeCallback( vtkObject* caller, unsigned long, void* vtkNotUsed(callData))
{
  // Manage parameter callback when robot description is available
  vtkMRMLROS2ParameterNode* parameterNode = vtkMRMLROS2ParameterNode::SafeDownCast(caller);
  if (!parameterNode) {
    vtkErrorMacro(<< "ObserveParameterNodeCallback: parameter node is not valid");
    return;
  }
  // Use IsParameterSet to check if the parameter is set
  if (!mRobotDescriptionParameterNode->IsParameterSet(mInternals->mParameterName)) {
    // Silently return until parameter is available
    return;
  }

  if (mRobotDescriptionParameterNode->GetParameterType(mInternals->mParameterName) != "string" &&
      mRobotDescriptionParameterNode->GetParameterType(mInternals->mParameterName) != "String") {
    std::string outtype = mRobotDescriptionParameterNode->GetParameterType(mInternals->mParameterName);
    vtkErrorMacro(<< "ObserveParameterNodeCallback: parameter \"" << mInternals->mParameterName << "\" is of type " << outtype << " and not string.");
    return;
  }

  std::string newDescription = mRobotDescriptionParameterNode->GetParameterAsString(mInternals->mParameterName);
  if (newDescription != mInternals->mRobotDescription || mNumberOfLinks == 0) {
    mInternals->mRobotDescription = newDescription;
    if (ParseRobotDescription()) {
      SetupRobotVisualization();
    } else {
      vtkErrorMacro(<< "ObserveParameterNodeCallback: Failed to parse robot description");
    }
  }

}


bool vtkMRMLROS2RobotNode::RemoveGoalStateRobot()
{
  vtkMRMLScene* scene = this->GetScene();
  if (!scene) {
    vtkErrorMacro(<< "RemoveGoalStateRobot: robot node needs to be added to the scene first");
    return false;
  }

  auto removeGoalReferences = [&](const char* role) {
    int count = this->GetNumberOfNodeReferences(role);
    for (int i = count - 1; i >= 0; --i) {
      vtkMRMLNode* node = this->GetNthNodeReference(role, i);
      if (node) {
        vtkMRMLModelNode* modelNode = vtkMRMLModelNode::SafeDownCast(node);
        if (modelNode) {
          std::vector<const char*> displayNodeIDs;
          int numDisplayNodes = modelNode->GetNumberOfDisplayNodes();
          for (int d = 0; d < numDisplayNodes; ++d) {
            displayNodeIDs.push_back(modelNode->GetNthDisplayNodeID(d));
          }
          for (const char* displayNodeID : displayNodeIDs) {
            vtkMRMLNode* displayNode = scene->GetNodeByID(displayNodeID);
            if (displayNode) {
              scene->RemoveNode(displayNode);
            }
          }
          modelNode->RemoveAllDisplayNodeIDs();
        }
        scene->RemoveNode(node);
      }
      this->RemoveNthNodeReferenceID(role, i);
    }
  };

  removeGoalReferences("goal_model");
  removeGoalReferences("goal_transform");
  return true;
}


bool vtkMRMLROS2RobotNode::CreateGoalStateRobot(vtkMRMLROS2RobotNode * sourceRobot)
{
  vtkMRMLScene* scene = this->GetScene();
  if (!scene) {
    vtkErrorMacro(<< "CreateGoalStateRobot: robot node needs to be added to the scene first");
    return false;
  }

  if (!sourceRobot) {
    vtkErrorMacro(<< "CreateGoalStateRobot: source robot is null");
    return false;
  }

  if (!sourceRobot->GetScene()) {
    vtkErrorMacro(<< "CreateGoalStateRobot: source robot must be added to the scene first");
    return false;
  }

  // Check if goal robot already exists
  if (this->GetNumberOfNodeReferences("goal_model") > 0) {
    vtkWarningMacro(<< "CreateGoalStateRobot: goal robot already exists, skipping.");
    return true;
  }

  int lookupCount = sourceRobot->GetNumberOfNodeReferences("lookup");
  std::vector<vtkSmartPointer<vtkMRMLLinearTransformNode>> goalTransforms;
  goalTransforms.reserve(lookupCount);
  std::unordered_map<std::string, vtkSmartPointer<vtkMRMLLinearTransformNode>> goalTransformsByLookupId;

  for (int i = 0; i < lookupCount; ++i) {
    vtkMRMLROS2Tf2LookupNode* lookup = vtkMRMLROS2Tf2LookupNode::SafeDownCast(sourceRobot->GetNthNodeReference("lookup", i));
    if (!lookup) {
      goalTransforms.push_back(nullptr);
      continue;
    }

    vtkSmartPointer<vtkMRMLLinearTransformNode> goalTransform = vtkSmartPointer<vtkMRMLLinearTransformNode>::New();
    scene->AddNode(goalTransform);

    std::string childName = lookup->GetChildID();
    if (!mInternals->mTfPrefix.empty() && childName.rfind(mInternals->mTfPrefix, 0) == 0) {
      childName = childName.substr(mInternals->mTfPrefix.size());
    }
    std::string transformName = childName + "_goal_transform";
    goalTransform->SetName(transformName.c_str());

    vtkNew<vtkMatrix4x4> matrix;
    lookup->GetMatrixTransformToParent(matrix);
    goalTransform->SetMatrixTransformToParent(matrix);

    goalTransforms.push_back(goalTransform);
    goalTransformsByLookupId[lookup->GetID()] = goalTransform;
    this->AddNodeReferenceID("goal_transform", goalTransform->GetID());
  }

  for (int i = 0; i < lookupCount; ++i) {
    vtkMRMLROS2Tf2LookupNode* lookup = vtkMRMLROS2Tf2LookupNode::SafeDownCast(sourceRobot->GetNthNodeReference("lookup", i));
    if (!lookup || i >= static_cast<int>(goalTransforms.size()) || !goalTransforms[i]) {
      continue;
    }

    std::string parentFrame = lookup->GetParentID();
    for (int j = 0; j < lookupCount; ++j) {
      vtkMRMLROS2Tf2LookupNode* potentialParent = vtkMRMLROS2Tf2LookupNode::SafeDownCast(sourceRobot->GetNthNodeReference("lookup", j));
      if (!potentialParent || i == j || j >= static_cast<int>(goalTransforms.size()) || !goalTransforms[j]) {
        continue;
      }
      if (potentialParent->GetChildID() == parentFrame) {
        goalTransforms[i]->SetAndObserveTransformNodeID(goalTransforms[j]->GetID());
        break;
      }
    }
  }

  int modelCount = sourceRobot->GetNumberOfNodeReferences("model");
  vtkDebugMacro(<< "Found " << modelCount << " model nodes to duplicate");
  for (int i = 0; i < modelCount; ++i) {
    vtkMRMLModelNode* original = vtkMRMLModelNode::SafeDownCast(sourceRobot->GetNthNodeReference("model", i));
    if (!original) {
      continue;
    }

    auto goalTransformIt = goalTransformsByLookupId.find(original->GetTransformNodeID() ? original->GetTransformNodeID() : "");
    if (goalTransformIt == goalTransformsByLookupId.end() || !goalTransformIt->second) {
      vtkWarningMacro(<< "CreateGoalStateRobot: could not map model '"
                      << (original->GetName() ? original->GetName() : "model")
                      << "' to a goal transform");
      continue;
    }
    vtkMRMLLinearTransformNode* goalTransform = goalTransformIt->second;

    vtkSmartPointer<vtkMRMLModelNode> goal = vtkSmartPointer<vtkMRMLModelNode>::New();
    scene->AddNode(goal);

    std::string goalName = std::string(original->GetName() ? original->GetName() : "model") + "_goal";
    goal->SetName(goalName.c_str());
    vtkDebugMacro(<< "  Creating goal: " << goalName << " with transform: " << goalTransform->GetName());

    if (original->GetMesh()) {
      goal->SetAndObserveMesh(original->GetMesh());
    }

    vtkMRMLModelDisplayNode* origDisp = vtkMRMLModelDisplayNode::SafeDownCast(original->GetDisplayNode());
    vtkNew<vtkMRMLModelDisplayNode> goalDisp;
    scene->AddNode(goalDisp.GetPointer());
    if (origDisp) {
      goalDisp->Copy(origDisp);
    }
    goalDisp->SetColor(0.0, 1.0, 1.0);
    goalDisp->SetOpacity(0.30);
    goal->SetAndObserveDisplayNodeID(goalDisp->GetID());

    goal->SetAndObserveTransformNodeID(goalTransform->GetID());
    this->AddNodeReferenceID("goal_model", goal->GetID());
  }

  for (int i = 0; i < lookupCount && i < static_cast<int>(goalTransforms.size()); ++i) {
    vtkMRMLROS2Tf2LookupNode* lookup = vtkMRMLROS2Tf2LookupNode::SafeDownCast(sourceRobot->GetNthNodeReference("lookup", i));
    if (!lookup || !goalTransforms[i]) {
      continue;
    }
    vtkNew<vtkMatrix4x4> matrix;
    lookup->GetMatrixTransformToParent(matrix);
    goalTransforms[i]->SetMatrixTransformToParent(matrix);
    goalTransforms[i]->Modified();
  }

  vtkSmartPointer<vtkMRMLROS2RobotNode> sourceRobotSafe = sourceRobot;
  QTimer::singleShot(400, [this, sourceRobotSafe, goalTransforms]() {
    int lc = sourceRobotSafe->GetNumberOfNodeReferences("lookup");
    for (int i = 0; i < lc && i < static_cast<int>(goalTransforms.size()); ++i) {
      vtkMRMLROS2Tf2LookupNode* lookup = vtkMRMLROS2Tf2LookupNode::SafeDownCast(sourceRobotSafe->GetNthNodeReference("lookup", i));
      if (!lookup || !goalTransforms[i]) {
        continue;
      }
      vtkNew<vtkMatrix4x4> matrix;
      lookup->GetMatrixTransformToParent(matrix);
      goalTransforms[i]->SetMatrixTransformToParent(matrix);
      goalTransforms[i]->Modified();
    }
  });

  vtkDebugMacro(<< "goal creation complete!");
  return true;
}


bool vtkMRMLROS2RobotNode::ParseRobotDescription(void)
{
  // Parser the urdf file into an urdf model - to get names of links and pos/ rpy
  if (!mInternals->mURDFModel.initString(mInternals->mRobotDescription)) {
    vtkErrorMacro(<< "ParseRobotDescription: failed to parse robot description");
    return false;
  }
  return true;
}


void vtkMRMLROS2RobotNode::SetupRobotVisualization(void)
{
  // Before setting up new visualization, remove any existing robot nodes 
  // to avoid duplication and leaks.
  this->RemoveRobotVisualization();

  // 1. Initialize lookup list and visual vectors from URDF
  auto root = mInternals->mURDFModel.getRoot();
  if (!root) {
    vtkErrorMacro(<< "SetupRobotVisualization: root link not found in URDF model");
    return;
  }
  
  std::string root_name = root->name;
  mInternals->mLinkNames.push_back(root_name);
  mInternals->mLinkParentNames.push_back(root_name);
  mInternals->mVisualVector.push_back(root->visual);
  mInternals->mMaterialsMap = mInternals->mURDFModel.materials_;
  mInternals->mLinkMaterials.push_back(root->visual != nullptr ? root->visual->material_name : "");
  mInternals->mLinkOrigins.push_back(root->visual != nullptr ? root->visual->origin : urdf::Pose());

  // BFS to explore links
  size_t lastExplored = 0;
  while (lastExplored < mInternals->mVisualVector.size()) {
    auto parentLink = mInternals->mURDFModel.getLink(mInternals->mLinkNames[lastExplored]);
    if (parentLink) {
      for (const auto& childLink : parentLink->child_links) {
        if (!childLink) continue;
        mInternals->mLinkNames.push_back(childLink->name);
        mInternals->mLinkParentNames.push_back(parentLink->name);
        mInternals->mVisualVector.push_back(childLink->visual);
        mInternals->mLinkMaterials.push_back(childLink->visual != nullptr ? childLink->visual->material_name : "");
        mInternals->mLinkOrigins.push_back(childLink->visual != nullptr ? childLink->visual->origin : urdf::Pose());
      }
    }
    lastExplored++;
  }
  mNumberOfLinks = mInternals->mLinkNames.size();

  // 2. Resolve mesh filenames
  mInternals->mLinkModelFiles.resize(mNumberOfLinks);
  for (size_t index = 0; index < mNumberOfLinks; ++index) {
    auto visual = mInternals->mVisualVector[index];
    if (visual && visual->geometry) {
      auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(visual->geometry);
      if (mesh) {
        std::string filename = mesh->filename;
        std::regex pkg_regex("^package:\\/\\/([a-zA-Z0-9_-]+)\\/(.*)");
        std::smatch match;
        if (std::regex_search(filename, match, pkg_regex)) {
          try {
            filename = ament_index_cpp::get_package_share_directory(match[1]) + "/" + std::string(match[2]);
          } catch (...) {
            vtkErrorMacro(<< "Failed to resolve package " << std::string(match[1]));
          }
        }
        mInternals->mLinkModelFiles[index] = filename;
      }
    }
  }

  // 3. Create MRML nodes (Lookups and Models)
  for (size_t i = 0; i < mNumberOfLinks; i++) {
    // Create Lookup
    vtkSmartPointer<vtkMRMLROS2Tf2LookupNode> lookup;
    std::string parentName = mInternals->mLinkParentNames[i];
    std::string childName = mInternals->mLinkNames[i];
    
    // For fixed frames, use the provided name, otherwise use prefix
    if (i == 0 && !mInternals->mFixedFrame.empty()) {
      lookup = mMRMLROS2Node->CreateAndAddTf2LookupNode(mInternals->mFixedFrame, mInternals->mTfPrefix + childName);
    } else {
      lookup = mMRMLROS2Node->CreateAndAddTf2LookupNode(mInternals->mTfPrefix + parentName, mInternals->mTfPrefix + childName);
    }
    mInternals->mLookupNodes.push_back(lookup);
    this->SetNthNodeReferenceID("lookup", i, lookup ? lookup->GetID() : nullptr);

    // Create Model Nodes (one or more parts) and apply local offset
    auto visual = mInternals->mVisualVector[i];
    std::vector< vtkSmartPointer<vtkPointSet> > meshParts;
    std::vector< std::vector<double> > meshColors;
    if (visual) {
      LoadModelFile(mInternals->mLinkModelFiles[i], meshParts, meshColors);
      
      // Apply URDF scale and unit conversion
      double sx = 1.0, sy = 1.0, sz = 1.0;
      auto meshGeom = std::dynamic_pointer_cast<urdf::Mesh>(visual->geometry);
      if (meshGeom) { sx = meshGeom->scale.x; sy = meshGeom->scale.y; sz = meshGeom->scale.z; }
      
      vtkNew<vtkTransform> scaleTransform;
      scaleTransform->Scale(sx * 1000.0, sy * 1000.0, sz * 1000.0);
      for (size_t partIndex = 0; partIndex < meshParts.size(); ++partIndex) {
        vtkNew<vtkTransformFilter> tf;
        tf->SetInputData(meshParts[partIndex]);
        tf->SetTransform(scaleTransform);
        tf->Update();
        meshParts[partIndex] = tf->GetOutput();
      }
    }

    std::vector< vtkSmartPointer<vtkMRMLModelNode> > linkModels;
    linkModels.reserve(meshParts.size());

    bool colorSetFromUrdf = false;
    std::vector<double> urdfColor = {0.5, 0.5, 0.5};
    if (!mInternals->mMaterialsMap.empty() && !mInternals->mLinkMaterials[i].empty()) {
      auto materialIt = mInternals->mMaterialsMap.find(mInternals->mLinkMaterials[i]);
      if (materialIt != mInternals->mMaterialsMap.end() && materialIt->second) {
        auto mat = materialIt->second;
        urdfColor = {mat->color.r, mat->color.g, mat->color.b};
        colorSetFromUrdf = true;
      }
    }

    // Local offset transform
    vtkNew<vtkTransform> offsetTf;
    auto origin = mInternals->mLinkOrigins[i];
    offsetTf->Translate(origin.position.x * 1000.0, origin.position.y * 1000.0, origin.position.z * 1000.0);
    double r, p, y;
    origin.rotation.getRPY(r, p, y);
    offsetTf->RotateZ(y * (180.0/M_PI));
    offsetTf->RotateY(p * (180.0/M_PI));
    offsetTf->RotateX(r * (180.0/M_PI));

    for (size_t partIndex = 0; partIndex < meshParts.size(); ++partIndex) {
      vtkNew<vtkMRMLModelNode> modelNode;
      this->GetScene()->AddNode(modelNode);
      modelNode->SetName((mInternals->mLinkNames[i] + "_model_" + std::to_string(partIndex)).c_str());
      modelNode->SetAndObserveMesh(meshParts[partIndex]);
      modelNode->ApplyTransform(offsetTf);

      vtkNew<vtkMRMLModelDisplayNode> displayNode;
      this->GetScene()->AddNode(displayNode);
      modelNode->SetAndObserveDisplayNodeID(displayNode->GetID());
      displayNode->SetAmbient(0.2);
      displayNode->SetDiffuse(0.8);

      if (colorSetFromUrdf) {
        displayNode->SetColor(urdfColor[0], urdfColor[1], urdfColor[2]);
      } else if (partIndex < meshColors.size() && meshColors[partIndex].size() >= 3) {
        displayNode->SetColor(meshColors[partIndex][0], meshColors[partIndex][1], meshColors[partIndex][2]);
      } else {
        displayNode->SetColor(0.5, 0.5, 0.5);
      }

      this->AddNodeReferenceID("model", modelNode->GetID());
      linkModels.push_back(modelNode);
    }

    mInternals->mLinkModels.push_back(linkModels);
  }

  // 4. Finalize Transform Tree
  for (size_t i = 0; i < mInternals->mLookupNodes.size(); i++) {
    auto lookup = mInternals->mLookupNodes[i];
    if (!lookup) continue;
    lookup->SetModifiedOnLookup(i == 0);
    std::string parent = lookup->GetParentID();
    for (size_t j = 0; j < mInternals->mLookupNodes.size(); j++) {
      auto potentialParent = mInternals->mLookupNodes[j];
      if (potentialParent && potentialParent->GetChildID() == parent && i != j) {
        lookup->SetAndObserveTransformNodeID(potentialParent->GetID());
      }
    }
    if (i < mInternals->mLinkModels.size()) {
      for (auto& model : mInternals->mLinkModels[i]) {
        if (model) {
          model->SetAndObserveTransformNodeID(lookup->GetID());
        }
      }
    }
  }

  if (!SetupKDLIKWithLimits()) {
    vtkErrorMacro(<< "SetupRobotVisualization: failed to initialize KDL chain and joint limits");
  }

  // Clear transient vectors
  mInternals->mLinkModels.clear();
  mInternals->mLookupNodes.clear();
  mInternals->mLinkNames.clear();
  mInternals->mLinkParentNames.clear();
  mInternals->mLinkModelFiles.clear();
}

void vtkMRMLROS2RobotNode::LoadModelFile(const std::string& filename,
                                         std::vector< vtkSmartPointer<vtkPointSet> >& meshParts,
                                         std::vector< std::vector<double> >& meshColors)
{
  meshParts.clear();
  meshColors.clear();

  if (filename.empty()) {
    return;
  }
  
  std::string ext = vtksys::SystemTools::LowerCase(vtksys::SystemTools::GetFilenameLastExtension(filename));

  if (ext == ".stl") {
    vtkNew<vtkSTLReader> reader;
    reader->SetFileName(filename.c_str());
    reader->Update();
    meshParts.push_back(reader->GetOutput());
    meshColors.push_back({0.5, 0.5, 0.5});
    return;
  } else if (ext == ".obj") {
    vtkNew<vtkOBJReader> reader;
    reader->SetFileName(filename.c_str());
    reader->Update();
    meshParts.push_back(reader->GetOutput());
    meshColors.push_back({0.5, 0.5, 0.5});
    return;
  } else {
    std::vector<vtkAssImpMeshPart> parts = vtkAssImpConversion::vtkAssImpToPolyDataParts(filename);
    for (const auto& part : parts) {
      if (!part.PolyData) {
        continue;
      }
      meshParts.push_back(part.PolyData);
      meshColors.push_back(part.Color.empty() ? std::vector<double>{0.5, 0.5, 0.5} : part.Color);
    }
    if (!meshParts.empty()) {
      return;
    }
  }
  
  vtkErrorMacro(<< "Failed to load model file: " << filename << ". Using fallback sphere.");
  vtkNew<vtkSphereSource> sphere;
  sphere->SetRadius(10.0);
  sphere->Update();
  meshParts.push_back(sphere->GetOutput());
  meshColors.push_back({0.5, 0.5, 0.5});
}


void vtkMRMLROS2RobotNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);
}


void vtkMRMLROS2RobotNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent); // This will take care of referenced nodes
  vtkMRMLWriteXMLBeginMacro(of);
  vtkMRMLWriteXMLStdStringMacro(RobotName, RobotName);
  vtkMRMLWriteXMLEndMacro();
}


void vtkMRMLROS2RobotNode::ReadXMLAttributes(const char** atts)
{
  int wasModifying = this->StartModify();
  Superclass::ReadXMLAttributes(atts); // This will take care of referenced nodes
  vtkMRMLReadXMLBeginMacro(atts);
  vtkMRMLReadXMLStdStringMacro(RobotName, RobotName);
  vtkMRMLReadXMLEndMacro();
  this->EndModify(wasModifying);
}

// MoveIt IK implementation (commented out for faster build)
bool vtkMRMLROS2RobotNode::SetupIKMoveIt(const std::string & groupName)
{

  if (!mMRMLROS2Node) {
    vtkErrorMacro(<< "setupIK: ROS2 node not available");
    return false;
  }

  if (mInternals->mRobotDescription.empty()) {
    vtkErrorMacro(<< "setupIK: robot description not available");
    return false;
  }

  try {
    auto node = mMRMLROS2Node->mInternals->mNodePointer;
    std::string prefix = "robot_description_kinematics." + groupName;

    // Helper for declaring/updating parameters
    auto ensureParam = [&](const std::string& name, auto value) {
      if (!node->has_parameter(name)) {
        node->declare_parameter(name, value);
      } else {
        node->set_parameter(rclcpp::Parameter(name, value));
      }
    };

    // Set kinematics parameters
    ensureParam(prefix + ".kinematics_solver", std::string("kdl_kinematics_plugin/KDLKinematicsPlugin"));
    ensureParam(prefix + ".kinematics_solver_search_resolution", 0.005);
    ensureParam(prefix + ".kinematics_solver_timeout", 0.05);

    if (mInternals->RobotModelLoaderPtr && mInternals->PlanningSceneMonitorPtr && mInternals->IKGroupName == groupName) {
      return true; // Already configured
    }

    // Load and cache RobotModel
    mInternals->RobotModelLoaderPtr = std::make_unique<robot_model_loader::RobotModelLoader>(node, "robot_description");
    mInternals->RobotModelPtr = mInternals->RobotModelLoaderPtr->getModel();

    if (!mInternals->RobotModelPtr) {
      vtkErrorMacro(<< "setupIK: Failed to load RobotModel");
      return false;
    }

    // Cache JointModelGroup
    mInternals->JointModelGroupPtr = mInternals->RobotModelPtr->getJointModelGroup(groupName);

    if (!mInternals->JointModelGroupPtr) {
      vtkErrorMacro(<< "setupIK: joint model group '" << groupName << "' not found");
      return false;
    }

    mInternals->PlanningSceneMonitorPtr =
      std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node, "robot_description", "slicer_ros2_moveit_ik");
    if (!mInternals->PlanningSceneMonitorPtr || !mInternals->PlanningSceneMonitorPtr->getPlanningScene()) {
      vtkErrorMacro(<< "setupIK: failed to initialize PlanningSceneMonitor");
      return false;
    }
    mInternals->PlanningSceneMonitorPtr->startStateMonitor();
    mInternals->PlanningSceneMonitorPtr->startSceneMonitor();
    mInternals->PlanningSceneMonitorPtr->startWorldGeometryMonitor();
    mInternals->PlanningSceneMonitorPtr->requestPlanningSceneState();

    // Verify solver is available
    const auto& solver = mInternals->JointModelGroupPtr->getSolverInstance();

    if (!solver) {
      vtkErrorMacro(<< "setupIK: no kinematics solver for group '" << groupName << "'");
      return false;
    }

    mInternals->IKGroupName = groupName;
    return true;
  }
  catch (const std::exception& e) {
    vtkErrorMacro(<< "setupIK: exception - " << e.what());
    return false;
  }
}

std::vector<double> vtkMRMLROS2RobotNode::ComputeMoveItIK(vtkMatrix4x4* targetPose, const std::string& tipLink, const std::vector<double>& seedJointValues, double timeout, bool avoidCollisions)
{
  if (!targetPose) {
    vtkErrorMacro(<< "ComputeMoveItIK: target pose is null");
    return {};
  }

  // Setup IK if needed (only once, since we auto-discover the group)
  if (!mInternals->RobotModelPtr || !mInternals->JointModelGroupPtr) {
      vtkErrorMacro(<< "ComputeMoveItIK: SetupIKMoveIt failed");
      return {};
  }

  try {
    // Create robot state for solving
    moveit::core::RobotState robot_state(mInternals->RobotModelPtr);

    if (!seedJointValues.empty() &&
        seedJointValues.size() == mInternals->JointModelGroupPtr->getVariableCount()) {
      robot_state.setJointGroupPositions(mInternals->JointModelGroupPtr, seedJointValues);
    } else {
      robot_state.setToDefaultValues();
    }

    vtkNew<vtkMatrix4x4> targetPoseSI;
    targetPoseSI->DeepCopy(targetPose);
    vtkMRMLROS2::ToSI(targetPoseSI);

    // Convert vtkMatrix4x4 to geometry_msgs Pose
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = targetPoseSI->GetElement(0, 3);  // mm to m
    pose_msg.position.y = targetPoseSI->GetElement(1, 3);
    pose_msg.position.z = targetPoseSI->GetElement(2, 3);

    // Extract rotation matrix and convert to quaternion
    Eigen::Matrix3d rot_matrix;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rot_matrix(i, j) = targetPoseSI->GetElement(i, j);
      }
    }
    
    Eigen::Quaterniond quat(rot_matrix);
    pose_msg.orientation.x = quat.x();
    pose_msg.orientation.y = quat.y();
    pose_msg.orientation.z = quat.z();
    pose_msg.orientation.w = quat.w();

    moveit::core::GroupStateValidityCallbackFn validity_callback;
    if (avoidCollisions && mInternals->PlanningSceneMonitorPtr) {
      validity_callback =
        [this](moveit::core::RobotState* state,
               const moveit::core::JointModelGroup* joint_group,
               const double* joint_group_variable_values) -> bool {
          if (!state || !joint_group || !joint_group_variable_values) {
            return false;
          }
          planning_scene_monitor::LockedPlanningSceneRO planning_scene(mInternals->PlanningSceneMonitorPtr);
          if (!planning_scene) {
            return true;
          }

          // Check collisions against the planning scene's current robot state so
          // attached collision objects, such as runtime tools, are included.
          moveit::core::RobotState collision_state(planning_scene->getCurrentState());
          collision_state.setJointGroupPositions(joint_group, joint_group_variable_values);
          collision_state.update();

          state->setJointGroupPositions(joint_group, joint_group_variable_values);
          state->update();
          return !planning_scene->isStateColliding(collision_state, joint_group->getName());
        };
    }

    // Call IK using setFromIK and reject candidate states that collide with the current planning scene.
    bool found_ik = robot_state.setFromIK(mInternals->JointModelGroupPtr, pose_msg, tipLink, timeout, validity_callback);
    if (!found_ik) {
      // vtkWarningMacro(<< "ComputeMoveItIK: IK solution not found for group '" << mInternals->IKGroupName << "'");
      return {};
    }

    std::vector<double> solution;
    robot_state.copyJointGroupPositions(mInternals->JointModelGroupPtr, solution);
    return solution;
  }
  catch (const std::exception& e) {
    vtkErrorMacro(<< "ComputeMoveItIK: exception - " << e.what());
    return {};
  }
}

std::vector<double> vtkMRMLROS2RobotNode::ComputeMoveItPositionAxisIK(
  vtkMatrix4x4* targetPose,
  const std::string& tipLink,
  const std::vector<double>& seedJointValues,
  double timeout,
  bool avoidCollisions)
{
  mInternals->LastPositionAxisIKMessage.clear();
  mInternals->LastPositionAxisIKPositionResidualMm = -1.0;
  mInternals->LastPositionAxisIKAxisResidualDeg = -1.0;
  mInternals->LastPositionAxisIKBestJointValues.clear();
  mLastMoveItPositionAxisIKTerminationReason = "not_initialized";
  mLastMoveItPositionAxisIKIterationCount = 0;
  mLastMoveItPositionAxisIKCollisionCheckStatus =
    avoidCollisions ? "not_attempted" : "not_requested";
  mLastMoveItPositionAxisIKConditionRatio = -1.0;
  if (!mInternals->RobotModelPtr || !mInternals->JointModelGroupPtr)
  {
    mInternals->LastPositionAxisIKMessage = "MoveIt position-axis IK is not initialized.";
    return {};
  }
  if (!targetPose)
  {
    mLastMoveItPositionAxisIKTerminationReason = "invalid_input";
    mInternals->LastPositionAxisIKMessage = "MoveIt position-axis IK is not initialized.";
    return {};
  }

  try
  {
    const moveit::core::LinkModel* tipModel = mInternals->RobotModelPtr->getLinkModel(tipLink);
    const std::size_t variableCount = mInternals->JointModelGroupPtr->getVariableCount();
    if (!tipModel || variableCount == 0 ||
        (!seedJointValues.empty() && seedJointValues.size() != variableCount))
    {
      mLastMoveItPositionAxisIKTerminationReason = "invalid_input";
      mInternals->LastPositionAxisIKMessage = "MoveIt position-axis IK received an invalid tip or seed.";
      return {};
    }

    vtkNew<vtkMatrix4x4> targetPoseSI;
    targetPoseSI->DeepCopy(targetPose);
    vtkMRMLROS2::ToSI(targetPoseSI);
    const Eigen::Vector3d targetPosition(
      targetPoseSI->GetElement(0, 3),
      targetPoseSI->GetElement(1, 3),
      targetPoseSI->GetElement(2, 3));
    Eigen::Vector3d targetAxis(
      targetPoseSI->GetElement(0, 2),
      targetPoseSI->GetElement(1, 2),
      targetPoseSI->GetElement(2, 2));
    if (!targetPosition.allFinite() || !targetAxis.allFinite() || targetAxis.norm() < 1.0e-12)
    {
      mLastMoveItPositionAxisIKTerminationReason = "invalid_target";
      mInternals->LastPositionAxisIKMessage = "MoveIt position-axis IK target is non-finite or degenerate.";
      return {};
    }
    targetAxis.normalize();
    mLastMoveItPositionAxisIKTerminationReason = "iteration_limit";
    if (avoidCollisions)
    {
      mLastMoveItPositionAxisIKCollisionCheckStatus =
        mInternals->PlanningSceneMonitorPtr ? "not_reached" : "unavailable";
    }

    moveit::core::RobotState state(mInternals->RobotModelPtr);
    state.setToDefaultValues();
    if (!seedJointValues.empty())
    {
      state.setJointGroupPositions(mInternals->JointModelGroupPtr, seedJointValues);
    }
    state.enforceBounds(mInternals->JointModelGroupPtr);
    state.update();

    constexpr double positionToleranceM = 0.00025;
    const double axisToleranceRad = 0.5 * M_PI / 180.0;
    constexpr int maximumIterations = 120;
    constexpr double damping = 1.0e-3;
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(std::max(0.01, timeout));
    const std::vector<std::string>& variableNames =
      mInternals->JointModelGroupPtr->getVariableNames();
    std::vector<double> bestValues;
    double bestPositionError = std::numeric_limits<double>::infinity();
    double bestAxisError = std::numeric_limits<double>::infinity();
    double bestScore = std::numeric_limits<double>::infinity();
    std::vector<double> solution;

    for (int iteration = 0;
         iteration < maximumIterations && std::chrono::steady_clock::now() < deadline;
         ++iteration)
    {
      mLastMoveItPositionAxisIKIterationCount = iteration + 1;
      state.update();
      const Eigen::Isometry3d& currentTransform = state.getGlobalLinkTransform(tipModel);
      const Eigen::Vector3d currentAxis = currentTransform.linear().col(2).normalized();
      const Eigen::Vector3d positionError = targetPosition - currentTransform.translation();
      const double positionErrorM = positionError.norm();
      const double axisErrorRad = std::acos(std::clamp(currentAxis.dot(targetAxis), -1.0, 1.0));
      const double score = std::pow(positionErrorM / positionToleranceM, 2.0) +
        std::pow(axisErrorRad / axisToleranceRad, 2.0);
      if (score < bestScore)
      {
        bestScore = score;
        bestPositionError = positionErrorM;
        bestAxisError = axisErrorRad;
        state.copyJointGroupPositions(mInternals->JointModelGroupPtr, bestValues);
      }

      if (positionErrorM <= positionToleranceM && axisErrorRad <= axisToleranceRad)
      {
        bool collisionFree = true;
        if (avoidCollisions && mInternals->PlanningSceneMonitorPtr)
        {
          try
          {
            planning_scene_monitor::LockedPlanningSceneRO planningScene(
              mInternals->PlanningSceneMonitorPtr);
            if (planningScene)
            {
              moveit::core::RobotState collisionState(planningScene->getCurrentState());
              std::vector<double> values;
              state.copyJointGroupPositions(mInternals->JointModelGroupPtr, values);
              collisionState.setJointGroupPositions(mInternals->JointModelGroupPtr, values);
              collisionState.update();
              collisionFree = !planningScene->isStateColliding(
                collisionState, mInternals->JointModelGroupPtr->getName());
              mLastMoveItPositionAxisIKCollisionCheckStatus =
                collisionFree ? "clear" : "colliding";
            }
            else
            {
              mLastMoveItPositionAxisIKCollisionCheckStatus = "unavailable";
              collisionFree = false;
            }
          }
          catch (...)
          {
            mLastMoveItPositionAxisIKCollisionCheckStatus = "unavailable";
            throw;
          }
        }
        else if (avoidCollisions)
        {
          mLastMoveItPositionAxisIKCollisionCheckStatus = "unavailable";
          collisionFree = false;
        }
        std::vector<double> values;
        state.copyJointGroupPositions(mInternals->JointModelGroupPtr, values);
        mInternals->LastPositionAxisIKBestJointValues = values;
        mInternals->LastPositionAxisIKPositionResidualMm = positionErrorM * 1000.0;
        mInternals->LastPositionAxisIKAxisResidualDeg = axisErrorRad * 180.0 / M_PI;
        if (!collisionFree)
        {
          const bool collisionCheckUnavailable =
            mLastMoveItPositionAxisIKCollisionCheckStatus == "unavailable";
          mLastMoveItPositionAxisIKTerminationReason =
            collisionCheckUnavailable ? "collision_check_unavailable" : "colliding";
          mInternals->LastPositionAxisIKMessage = collisionCheckUnavailable
            ? "Position-axis IK converged within tolerance but collision checking was unavailable."
            : "Position-axis IK converged within tolerance but the endpoint is colliding.";
        }
        else
        {
          mLastMoveItPositionAxisIKTerminationReason = "converged";
          mInternals->LastPositionAxisIKMessage =
            "Position-axis IK converged using J1-J5; axial tool roll was unconstrained.";
          solution = values;
        }
        break;
      }

      Eigen::MatrixXd geometricJacobian;
      if (!state.getJacobian(
            mInternals->JointModelGroupPtr,
            tipModel,
            Eigen::Vector3d::Zero(),
            geometricJacobian) ||
          geometricJacobian.rows() != 6 ||
          geometricJacobian.cols() != static_cast<Eigen::Index>(variableCount))
      {
        mLastMoveItPositionAxisIKTerminationReason = "jacobian_failure";
        mInternals->LastPositionAxisIKMessage = "MoveIt could not compute the position-axis Jacobian.";
        break;
      }

      // Translation contributes three constraints.  Project angular velocity
      // into the plane normal to the requested drill axis; the omitted axial
      // component is the intentionally free housing-roll task dimension.
      const Eigen::Matrix3d axisProjector =
        Eigen::Matrix3d::Identity() - targetAxis * targetAxis.transpose();
      Eigen::MatrixXd taskJacobian(6, variableCount);
      taskJacobian.topRows(3) = geometricJacobian.topRows(3) / positionToleranceM;
      taskJacobian.bottomRows(3) =
        axisProjector * geometricJacobian.bottomRows(3) / axisToleranceRad;
      Eigen::VectorXd taskError(6);
      taskError.head(3) = positionError / positionToleranceM;
      taskError.tail(3) = currentAxis.cross(targetAxis) / axisToleranceRad;
      Eigen::MatrixXd normal = taskJacobian.transpose() * taskJacobian;
      normal.diagonal().array() += damping * damping;
      Eigen::VectorXd delta = normal.ldlt().solve(taskJacobian.transpose() * taskError);
      if (!delta.allFinite())
      {
        mLastMoveItPositionAxisIKTerminationReason = "nonfinite_step";
        mInternals->LastPositionAxisIKMessage = "MoveIt position-axis IK produced a non-finite update.";
        break;
      }

      double stepScale = 1.0;
      for (std::size_t index = 0; index < variableCount; ++index)
      {
        const moveit::core::JointModel* joint =
          mInternals->RobotModelPtr->getJointOfVariable(variableNames[index]);
        const double maximumStep =
          joint && joint->getType() == moveit::core::JointModel::PRISMATIC ? 0.002 : 0.10;
        if (std::abs(delta[static_cast<Eigen::Index>(index)]) > maximumStep)
        {
          stepScale = std::min(
            stepScale,
            maximumStep / std::abs(delta[static_cast<Eigen::Index>(index)]));
        }
      }
      if (delta.norm() * stepScale < 1.0e-12)
      {
        mLastMoveItPositionAxisIKTerminationReason = "stalled";
        mInternals->LastPositionAxisIKMessage = "MoveIt position-axis IK stalled before reaching tolerance.";
        break;
      }
      std::vector<double> values;
      state.copyJointGroupPositions(mInternals->JointModelGroupPtr, values);
      for (std::size_t index = 0; index < variableCount; ++index)
      {
        values[index] += stepScale * delta[static_cast<Eigen::Index>(index)];
      }
      state.setJointGroupPositions(mInternals->JointModelGroupPtr, values);
      state.enforceBounds(mInternals->JointModelGroupPtr);
    }

    if (mLastMoveItPositionAxisIKTerminationReason == "iteration_limit")
    {
      if (mLastMoveItPositionAxisIKIterationCount < maximumIterations)
      {
        mLastMoveItPositionAxisIKTerminationReason = "timeout";
      }
    }
    if (mLastMoveItPositionAxisIKTerminationReason != "converged" &&
        mLastMoveItPositionAxisIKTerminationReason != "colliding")
    {
      mInternals->LastPositionAxisIKBestJointValues = bestValues;
      mInternals->LastPositionAxisIKPositionResidualMm =
        std::isfinite(bestPositionError) ? bestPositionError * 1000.0 : -1.0;
      mInternals->LastPositionAxisIKAxisResidualDeg =
        std::isfinite(bestAxisError) ? bestAxisError * 180.0 / M_PI : -1.0;
      if (mLastMoveItPositionAxisIKTerminationReason == "timeout" ||
          mLastMoveItPositionAxisIKTerminationReason == "iteration_limit")
      {
        std::ostringstream message;
        message << "Position-axis IK did not reach tolerance; best residuals were "
                << mInternals->LastPositionAxisIKPositionResidualMm << " mm and "
                << mInternals->LastPositionAxisIKAxisResidualDeg << " deg.";
        mInternals->LastPositionAxisIKMessage = message.str();
      }
    }

    // This diagnostic is deliberately computed once, after the solve, at its
    // best found state. It must not affect the solve result or iteration path.
    try
    {
      constexpr std::size_t taskVariableCount = 5;
      if (bestValues.size() == taskVariableCount)
      {
        moveit::core::RobotState bestState(mInternals->RobotModelPtr);
        bestState.setToDefaultValues();
        bestState.setJointGroupPositions(mInternals->JointModelGroupPtr, bestValues);
        bestState.update();
        Eigen::MatrixXd geometricJacobian;
        if (bestState.getJacobian(
              mInternals->JointModelGroupPtr,
              tipModel,
              Eigen::Vector3d::Zero(),
              geometricJacobian) &&
            geometricJacobian.rows() == 6 && geometricJacobian.cols() == 5)
        {
          const Eigen::Matrix3d axisProjector =
            Eigen::Matrix3d::Identity() - targetAxis * targetAxis.transpose();
          Eigen::MatrixXd taskJacobian(6, 5);
          taskJacobian.topRows(3) = geometricJacobian.topRows(3) / positionToleranceM;
          taskJacobian.bottomRows(3) =
            axisProjector * geometricJacobian.bottomRows(3) / axisToleranceRad;
          const Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(taskJacobian);
          const Eigen::VectorXd singularValues = decomposition.singularValues();
          if (singularValues.size() == 5 && singularValues.allFinite() &&
              singularValues[0] > 0.0)
          {
            const double ratio = singularValues[4] / singularValues[0];
            if (std::isfinite(ratio))
            {
              mLastMoveItPositionAxisIKConditionRatio = std::clamp(ratio, 0.0, 1.0);
            }
          }
        }
      }
    }
    catch (...)
    {
      mLastMoveItPositionAxisIKConditionRatio = -1.0;
    }
    return solution;
  }
  catch (const std::exception& exception)
  {
    mLastMoveItPositionAxisIKTerminationReason = "exception";
    mInternals->LastPositionAxisIKMessage =
      std::string("MoveIt position-axis IK exception: ") + exception.what();
    return {};
  }
}

std::string vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKMessage() const
{
  return mInternals->LastPositionAxisIKMessage;
}

double vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKPositionResidualMm() const
{
  return mInternals->LastPositionAxisIKPositionResidualMm;
}

double vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKAxisResidualDeg() const
{
  return mInternals->LastPositionAxisIKAxisResidualDeg;
}

std::vector<double> vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKBestJointValues() const
{
  return mInternals->LastPositionAxisIKBestJointValues;
}

std::string vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKTerminationReason() const
{
  return mLastMoveItPositionAxisIKTerminationReason;
}

int vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKIterationCount() const
{
  return mLastMoveItPositionAxisIKIterationCount;
}

std::string vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKCollisionCheckStatus() const
{
  return mLastMoveItPositionAxisIKCollisionCheckStatus;
}

double vtkMRMLROS2RobotNode::GetLastMoveItPositionAxisIKConditionRatio() const
{
  return mLastMoveItPositionAxisIKConditionRatio;
}

std::vector<std::string> vtkMRMLROS2RobotNode::GetMoveItCollidingBodyPairs(
  const std::string& groupName,
  const std::vector<double>& jointValues)
{
  std::vector<std::string> pairs;
  if (!mInternals->RobotModelPtr || !mInternals->PlanningSceneMonitorPtr)
  {
    vtkErrorMacro(<< "GetMoveItCollidingBodyPairs: MoveIt scene is unavailable");
    return pairs;
  }
  const moveit::core::JointModelGroup* joint_group =
    mInternals->RobotModelPtr->getJointModelGroup(groupName);
  if (!joint_group || jointValues.size() != joint_group->getVariableCount())
  {
    vtkErrorMacro(<< "GetMoveItCollidingBodyPairs: invalid group or joint vector");
    return pairs;
  }
  try
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(
      mInternals->PlanningSceneMonitorPtr);
    moveit::core::RobotState state(scene->getCurrentState());
    state.setJointGroupPositions(joint_group, jointValues);
    state.update();
    if (!state.satisfiesBounds(joint_group))
    {
      pairs.emplace_back("__JOINT_BOUNDS__\t__VIOLATION__");
    }
    collision_detection::CollisionRequest request;
    request.group_name = groupName;
    request.contacts = true;
    request.max_contacts = 100;
    request.max_contacts_per_pair = 1;
    collision_detection::CollisionResult result;
    scene->checkCollision(request, result, state);
    for (const auto& contact_entry : result.contacts)
    {
      pairs.push_back(
        contact_entry.first.first + "\t" + contact_entry.first.second);
    }
  }
  catch (const std::exception& exc)
  {
    vtkErrorMacro(<< "GetMoveItCollidingBodyPairs: " << exc.what());
  }
  return pairs;
}

std::vector<std::string> vtkMRMLROS2RobotNode::GetMoveItSceneGeometryDiagnostic(
  const std::string& objectId)
{
  std::vector<std::string> evidence;
  if (!mInternals->RobotModelPtr || !mInternals->PlanningSceneMonitorPtr)
  {
    evidence.emplace_back("scene=unavailable");
    return evidence;
  }
  planning_scene_monitor::LockedPlanningSceneRO scene(
    mInternals->PlanningSceneMonitorPtr);
  if (!scene)
  {
    evidence.emplace_back("scene=unavailable");
    return evidence;
  }
  const auto object = scene->getWorld()->getObject(objectId);
  evidence.emplace_back(std::string("probe_present=") + (object ? "true" : "false"));
  if (object)
  {
    evidence.emplace_back("probe_shapes=" + std::to_string(object->shapes_.size()));
    const auto& position = object->pose_.translation();
    evidence.emplace_back("probe_pose_xyz_m=" + std::to_string(position.x()) + "," +
      std::to_string(position.y()) + "," + std::to_string(position.z()));
  }
  for (const auto* link : mInternals->RobotModelPtr->getLinkModelsWithCollisionGeometry())
  {
    evidence.emplace_back("link=" + link->getName() + ":shapes=" +
      std::to_string(link->getShapes().size()));
  }
  return evidence;
}

std::vector<std::string> vtkMRMLROS2RobotNode::GetMoveItWholeRobotCollidingBodyPairs(
  const std::string& groupName,
  const std::vector<double>& jointValues)
{
  std::vector<std::string> pairs;
  if (!mInternals->RobotModelPtr || !mInternals->PlanningSceneMonitorPtr)
    return pairs;
  const auto* group = mInternals->RobotModelPtr->getJointModelGroup(groupName);
  if (!group || jointValues.size() != group->getVariableCount())
    return pairs;
  planning_scene_monitor::LockedPlanningSceneRO scene(mInternals->PlanningSceneMonitorPtr);
  if (!scene)
    return pairs;
  moveit::core::RobotState state(scene->getCurrentState());
  state.setJointGroupPositions(group, jointValues);
  state.update();
  collision_detection::CollisionRequest request;
  request.contacts = true;
  request.max_contacts = 100;
  request.max_contacts_per_pair = 1;
  collision_detection::CollisionResult result;
  scene->checkCollision(request, result, state);
  for (const auto& contact : result.contacts)
    pairs.push_back(contact.first.first + "\t" + contact.first.second);
  return pairs;
}


bool vtkMRMLROS2RobotNode::SetupKDLIKWithLimits(void)
{
  try {
    auto rootAndTip = FindRootAndTipLinks();
    if (rootAndTip.size() < 2 || rootAndTip[0].empty() || rootAndTip[1].empty()) {
      vtkErrorMacro(<< "setupKDLIKWithLimits: Failed to determine root/tip links from URDF");
      return false;
    }
    const std::string& defaultRoot = rootAndTip[0];
    const std::string& defaultTip = rootAndTip[1];
    vtkInfoMacro(<< "Auto KDL setup with limits. Root: '" << defaultRoot
                 << "' Tip: '" << defaultTip << "'");

    // Create kdltree using KDL parser
    KDL::Tree kdlTree;
    if (!kdl_parser::treeFromString(mInternals->mRobotDescription, kdlTree)) {
      vtkErrorMacro(<< "setupKDLIKWithLimits: Failed to parse URDF to KDL tree");
      return false;
    }

    // Extract chain from tree
    mInternals->KDLChain = std::make_unique<KDL::Chain>();
    if (!kdlTree.getChain(defaultRoot, defaultTip, *mInternals->KDLChain)) {
      vtkErrorMacro(<< "setupKDLIKWithLimits: Failed to extract chain from " << defaultRoot << " to " << defaultTip);
      return false;
    }else{
      vtkInfoMacro(<< "setupKDLIKWithLimits: Successfully extracted chain from " << defaultRoot << " to " << defaultTip);
    }

    // Get number of joints in KDL chain
    unsigned int nj = mInternals->KDLChain->getNrOfJoints();

    // Initialize joint limits arrays
    mInternals->KDLJointMin = KDL::JntArray(nj);
    mInternals->KDLJointMax = KDL::JntArray(nj);

    // Extract joint limits from mURDFModel
    unsigned int joint_idx = 0;
    for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
      const KDL::Segment& segment = mInternals->KDLChain->getSegment(i);
      const KDL::Joint& joint = segment.getJoint();
      
      if (joint.getType() != KDL::Joint::None) {
        std::string joint_name = joint.getName();
        // Print joint name for debugging
        vtkInfoMacro(<< "Processing joint: " << joint_name);
        auto urdf_joint = mInternals->mURDFModel.getJoint(joint_name);
          
          // Check if joint is continuous
          if (urdf_joint && urdf_joint->type == urdf::Joint::CONTINUOUS) {
              mInternals->KDLJointMin(joint_idx) = -2 * M_PI; 
              mInternals->KDLJointMax(joint_idx) = 2 * M_PI;
              vtkInfoMacro(<< "  Joint " << joint_name << " is CONTINUOUS. Setting wide limits [-2pi, 2pi].");
          } 
          else if (urdf_joint && urdf_joint->limits) {
            // Otherwise, get limits if available
            mInternals->KDLJointMin(joint_idx) = urdf_joint->limits->lower;
            mInternals->KDLJointMax(joint_idx) = urdf_joint->limits->upper;
          } 
          else {
            // Default limits if not specified
            mInternals->KDLJointMin(joint_idx) = -M_PI;
            mInternals->KDLJointMax(joint_idx) = M_PI;
          }
          joint_idx++;
        }
      }

    // Create solvers (NR_JL with joint limits)
    mInternals->KDLFkSolver = std::make_unique<KDL::ChainFkSolverPos_recursive>(*mInternals->KDLChain);
    mInternals->KDLIkSolverVel = std::make_unique<KDL::ChainIkSolverVel_pinv>(*mInternals->KDLChain);
    mInternals->KDLIkSolverJL = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
        *mInternals->KDLChain, mInternals->KDLJointMin, mInternals->KDLJointMax, *mInternals->KDLFkSolver, *mInternals->KDLIkSolverVel, 100, 1e-6);

    mInternals->KDLRootLink = defaultRoot;
    mInternals->KDLTipLink = defaultTip;
    mInternals->KDLUseJointLimits = true;

    // Print joint limits for verification
    vtkInfoMacro(<< "Joint limits for KDL IK solver:");
    for (unsigned int j = 0; j < nj; j++) {
      vtkInfoMacro(<< "  Joint " << j << ": [" << mInternals->KDLJointMin(j) << ", " << mInternals->KDLJointMax(j) << "]");
    }

    vtkInfoMacro(<< "setupKDLIKWithLimits: Successfully initialized KDL IK solver (NR_JL) for chain " 
                 << defaultRoot << " -> " << defaultTip 
                 << " with " << mInternals->KDLChain->getNrOfJoints() << " joints and joint limits");
 
    return true;
  }
  catch (const std::exception& e) {
    vtkErrorMacro(<< "setupKDLIKWithLimits: exception - " << e.what());
    return false;
  }
}

std::vector<std::string> vtkMRMLROS2RobotNode::FindRootAndTipLinks() const
{
  auto urdfRoot = mInternals->mURDFModel.getRoot();
  if (!urdfRoot) {
    vtkErrorMacro(<< "FindRootAndTipLinks: URDF root link is null");
    return {};
  }

  // Auto-detect structural root: first child of URDF root with no inertia AND has children
  // This finds the kinematic root (e.g., "base_link") rather than the physical tree root
  auto rootLink = urdfRoot;
  for (const auto& childLink : urdfRoot->child_links) {
    if (childLink && !childLink->inertial && !childLink->child_links.empty()) {
      rootLink = childLink;
      break;
    }
  }

  std::string rootLinkName = rootLink->name;
  std::string tipLinkName = rootLinkName;

  struct TraversalState
  {
    std::shared_ptr<const urdf::Link> Link;
    int MovableJointCount;
    int Depth;
  };

  int bestMovableJointCount = -1;
  int bestDepth = -1;

  std::queue<TraversalState> bfsQueue;
  bfsQueue.push({rootLink, 0, 0});

  while (!bfsQueue.empty()) {
    auto state = bfsQueue.front();
    bfsQueue.pop();

    if (!state.Link) {
      continue;
    }

    if (state.Link->child_links.empty()) {
      if (state.MovableJointCount > bestMovableJointCount ||
          (state.MovableJointCount == bestMovableJointCount && state.Depth > bestDepth)) {
        bestMovableJointCount = state.MovableJointCount;
        bestDepth = state.Depth;
        tipLinkName = state.Link->name;
      }
      continue;
    }

    for (const auto& childLink : state.Link->child_links) {
      int movableJointCount = state.MovableJointCount;
      if (childLink && childLink->parent_joint && childLink->parent_joint->type != urdf::Joint::FIXED) {
        movableJointCount++;
      }
      bfsQueue.push({childLink, movableJointCount, state.Depth + 1});
    }
  }

  return {rootLinkName, tipLinkName};
}


std::vector<double> vtkMRMLROS2RobotNode::ComputeKDLIK(vtkMatrix4x4* targetPose,
                                                       const std::vector<double>& seedJointValues)
{
  if (!targetPose) {
    vtkErrorMacro(<< "ComputeKDLIK: target pose is null");
    return {};
  }

  if (!mInternals->KDLUseJointLimits && !mInternals->KDLIkSolver) {
    vtkErrorMacro(<< "ComputeKDLIK: KDL IK solver not initialized. Call setupKDLIK or setupKDLIKWithLimits first");
    return {};
  }

  if (mInternals->KDLUseJointLimits && !mInternals->KDLIkSolverJL) {
    vtkErrorMacro(<< "ComputeKDLIK: KDL IK solver with joint limits not initialized. Call setupKDLIKWithLimits first");
    return {};
  }

  try {
    // Convert vtkMatrix4x4 to KDL Frame
    vtkNew<vtkMatrix4x4> targetPoseSI;
    targetPoseSI->DeepCopy(targetPose);
    vtkMRMLROS2::ToSI(targetPoseSI);

    KDL::Frame targetFrame;
    
    // Set position (convert mm to m)
    targetFrame.p.x(targetPoseSI->GetElement(0, 3));
    targetFrame.p.y(targetPoseSI->GetElement(1, 3));
    targetFrame.p.z(targetPoseSI->GetElement(2, 3));
    
    // Set rotation
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        targetFrame.M(i, j) = targetPoseSI->GetElement(i, j);
      }
    }

    // Setup seed configuration
    KDL::JntArray qSeed(mInternals->KDLChain->getNrOfJoints());
    if (!seedJointValues.empty() && seedJointValues.size() == mInternals->KDLChain->getNrOfJoints()) {
      for (size_t i = 0; i < seedJointValues.size(); i++) {
        qSeed(i) = seedJointValues[i];
        // vtkInfoMacro(<< "Seed joint " << i << ": " << qSeed(i));
      }
    } else {
      // Use zeros as default seed
      vtkInfoMacro(<< "ComputeKDLIK: Using zero joint angles as seed");
      qSeed.data.setZero();
    }

    // Solve IK
    KDL::JntArray qSolution(mInternals->KDLChain->getNrOfJoints());
    int result;
    
    if (mInternals->KDLUseJointLimits) {
      result = mInternals->KDLIkSolverJL->CartToJnt(qSeed, targetFrame, qSolution);
    } else {
      result = mInternals->KDLIkSolver->CartToJnt(qSeed, targetFrame, qSolution);
    }

    if (result < 0) {
      // vtkWarningMacro(<< "ComputeKDLIK: IK solution not found (error code: " << result << ")");
      return {};
    }

    std::vector<double> solution;
    solution.reserve(qSolution.rows());
    for (unsigned int i = 0; i < qSolution.rows(); i++) {
      solution.push_back(qSolution(i));
    }
    return solution;
  }
  catch (const std::exception& e) {
    vtkErrorMacro(<< "ComputeKDLIK: exception - " << e.what());
    return {};
  }
}


std::vector<std::string> vtkMRMLROS2RobotNode::GetSegments()
{
  std::vector<std::string> segmentNames;
  if (!mInternals->KDLChain) {
    vtkWarningMacro(<< "GetSegments: KDL chain not initialized");
    return segmentNames;
  }
  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Segment& segment = mInternals->KDLChain->getSegment(i);
    segmentNames.push_back(segment.getName());
  }
  return segmentNames;
}

std::vector<std::string> vtkMRMLROS2RobotNode::GetJoints()
{
  std::vector<std::string> jointNames;
  if (!mInternals->KDLChain) {
    vtkWarningMacro(<< "GetJoints: KDL chain not initialized");
    return jointNames;
  }
  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Segment& segment = mInternals->KDLChain->getSegment(i);
    const KDL::Joint& joint = segment.getJoint();
    if (joint.getType() != KDL::Joint::None) {
      jointNames.push_back(joint.getName());
    }
  }
  return jointNames;
}

std::vector<double> vtkMRMLROS2RobotNode::GetJointLowerPositionLimits()
{
  std::vector<double> lower;
  if (!mInternals->KDLChain) {
    vtkWarningMacro(<< "GetJointLowerPositionLimits: KDL chain not initialized");
    return lower;
  }
  int joint_idx = 0;
  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Joint& joint = mInternals->KDLChain->getSegment(i).getJoint();
    if (joint.getType() != KDL::Joint::None) {
      lower.push_back(mInternals->KDLJointMin(joint_idx));
      joint_idx++;
    }
  }
  return lower;
}

std::vector<double> vtkMRMLROS2RobotNode::GetJointUpperPositionLimits()
{
  std::vector<double> upper;
  if (!mInternals->KDLChain) {
    vtkWarningMacro(<< "GetJointUpperPositionLimits: KDL chain not initialized");
    return upper;
  }
  int joint_idx = 0;
  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Joint& joint = mInternals->KDLChain->getSegment(i).getJoint();
    if (joint.getType() != KDL::Joint::None) {
      upper.push_back(mInternals->KDLJointMax(joint_idx));
      joint_idx++;
    }
  }
  return upper;
}

std::vector<double> vtkMRMLROS2RobotNode::GetJointVelocityLimits()
{
  std::vector<double> vel;
  if (!mInternals->KDLChain) {
    vtkWarningMacro(<< "GetJointVelocityLimits: KDL chain not initialized");
    return vel;
  }
  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Joint& joint = mInternals->KDLChain->getSegment(i).getJoint();
    if (joint.getType() != KDL::Joint::None) {
      auto urdf_joint = mInternals->mURDFModel.getJoint(joint.getName());
      double v = 0.0;
      if (urdf_joint && urdf_joint->limits) {
        v = urdf_joint->limits->velocity;
      }
      vel.push_back(v);
    }
  }
  return vel;
}

std::vector<std::string> vtkMRMLROS2RobotNode::GetJointTypes()
{
  std::vector<std::string> types;
  if (!mInternals->KDLChain) {
    vtkWarningMacro(<< "GetJointTypes: KDL chain not initialized");
    return types;
  }
  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Joint& joint = mInternals->KDLChain->getSegment(i).getJoint();
    if (joint.getType() == KDL::Joint::None) continue;
    auto urdf_joint = mInternals->mURDFModel.getJoint(joint.getName());
    std::string type_str = "revolute";
    if (urdf_joint) {
      switch (urdf_joint->type) {
        case urdf::Joint::PRISMATIC:   type_str = "prismatic";  break;
        case urdf::Joint::CONTINUOUS:  type_str = "continuous"; break;
        case urdf::Joint::REVOLUTE:    type_str = "revolute";   break;
        default:                       type_str = "revolute";   break;
      }
    }
    types.push_back(type_str);
  }
  return types;
}

vtkMatrix4x4* vtkMRMLROS2RobotNode::ComputeLocalTransform(const std::vector<double>& jointValues, vtkMatrix4x4* outTransform, const std::string& linkName)
{
  if (!outTransform || !mInternals->KDLChain) return nullptr;

  // 1. Find the Segment and its Joint Index
  size_t segmentIndex = 0;
  bool found = false;
  size_t kdlJointIndex = 0; // Tracks which "q" index corresponds to this segment

  for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
    const KDL::Segment& seg = mInternals->KDLChain->getSegment(i);
    
    // Count moving joints up to this point to find the correct 'q' index
    if (seg.getJoint().getType() != KDL::Joint::None) {
        if (seg.getName() == linkName) {
            segmentIndex = i;
            found = true;
            break;
        }
        kdlJointIndex++;
    } else if (seg.getName() == linkName) {
        // Found it, but it's a fixed joint (no q index increment)
        segmentIndex = i;
        found = true;
        break;
    }
  }

  if (!found) {
     vtkErrorMacro(<< "Link '" << linkName << "' not found.");
     return nullptr;
  }

  // 2. Get the specific joint angle for this segment
  double q_val = 0.0;
  const KDL::Segment& targetSeg = mInternals->KDLChain->getSegment(segmentIndex);
  
  if (targetSeg.getJoint().getType() != KDL::Joint::None) {
      if (kdlJointIndex < jointValues.size()) {
          q_val = jointValues[kdlJointIndex];
      } else if (targetSeg.getJoint().getName()
                   == "pneumatic_spindle-Copy_Revolute-6"
                 && jointValues.size() + 1 == mInternals->KDLChain->getNrOfJoints()) {
          // The visual KDL branch retains the external air-rotor joint for
          // mesh/TF compatibility.  Planning callers now provide only J1-J5;
          // the uncontrolled spindle contributes no commanded pose and is
          // therefore evaluated at its neutral visual angle.
          q_val = 0.0;
      } else {
          vtkErrorMacro(<< "Joint index out of bounds.");
          return nullptr;
      }
  }

  // 3. Compute LOCAL Pose (Parent -> Child)
  // This is what Slicer needs for nested hierarchies.
  KDL::Frame localFrame = targetSeg.pose(q_val);

  // 4. Convert to VTK Matrix
  outTransform->Identity();
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      outTransform->SetElement(r, c, localFrame.M(r, c));
    }
    // Set translation (meters)
    outTransform->SetElement(r, 3, localFrame.p(r));
  }
  // Scale Meters -> Millimeters
  vtkMRMLROS2::FromSI(outTransform);

  return outTransform;
}

vtkMatrix4x4* vtkMRMLROS2RobotNode::ComputeKDLFK(const std::vector<double>& jointValues, vtkMatrix4x4* outTransform, const std::string& linkName)
{
  if (!outTransform) {
    vtkErrorMacro(<< "ComputeKDLFK: output transform is null");
    return nullptr;
  }
  // Step 6 uses the fixed, non-spinning TCP sibling of J6.  Its MoveIt group
  // has five variables, while the legacy visual KDL chain still contains the
  // downstream six-joint burr branch for the expert display.  Evaluate the
  // canonical planning frame through the authoritative MoveIt state instead
  // of trying to feed a five-value vector into that six-joint visual chain.
  if (linkName == "dentobot_drill_tcp" && mInternals->RobotModelPtr &&
      mInternals->JointModelGroupPtr &&
      (jointValues.size() == mInternals->JointModelGroupPtr->getVariableCount() ||
       jointValues.size() == mInternals->JointModelGroupPtr->getVariableCount() + 1)) {
    try {
      // Accept one legacy six-value visual vector at this compatibility
      // boundary, but never pass its external-spindle slot to MoveIt.  The
      // canonical TCP is upstream of J6, so the historical sixth value has no
      // effect on this FK result.
      std::vector<double> planning_joint_values(
        jointValues.begin(),
        jointValues.begin() + mInternals->JointModelGroupPtr->getVariableCount());
      moveit::core::RobotState planning_state(mInternals->RobotModelPtr);
      planning_state.setJointGroupPositions(mInternals->JointModelGroupPtr, planning_joint_values);
      planning_state.update();
      const Eigen::Isometry3d transform = planning_state.getGlobalLinkTransform(linkName);
      outTransform->Identity();
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          outTransform->SetElement(row, column, transform.linear()(row, column));
        }
        outTransform->SetElement(row, 3, transform.translation()(row));
      }
      vtkMRMLROS2::FromSI(outTransform);
      return outTransform;
    }
    catch (const std::exception& exception) {
      vtkErrorMacro(<< "ComputeKDLFK: canonical planning TCP FK failed - " << exception.what());
      return nullptr;
    }
  }
  if (!mInternals->KDLChain || !mInternals->KDLFkSolver) {
    vtkWarningMacro(<< "ComputeKDLFK: KDL chain or FK solver not initialized");
    return nullptr;
  }
  if (jointValues.size() != mInternals->KDLChain->getNrOfJoints()) {
    vtkErrorMacro(<< "ComputeKDLFK: expected " << mInternals->KDLChain->getNrOfJoints()
                  << " joint values but got " << jointValues.size());
    return nullptr;
  }

  unsigned int segmentIndex = mInternals->KDLChain->getNrOfSegments() - 1; 

  if (!linkName.empty()) {
    bool found = false;
    for (unsigned int i = 0; i < mInternals->KDLChain->getNrOfSegments(); i++) {
      if (mInternals->KDLChain->getSegment(i).getName() == linkName) {
        segmentIndex = i;
        found = true;
        break;
      }
    }
    if (!found) {
      vtkErrorMacro(<< "ComputeKDLFK: link '" << linkName << "' not found in KDL chain");
      return nullptr;
    }
  }

  KDL::JntArray q(mInternals->KDLChain->getNrOfJoints());
  for (unsigned int i = 0; i < q.rows(); i++) {
    q(i) = jointValues[i];
  }


  KDL::Frame frame;
  // KDL uses 1-based indexing for 'segmentNr', so we add 1 to our 0-based index.
  int result = mInternals->KDLFkSolver->JntToCart(q, frame, segmentIndex + 1);
  
  if (result < 0) {
    vtkErrorMacro(<< "ComputeKDLFK: KDL FK failed with error code " << result);
    return nullptr;
  }

  

  outTransform->Identity();

  // Copy Rotation
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      outTransform->SetElement(r, c, frame.M(r, c));
    }
  }

  // Copy Translation (Applying scale)
  outTransform->SetElement(0, 3, frame.p.x());
  outTransform->SetElement(1, 3, frame.p.y());
  outTransform->SetElement(2, 3, frame.p.z());
  vtkMRMLROS2::FromSI(outTransform);

  return outTransform;
}


void vtkMRMLROS2RobotNode::UpdateScene(vtkMRMLScene *scene)
{
  Superclass::UpdateScene(scene);
  int nbNodeRefs = this->GetNumberOfNodeReferences("node");
  vtkMRMLROS2NodeNode * rosNode = nullptr;
  if (nbNodeRefs == 0) {
    // assigned to the default ROS node
    rosNode = vtkMRMLROS2NodeNode::SafeDownCast(
      scene->GetFirstNodeByName("ros2:node:slicer"));
  } else if (nbNodeRefs == 1) {
    rosNode = vtkMRMLROS2NodeNode::SafeDownCast(
      this->GetNthNodeReference("node", 0));
  } else {
    vtkErrorMacro(<< "UpdateScene: more than one ROS2 node reference defined for robot \"" << GetName() << "\"");
    return;
  }

  if (!rosNode) {
    vtkErrorMacro(<< "UpdateScene: ROS2 node unavailable. Unable to set reference for robot \"" << GetName() << "\"");
    return;
  }

  bool alreadyReferenced = false;
  const int robotCount = rosNode->GetNumberOfNodeReferences("robot");
  for (int index = 0; index < robotCount; ++index) {
    const char * referenceId = rosNode->GetNthNodeReferenceID("robot", index);
    if (referenceId && this->GetID() && std::string(referenceId) == this->GetID()) {
      alreadyReferenced = true;
      break;
    }
  }
  if (!alreadyReferenced) {
    rosNode->SetNthNodeReferenceID("robot", robotCount, this->GetID());
  }
  this->SetNodeReferenceID("node", rosNode->GetID());
  mMRMLROS2Node = rosNode;
}

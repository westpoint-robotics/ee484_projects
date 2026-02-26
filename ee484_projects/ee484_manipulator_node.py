#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
import threading
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from math import cos, sin, pi, sqrt, atan2
import traceback
import time
import json

# ROS Messages and Actions
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from sensor_msgs.msg import JointState

# URDF and Kinematics related imports
from ament_index_python.packages import get_package_share_directory
from urdf_parser_py.urdf import URDF

# === SVG Parsing Imports ===
import xml.etree.ElementTree as ET
from svg.path import parse_path, Line, Arc, CubicBezier, QuadraticBezier
import matplotlib.pyplot as plt # Keep for optional plotting

# === Kinematics Functions  ===
# --- START Kinematics Functions ---

def homogeneous_transform(translation, rpy):
    """
    Calculates a 4x4 Homogeneous Transformation Matrix from translation and rotation.

    WHAT:
    This function creates a standard 4x4 matrix used in robotics and 3D graphics
    to represent the position and orientation (pose) of an object (like a robot link)
    relative to a reference coordinate frame. It combines both rotation and
    translation into a single matrix operation.

    WHY:
    Using a single 4x4 matrix makes it easy to:
    1. Chain multiple transformations together (e.g., link1 relative to base,
       link2 relative to link1) by simply multiplying their matrices.
    2. Apply the transformation to points (represented as 4x1 vectors [x, y, z, 1])
       using a single matrix multiplication.
    3. Represent both rotation and translation compactly.

    HOW:
    1. Input: Takes a `translation` vector [x, y, z] and `rpy` (Roll, Pitch, Yaw)
       Euler angles [roll, pitch, yaw] in radians.
    2. Rotation Matrices: Calculates the individual 3x3 rotation matrices for
       rotation around the X-axis (Roll), Y-axis (Pitch), and Z-axis (Yaw) using
       standard trigonometric formulas (cos, sin).
    3. Combined Rotation: Multiplies the individual rotation matrices in a specific
       order (here: Z * Y * X, which corresponds to rotating around the fixed world
       Z, then Y, then X axis, or equivalently, intrinsic rotation around moving
       X, then Y, then Z axes). The order is critical and defines the convention used.
       The result is a single 3x3 matrix `R` representing the combined orientation.
    4. Assemble 4x4 Matrix:
       - Creates a 4x4 identity matrix (`np.eye(4)`). An identity matrix has 1s on
         the diagonal and 0s elsewhere; it represents no change in pose.
       - Places the calculated 3x3 rotation matrix `R` into the top-left 3x3
         block of the 4x4 matrix `T`. This part handles the orientation.
       - Places the input 3x1 `translation` vector into the top-right 3x1 column
         (elements T[0,3], T[1,3], T[2,3]). This part handles the position.
       - The bottom row remains [0, 0, 0, 1]. This is standard for affine
         transformations in homogeneous coordinates and ensures matrix multiplications
         work correctly for chaining poses and transforming points.

    Args:
        translation (list or np.array): A 3-element list or array [x, y, z]
                                        representing the displacement.
        rpy (list or np.array): A 3-element list or array [roll, pitch, yaw]
                                representing the rotation angles in radians.

    Returns:
        np.ndarray: A 4x4 numpy array representing the homogeneous transformation matrix.
                    [[R R R Tx]
                     [R R R Ty]
                     [R R R Tz]
                     [0 0 0 1 ]]
    """
    roll, pitch, yaw = rpy
    # Rotation matrix around X (Roll)
    R_x = np.array([[1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll) , cos(roll) ]]) # REDACTED: Complete the Rx matrix
    # Rotation matrix around Y (Pitch)
    R_y = np.array([[cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [ -sin(pitch) , 0, cos(pitch) ]]) # REDACTED: Complete the Ry matrix
    # Rotation matrix around Z (Yaw)
    R_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [ sin(yaw) , cos(yaw), 0],   # REDACTED: Complete the Rz matrix
                    [0, 0, 1]])

    # Combine rotations: R = Rz * Ry * Rx (order matters!)
    R = R_z @ R_y @ R_x

    # Create 4x4 identity matrix
    T = np.eye(4)
    # Place rotation matrix in top-left 3x3 corner
    T[:3, :3] = R # REDACTED: Assign the combined rotation matrix R
    # Place translation vector in top-right 3x1 column
    T[:3, 3] = translation # REDACTED: Assign the translation vector

    return T

def rotation_about_axis(axis, angle):
    """
    Calculates a 4x4 Homogeneous Transformation Matrix for rotation about an arbitrary axis.

    WHAT:
    Creates a 4x4 matrix representing ONLY a rotation by a given `angle` around a
    specified 3D `axis` vector. The translation part of this matrix will be zero.

    WHY:
    This is fundamental for representing joint rotations in robots, especially for
    revolute (rotating) joints. The URDF (Universal Robot Description Format)
    defines joints with an axis of rotation, which might not be simply X, Y, or Z.
    This function allows calculating the rotation caused by such a joint.

    HOW:
    1. Input: Takes an `axis` vector [ax, ay, az] and an `angle` in radians.
    2. Normalize Axis: The input `axis` vector is normalized (scaled) so that it
       becomes a unit vector (length = 1). This is required by the rotation formula.
       Normalization is done by dividing the vector by its magnitude (norm).
    3. Rodrigues' Rotation Formula: Uses the components of the normalized axis
       (x, y, z) and the trigonometric functions (cos, sin) of the `angle` to
       directly compute the elements of the 3x3 rotation matrix `R`. This formula
       (Rodrigues' rotation formula or a derived form) describes the rotation of
       a vector around an arbitrary axis.
    4. Assemble 4x4 Matrix:
       - Creates a 4x4 identity matrix (`np.eye(4)`).
       - Places the calculated 3x3 rotation matrix `R` into the top-left 3x3 block.
       - The translation part (top-right column) remains [0, 0, 0].
       - The bottom row remains [0, 0, 0, 1].

    Args:
        axis (list or np.array): A 3-element list or array [ax, ay, az] defining
                                 the axis of rotation. Does not need to be normalized.
        angle (float): The angle of rotation in radians.

    Returns:
        np.ndarray: A 4x4 numpy array representing the pure rotation transformation.
                    [[R R R 0]
                     [R R R 0]
                     [R R R 0]
                     [0 0 0 1]]
    """
    # Ensure axis is a numpy array and normalize it to unit length
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis) # REDACTED: Normalize the axis vector (use np.linalg.norm)
    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    t = 1 - c # Useful intermediate term: 1 - cos(angle)

    # Rodrigues' rotation formula elements for the 3x3 matrix
    R = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*x*y - s*x], # REDACTED: Element R[1,2]
        [t*x*z - s*y,  t*x*y + s*x,  t*z*z + c]  # REDACTED: Element R[2,1]
    ])

    # Create 4x4 identity matrix
    T = np.eye(4)
    # Place the 3x3 rotation matrix R into the top-left corner
    T[:3, :3] = R # REDACTED: Assign the rotation matrix R
    # The translation part remains zero for a pure rotation

    return T

def build_child_map(robot):
    return {j.child: j for j in robot.joints}

def get_chain(robot, base_link, tip_link):
    """
    Finds the sequence of movable joints connecting two links in a robot model.

    WHAT:
    Traces the kinematic structure of a robot (defined by a URDF object)
    to find the ordered list of joints that lie on the path starting from a
    specified `base_link` and ending at a specified `tip_link`.

    WHY:
    Kinematic calculations like Forward Kinematics (calculating end-effector pose
    from joint angles) and Jacobian computation (relating joint velocities to
    end-effector velocities) operate on a specific chain of joints. This function
    extracts that chain from the overall robot description.

    HOW:
    1. Input: Takes the `robot` object (parsed from URDF), the name of the
       starting link (`base_link`), and the name of the ending link (`tip_link`).
    2. Build Child Map (Helper): Creates a dictionary (`child_map`) where keys are
       child link names and values are the joint objects that connect them to their
       parent. This allows quick lookup of the joint leading *to* a link.
    3. Trace Backwards:
       - Starts with the `current_link` set to the desired `tip_link`.
       - Enters a loop that continues as long as `current_link` is not the `base_link`.
       - Inside the loop:
         - Uses the `child_map` to find the `joint` whose `child` attribute matches
           the `current_link`. This is the joint immediately "upstream" from the
           current link.
         - Includes error handling: if the link isn't found in the map, it means
           the chain is broken or the link names are incorrect, so an error is raised.
         - Appends the found `joint` object to the `chain` list.
         - Updates `current_link` to be the `parent` link of the found `joint`.
           This moves one step up the chain towards the base.
         - Checks if the parent link is the `base_link`; if so, the loop can stop.
         - Includes error handling for joints missing parent links (invalid URDF).
    4. Validate and Reverse:
       - After the loop, checks if the chain is valid (not empty and the last joint
         found indeed connects to the `base_link`). Raises an error if not.
       - Reverses the `chain` list. Since the tracing was done backward (tip to base),
         reversing puts the joints in the correct base-to-tip order needed for
         standard forward kinematics calculations...i.e., the tip/gripper/end-effector's pose
         after having set joints from base to the joint before the tip.

    Args:
        robot (URDF): The robot model object, typically loaded from a URDF file
                      using a library like `urdf_parser_py`.
        base_link (str): The name of the starting link of the kinematic chain.
        tip_link (str): The name of the ending link of the kinematic chain.

    Returns:
        list: A list of joint objects (from the `robot` model) ordered from the
              `base_link` towards the `tip_link`.

    Raises:
        ValueError: If a valid chain cannot be constructed (e.g., links not found,
                    disconnected structure, `tip_link` is the same as `base_link` implicitly).
    """
    # Build a map from child link name to the joint object that connects to it
    child_map = {j.child: j for j in robot.joints} # This helper is kept intact
    chain = [] # Initialize the list to store the joints in the chain
    current_link = tip_link # Start tracing from the desired end link

    # Loop backwards from tip_link until we reach the base_link
    while current_link != base_link:
        # Find the joint whose child is the current_link
        joint = child_map.get(current_link) # REDACTED: Get the joint from child_map using current_link

        # Error handling: If no joint is found leading to this link
        if joint is None:
            raise ValueError(f"Kinematic chain error: Link '{current_link}' not found as a child "
                             f"in the path from '{tip_link}' back towards '{base_link}'. "
                             "Check link names and URDF structure.")

        # Add the found joint to our list...chain list initialized above
        chain.append(joint) # REDACTED: Append the found joint to the chain list

        # Check if the parent of this joint is the base_link we are looking for
        if joint.parent == base_link:
            break # We've successfully reached the base

        # Move to the next link up the chain (towards the base)
        current_link = joint.parent # REDACTED: Update current_link to the joint's parent

        # Error handling: If a joint has no parent defined (shouldn't happen in valid URDF)
        if current_link is None:
            raise ValueError(f"URDF structure error: Joint '{joint.name}' has no parent link defined.")

    # Validation after the loop
    # Check if the chain is empty (e.g., if base_link == tip_link initially)
    if not chain:
         raise ValueError(f"Kinematic chain is empty. Is tip_link ('{tip_link}') the "
                          f"same as base_link ('{base_link}')?")
    # Check if the loop terminated correctly by reaching the base_link
    if chain[-1].parent != base_link:
         raise ValueError(f"Chain construction failed. Could not trace back to base_link "
                          f"'{base_link}'. Last parent found was '{chain[-1].parent}'.")

    # Reverse the chain: We traced tip->base, but FK needs base->tip order
    chain.reverse() # REDACTED: Reverse the chain list in place
    return chain

def forward_kinematics(robot, joint_positions_dict, base_link, tip_link):
    """
    Calculates the pose of the tip_link relative to the base_link given joint values.

    WHAT:
    Computes the Forward Kinematics (FK) for a specific kinematic chain within the
    robot. Given the current angle or position of each movable joint in the chain,
    it calculates the resulting 3D position and orientation (as a 4x4 matrix)
    of the `tip_link` frame with respect to the `base_link` frame.

    WHY:
    FK is essential to know where the robot's end-effector (or any other part) is
    located in space based on its current joint configuration. This is needed for:
    - Collision checking.
    - Visualizing the robot's state.
    - Providing the current pose feedback for Inverse Kinematics solvers.
    - Planning paths in joint space and verifying them in task space.

    HOW:
    1. Input: Takes the `robot` model, a dictionary `joint_positions_dict` mapping
       joint names to their current values (angles for revolute, displacement for
       prismatic), and the `base_link` and `tip_link` names.
    2. Get Chain: Calls `get_chain` to get the ordered list of joints connecting
       `base_link` to `tip_link`.
    3. Initialize Transformation: Starts with the overall transformation `T` as the
       4x4 identity matrix. This represents the pose of the `base_link` relative
       to itself (zero translation, zero rotation).
    4. Iterate Through Chain (Base to Tip): Loops through each `joint` in the chain:
       a. Fixed Transform: Calculates the static transformation (`T_fixed`) from the
          joint's parent link's frame to the joint's own origin frame. This is
          defined by the `<origin>` tag (xyz translation, rpy rotation) within the
          `<joint>` element in the URDF. Uses `homogeneous_transform`.
       b. Variable Transform: Calculates the transformation (`T_variable`) caused
          by the joint's *motion*, based on its type and current value:
          - Retrieves the joint's current value from `joint_positions_dict`. If the
            joint is not in the dictionary (e.g., a fixed joint in the chain),
            it defaults to 0.0.
          - If the joint is 'revolute' or 'continuous': Uses `rotation_about_axis`
            with the joint's defined `axis` and the retrieved `angle`.
          - If the joint is 'prismatic': Creates a pure translation matrix along
            the joint's defined `axis` by the retrieved displacement `disp`.
          - If the joint is 'fixed' or has no axis defined: `T_variable` remains
            the identity matrix (no motion).
       c. Chain Multiplication: Updates the overall transformation `T` by sequentially
          multiplying by the fixed and then the variable transforms for the current
          joint: `T = T @ T_fixed @ T_variable`.
          The order means: start from the pose `T` accumulated so far (relative to
          base), apply the fixed offset `T_fixed` to get to the joint's origin,
          then apply the motion `T_variable` relative to that origin.
    5. Final Pose: After iterating through all joints in the chain, the final `T`
       matrix represents the pose of the `tip_link` frame expressed in the
       coordinates of the `base_link` frame.

    Args:
        robot (URDF): The robot model object.
        joint_positions_dict (dict): A dictionary where keys are joint names (str)
                                     and values are their current positions (float,
                                     radians for revolute, meters for prismatic).
                                     Must contain entries for all non-fixed joints
                                     in the chain.
        base_link (str): The name of the reference link frame.
        tip_link (str): The name of the target link frame whose pose is computed.

    Returns:
        np.ndarray: A 4x4 numpy array representing the homogeneous transformation
                    from `base_link` to `tip_link` for the given joint positions.
    """
    # Get the ordered list of joints from base to tip
    chain = get_chain(robot, base_link, tip_link) # REDACTED: Call get_chain to obtain the list of joints
    # Initialize the total transformation as identity (pose of base_link in base_link frame)
    T = np.eye(4)

    # Iterate through each joint in the chain, starting from the base
    for joint in chain:
        # 1. Get the fixed transformation from the parent link to the joint's origin
        #    This is defined in the URDF's <origin> tag for the joint.

        T_fixed = homogeneous_transform(joint.origin.xyz, joint.origin.rpy) # REDACTED: Call homogeneous_transform with joint origin xyz and rpy

        # 2. Get the variable transformation due to the joint's movement
        T_variable = np.eye(4) # Default to identity (no movement for fixed joints)
        # Look up the current position/angle for this joint from the input dictionary
        # Default to 0.0 if the joint isn't specified (e.g., fixed joints might be in chain)
        current_joint_value = joint_positions_dict.get(joint.name, 0.0)

        # Check the joint type and apply the corresponding transformation
        if joint.type in ['revolute', 'continuous']:
            # For rotating joints, use rotation_about_axis
            if joint.axis is not None: # Axis must be defined
                angle = current_joint_value
                T_variable = rotation_about_axis(joint.axis, angle) # REDACTED: Call rotation_about_axis with joint axis and angle
        elif joint.type == 'prismatic':
            # For sliding joints, create a translation along the axis
            if joint.axis is not None: # Axis must be defined
                disp = current_joint_value
                # Ensure axis is normalized before scaling by displacement
                axis_norm = np.array(joint.axis, dtype=np.float64)
                norm = np.linalg.norm(axis_norm)
                if norm > 1e-6: # Avoid division by zero for zero-length axis
                   axis_norm /= norm
                   # Set the translation part of the 4x4 matrix
                   T_variable[:3, 3] = axis_norm * disp # REDACTED: Calculate translation vector (axis_norm * disp)
        # No explicit action needed for 'fixed' joints, T_variable remains identity.

        # 3. Update the overall transformation by chaining the current joint's transforms
        T = T @ T_fixed @ T_variable # REDACTED: Perform the matrix multiplication T = T @ T_fixed @ T_variable

    # After processing all joints, T is the pose of tip_link relative to base_link
    return T

def compute_pose_error(T_current, T_desired, pos_weight=1.0, orient_weight=0.0): # Default orient_weight=0
    # This function is kept intact as it's more complex and not explicitly requested for redaction
    # It calculates the 6D error (3 position, 3 orientation) between two poses
    pos_error = T_desired[:3, 3] - T_current[:3, 3]
    R_err = T_desired[:3, :3] @ T_current[:3, :3].T
    trace_val = np.trace(R_err); clipped_val = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(clipped_val)
    orient_error = np.zeros(3)
    if 1e-6 < abs(angle) < pi - 1e-6:
        scale = angle / (2.0 * sin(angle))
        orient_error = scale * np.array([ R_err[2, 1] - R_err[1, 2],
                                          R_err[0, 2] - R_err[2, 0],
                                          R_err[1, 0] - R_err[0, 1] ])
    elif abs(angle - pi) < 1e-6: # Handle near-pi case separately if needed
        axis_unnormalized = np.array([ R_err[2, 1] - R_err[1, 2],
                                       R_err[0, 2] - R_err[2, 0],
                                       R_err[1, 0] - R_err[0, 1] ])
        norm_axis = np.linalg.norm(axis_unnormalized)
        if norm_axis > 1e-6: orient_error = angle * axis_unnormalized / norm_axis
    return np.concatenate((pos_error * pos_weight, orient_error * orient_weight))

def compute_jacobian(current_joint_angles, robot, base_link, tip_link, joint_names, delta=1e-6):
    # It computes the Jacobian matrix using numerical differentiation (finite differences)
    n = len(current_joint_angles)
    if len(joint_names) != n: raise ValueError("Jacobian: Mismatch len(joint_names) vs len(angles)")
    joint_dict = {name: angle for name, angle in zip(joint_names, current_joint_angles)}
    T_current = forward_kinematics(robot, joint_dict, base_link, tip_link)
    J = np.zeros((6, n))
    for i in range(n):
        perturbed_angles = np.copy(current_joint_angles); perturbed_angles[i] += delta
        perturbed_dict = {name: angle for name, angle in zip(joint_names, perturbed_angles)}
        T_perturbed = forward_kinematics(robot, perturbed_dict, base_link, tip_link)
        # Note: Uses default weights (pos=1, orient=0) in compute_pose_error for Jacobian calculation
        error_perturbed = compute_pose_error(T_current, T_perturbed)
        if error_perturbed.shape != (6,): raise RuntimeError(f"Jacobian: Pose error shape {error_perturbed.shape}")
        J[:, i] = error_perturbed / delta
    return J

def ik_solver(T_desired, initial_pose_angles, robot, base_link, tip_link, joint_names,
              iterations=100, alpha=0.1, tol=1e-4, lambda_damping=0.01):
    """
    Computes Inverse Kinematics using the iterative Jacobian pseudo-inverse method.

    WHAT:
    Calculates the set of joint angles/positions (`q`) for a given kinematic chain
    (`joint_names` between `base_link` and `tip_link`) that will cause the `tip_link`
    to reach a specified target pose (`T_desired`). This is the inverse problem of
    Forward Kinematics. Since an analytical solution is often impossible or too
    complex, this function uses an iterative numerical approach.

    WHY:
    IK is crucial for controlling robots in "task space". Instead of telling the
    robot "set joint 1 to 0.5 rad, joint 2 to -1.0 rad...", we want to say "move
    the gripper to location (x,y,z) with orientation (r,p,y)". IK solvers figure
    out the necessary joint angles to achieve that task-space goal.

    HOW: (Iterative Jacobian Pseudo-Inverse with Damping - DLS)
    1. Input: Target pose `T_desired` (4x4 matrix), an initial guess for the joint
       angles `initial_pose_angles`, the `robot` model, link names, IK parameters.
    2. Initialization: Copies the `initial_pose_angles` into `joint_angles`, which
       will be updated iteratively. Sets up tracking variables.
    3. Iteration Loop: Repeats up to `iterations` times:
       a. Forward Kinematics (FK): Calculate the *current* pose `T_current` of the
          `tip_link` based on the *current* `joint_angles` using `forward_kinematics`.
       b. Compute Error: Calculate the difference between the `T_current` and the
          `T_desired` pose using `compute_pose_error`. This yields a 6D error vector
          `error` (3 position errors dx, dy, dz; 3 orientation errors drx, dry, drz).
          *Note: This implementation is configured by default (in compute_pose_error)
           to ignore orientation error (orient_weight=0), focusing only on position.*
       c. Check Convergence: Calculate the magnitude (norm) of the `error` vector.
          If `error_norm` is less than the specified tolerance `tol`, the solution
          is close enough. Return the current `joint_angles` and `True` (success).
       d. Check Divergence/Stagnation: Monitor if the error is increasing significantly,
          which might indicate divergence. If so, stop and return failure.
       e. Compute Jacobian: Calculate the Jacobian matrix `J` using `compute_jacobian`
          for the *current* `joint_angles`. The Jacobian (6xN matrix, N=num joints)
          relates infinitesimal changes in joint space (dq) to infinitesimal changes
          in task space (dx): `dx = J * dq`.
       f. Calculate Joint Change (The Core IK Step): We want to find `dq` that moves
          us towards the goal (i.e., reduces the `error`). This involves "inverting"
          the Jacobian relationship: `dq = J_inverse * error`.
          - Since `J` is usually not square or might be near a singularity (a pose
            where the robot loses some freedom of motion), we use the *damped pseudo-inverse*
            (Damped Least Squares - DLS):
            `J_pinv = J.T @ inv(J @ J.T + lambda_damping**2 * I)`
            where `J.T` is the transpose, `I` is the identity matrix, and `lambda_damping`
            is a small damping factor that prevents the matrix inversion from failing
            near singularities and improves stability, at the cost of potentially
            slowing convergence slightly.
          - Calculate the desired change in joint angles for this iteration:
            `delta_q = alpha * J_pinv @ error`. The `alpha` parameter is a step size
            (like a learning rate) controlling how much of the calculated change is
            applied in each step. Smaller alpha is more stable but slower.
       g. Update Joint Angles: Add the calculated change `delta_q` to the current
          `joint_angles`: `joint_angles += delta_q`.
       h. Safety Checks: Include checks within the loop for NaN (Not a Number) or Inf
          (Infinity) values in angles, error, Jacobian, or delta_q, which indicate
          numerical problems. Abort and return failure if detected.
    4. No Convergence: If the loop finishes without the error dropping below the
       tolerance, the solver failed to converge within the given iterations. Print
       a message and return the last attempted `joint_angles` and `False` (failure).

    Args:
        T_desired (np.ndarray): The target 4x4 pose matrix for the tip_link.
        initial_pose_angles (np.ndarray): A numpy array of starting joint angles
                                          (initial guess). Order must match `joint_names`.
        robot (URDF): The robot model object.
        base_link (str): Name of the base link of the chain.
        tip_link (str): Name of the tip link of the chain.
        joint_names (list): List of names (str) of the movable joints in the chain,
                            in the order corresponding to `initial_pose_angles`.
        iterations (int): Maximum number of iterations to perform.
        alpha (float): Step size (learning rate) for updating joint angles.
        tol (float): Convergence tolerance. The Euclidean norm of the pose error
                     must be below this value for success.
        lambda_damping (float): Damping factor for the DLS pseudo-inverse calculation.
                                Helps with stability near singularities.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The calculated joint angles (best effort, even if failed).
            - bool: `True` if converged successfully within tolerance, `False` otherwise.
    """
    # Make a copy of the initial guess to modify iteratively
    joint_angles = np.copy(initial_pose_angles)
    n_joints = len(joint_names)
    # Input validation
    if len(joint_angles) != n_joints:
         raise ValueError(f"IK Solver: Mismatch between initial_pose_angles ({len(joint_angles)}) "
                          f"and joint_names ({n_joints})")

    last_error_norm = float('inf') # Initialize for divergence check

    # Main iteration loop
    for i in range(iterations):
        # --- Safety Check: NaN in current angles ---
        if np.isnan(joint_angles).any():
             print(f"IK aborted at iteration {i}: NaN detected in joint angles.")
             return joint_angles, False # Indicate failure

        # --- 1. Calculate Current Pose (Forward Kinematics) ---
        # Create dictionary mapping joint names to current angles for FK function
        joint_positions_dict = {name: angle for name, angle in zip(joint_names, joint_angles)}
        try:
            T_current = forward_kinematics(robot, joint_positions_dict, base_link, tip_link) # REDACTED: Call forward_kinematics
            # --- 2. Calculate Pose Error ---
            # Note: compute_pose_error defaults to orient_weight=0.0 for this application
            error = compute_pose_error(T_current, T_desired, pos_weight=1.0, orient_weight=0.0) # REDACTED: Call compute_pose_error
        except Exception as e:
            # Handle potential errors during FK or error calculation
            print(f"IK aborted at iteration {i}: Error during FK or pose error calculation: {e}")
            # traceback.print_exc() # Uncomment for detailed traceback
            return joint_angles, False # Indicate failure

        # --- 3. Check for Convergence ---
        error_norm = np.linalg.norm(error) # Euclidean norm of the 6D error vector

        # --- Safety Check: NaN in error ---
        if np.isnan(error_norm):
             print(f"IK aborted at iteration {i}: NaN detected in error norm.")
             return joint_angles, False

        if error_norm < tol:
            # print(f"IK converged in {i} iterations. Final error norm: {error_norm:.6f}")
            return joint_angles, True # Return solution and success flag

        # --- Check for Divergence/Stagnation ---
        # If error significantly increased compared to last step (allow some initial fluctuation)
        if error_norm > last_error_norm * 1.5 and i > 10 :
             print(f"IK diverged or stagnated at iteration {i}. Error norm increased "
                   f"({last_error_norm:.4f} -> {error_norm:.4f}).")
             return joint_angles, False # Terminate on divergence

        last_error_norm = error_norm # Store current error norm for next iteration's check

        # --- 4. Compute Jacobian ---
        try:
            J = compute_jacobian(joint_angles, robot, base_link, tip_link, joint_names, delta=1e-6) # REDACTED: Call compute_jacobian
            # --- Safety Check: NaN/Inf in Jacobian ---
            if np.isnan(J).any() or np.isinf(J).any():
                 print(f"IK aborted at iteration {i}: NaN or Inf detected in Jacobian.")
                 return joint_angles, False

            # --- 5. Compute Damped Pseudo-Inverse (DLS) ---
            JT = J.T # Transpose of Jacobian
            JJT = J @ JT # J * J_transpose
            # Identity matrix matching the dimension of JJT (6x6 for typical 6D error)
            identity_matrix = np.eye(JJT.shape[0])
            # Add damping term (lambda^2 * I) to the diagonal of JJT
            damped_term = lambda_damping**2 * identity_matrix
            try:
                 # Calculate the inverse term: (J*J^T + lambda^2*I)^-1
                 inv_term = np.linalg.inv(JJT + damped_term)
            except np.linalg.LinAlgError:
                 # Handle cases where matrix is singular even with damping
                 print(f"IK solver: Matrix inversion failed at iteration {i} "
                       "(likely singularity even with damping).")
                 return joint_angles, False

            # Calculate the damped pseudo-inverse: J^T * (J*J^T + lambda^2*I)^-1
            J_pinv = JT @ inv_term # REDACTED: Calculate the pseudo-inverse using JT and inv_term

        except Exception as e:
             # Handle potential errors during Jacobian or inversion steps
             print(f"IK aborted at iteration {i}: Error during Jacobian computation or inversion: {e}")
             # traceback.print_exc()
             return joint_angles, False

        # --- 6. Calculate Change in Joint Angles ---
        # delta_q = step_size * pseudo_inverse * error_vector
        delta_q = alpha * J_pinv @ error # REDACTED: Calculate delta_q using alpha, J_pinv, and error

        # --- Safety Check: NaN/Inf in delta_q ---
        if np.isnan(delta_q).any() or np.isinf(delta_q).any():
             print(f"IK aborted at iteration {i}: NaN or Inf detected in joint angle update (delta_q).")
             return joint_angles, False

        # --- 7. Update Joint Angles ---
        joint_angles += delta_q # REDACTED: Add delta_q to the joint_angles

    # --- Loop finished without convergence ---
    print(f"IK failed to converge after {iterations} iterations. Final error norm: {error_norm:.6f}")
    return joint_angles, False # Return last attempt and failure flag

# --- END Kinematics Functions ---


# === SVG Parsing Functions ===
def parse_svg_file_robust(svg_filename, num_segments=6):
    if not os.path.exists(svg_filename): raise FileNotFoundError(f"SVG file not found: {svg_filename}")
    try:
        tree = ET.parse(svg_filename); root = tree.getroot()
        all_paths_points = [] # List of lists, one per path
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        paths = root.findall('.//svg:path', namespaces) or root.findall('.//path')
        if not paths: print("Warning: No <path> elements found in SVG."); return [], []
        print(f"Found {len(paths)} path elements.")
        for i, path_element in enumerate(paths):
            d = path_element.attrib.get('d')
            if not d: print(f"Warning: Path {i+1} has no 'd' attribute."); continue
            try:
                path_obj = parse_path(d); current_path_points = []
                for segment in path_obj:
                    start_complex = segment.start
                    if not current_path_points: # Add absolute start of the path
                        current_path_points.append((start_complex.real, start_complex.imag))
                    if isinstance(segment, Line):
                        current_path_points.append((segment.end.real, segment.end.imag))
                    elif isinstance(segment, (Arc, CubicBezier, QuadraticBezier)):
                        for t_step in range(1, num_segments + 1):
                            t = t_step / num_segments; point = segment.point(t)
                            pt = (point.real, point.imag)
                            # Avoid adding duplicate points if segment start/end coincide
                            if not current_path_points or np.linalg.norm(np.array(pt) - np.array(current_path_points[-1])) > 1e-6:
                                current_path_points.append(pt)
                # Handle 'Z' close path command explicitly
                if d.strip().upper().endswith('Z') and current_path_points:
                    start_pt = current_path_points[0]
                    if np.linalg.norm(np.array(current_path_points[-1]) - np.array(start_pt)) > 1e-6:
                        current_path_points.append(start_pt)
                if current_path_points:
                    # print(f"Extracted {len(current_path_points)} points from path {i+1}.") # Less verbose
                    all_paths_points.append(current_path_points)
                else: print(f"Warning: No points extracted from path {i+1}.")
            except Exception as e: print(f"Error parsing d for path {i+1}: {e}\n{traceback.format_exc()}")
        flat_point_list = [pt for path in all_paths_points for pt in path]
        print(f"SVG Parsing: Returning {len(flat_point_list)} total points across {len(all_paths_points)} paths.")
        return flat_point_list, all_paths_points # Return flat list and list of lists
    except Exception as e: print(f"SVG parsing failed: {e}\n{traceback.format_exc()}"); return [], []

# --- Plotting Function [Optional Utility] ---
def plot_svg_paths(list_of_paths, svg_filename):
    if not list_of_paths: print("No paths to plot."); return
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, path_points in enumerate(list_of_paths):
        if len(path_points) > 1:
            x_coords, y_coords = zip(*path_points) # More concise way to get coords
            ax.plot(x_coords, y_coords, marker='.', markersize=3, linestyle='-', label=f'Path {i+1}')
        elif len(path_points) == 1:
            ax.scatter(path_points[0][0], path_points[0][1], marker='x', s=50, label=f'Path {i+1} (single point)')
    ax.set_aspect('equal', adjustable='box'); ax.invert_yaxis()
    ax.set_xlabel("X Coordinate"); ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Visualized Path(s) from {os.path.basename(svg_filename)}")
    ax.grid(True); ax.legend()
    plt.show() # NOTE: This blocks continued execution until closed!

# --- Point Transformation Function ---
def transform_points_to_arm_vertical_plane(points_2d, scale=0.001, x_offset=0.20,
                                           y_offset=0.0, z_offset=0.15, center=True):
    if not points_2d: return []
    points_np = np.array(points_2d)
    min_xy = points_np.min(axis=0); max_xy = points_np.max(axis=0)
    size_xy = max_xy - min_xy
    center_xy = min_xy + 0.5 * size_xy
    points_3d = []
    for x_svg, y_svg in points_2d:
        X = x_offset
        if center:
            Y = -scale * (x_svg - center_xy[0]) + y_offset
            Z = -scale * (y_svg - center_xy[1]) + z_offset
        else:
            Y = -scale * x_svg + y_offset
            Z = -scale * y_svg + z_offset
        points_3d.append((X, Y, Z))
    print(f"SVG BBox Scaled: w={size_xy[0]*scale:.3f}m, h={size_xy[1]*scale:.3f}m")
    if points_3d:
        points_3d_np = np.array(points_3d)
        min_YZ = points_3d_np[:, 1:].min(axis=0); max_YZ = points_3d_np[:, 1:].max(axis=0)
        print(f"Transformed points approx: X={points_3d[0][0]:.3f}, Y=[{min_YZ[0]:.3f}, {max_YZ[0]:.3f}], Z=[{min_YZ[1]:.3f}, {max_YZ[1]:.3f}]")
    return points_3d


# === ROS 2 Node ===
# [ Rest of the IntegratedManipulatorNode class definition ]
class IntegratedManipulatorNode(Node):
    def __init__(self):
        super().__init__("integrated_manipulator_node")
        package_name_for_node = 'robopicasso'

        # --- Parameters ---

        # Motion Control Params
        self.declare_parameter("arm_controller_name", "arm_controller")
        self.declare_parameter("gripper_controller_name", "gripper_controller")
        self.declare_parameter("diff_drive_controller_name", "diff_drive_controller")
        self.declare_parameter("velocity_publish_rate", 10.0)
        self.declare_parameter("spin_angular_vel", 0.5)
        self.declare_parameter("gripper_open_pos", 0.025)
        self.declare_parameter("gripper_closed_pos", -0.01)
        self.declare_parameter("enable_cartesian_smoothing", True)
        self.declare_parameter("smoothing_window_length", 7)
        self.declare_parameter("smoothing_polyorder", 3)
        self.declare_parameter("max_joint_velocity", 5.0)
        self.declare_parameter("min_segment_time", 0.05)

        # Kinematics/URDF Params
        self.declare_parameter("joints", ["joint1", "joint2", "joint3", "joint4"])
        self.declare_parameter("urdf_package", "turtlebot3_manipulation_description")
        self.declare_parameter("urdf_relative_path", "urdf/tbotomx.urdf")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("tip_link", "end_effector_link")

        # IK Solver Params
        self.declare_parameter("ik_iterations", 400)
        self.declare_parameter("ik_alpha", 0.05)
        self.declare_parameter("ik_tolerance", 5e-3)
        self.declare_parameter("ik_damping", 0.1)

        # SVG Processing Params Detailed Declarations
        self.declare_parameter(
            "svg_relative_path",        # Parameter Name (string)
            "svg_files/VectorizedAbrams.svg" # Default Value (string)
        )
        self.declare_parameter(
            "svg_scale",                # Parameter Name (string)
            0.0002   # REDACTED/MODIFY   # Default Value (float)
        )
        # WHAT: A scaling factor to convert the coordinates *inside* the SVG file
        #       (which are often unitless or in pixels) into *meters* used by the robot.
        # HOW:  During the transformation process (`transform_points_to_arm_vertical_plane`),
        #       the X and Y values read from the SVG path data are multiplied by this number.
        #       For example, if a line in the SVG goes from (0,0) to (1000, 0) and the scale
        #       is 0.0001, the corresponding points in meters (before offsetting and mapping
        #       to robot axes) will be (0,0) and (0.1, 0).
        # WHY:  SVGs don't have an inherent real-world scale. This parameter lets the user
        #       control the physical size of the drawing produced by the robot. Adjusting
        #       this value makes the drawing larger or smaller in the robot's workspace.
        
        self.declare_parameter(
            "svg_x_offset",             # Parameter Name (string)
            0.150                       # Default Value (float, meters)
        )
        self.declare_parameter(
            "svg_y_offset",             # Parameter Name (string)
            -0.045                      # Default Value (float, meters)
        )
        self.declare_parameter(
            "svg_z_offset",             # Parameter Name (string)
            0.30                        # Default Value (float, meters)
        )
        self.declare_parameter(
            "svg_center",               # Parameter Name (string)
            True                        # Default Value (boolean)
        )
        self.declare_parameter(
            "svg_target_speed",         # Parameter Name (string)
            0.03                        # Default Value (float, meters per second)
        )
        self.declare_parameter(
            "svg_fixed_roll",           # Parameter Name (string)
            0.0                         # Default Value (float, radians)
        )
        self.declare_parameter(
            "svg_fixed_pitch",          # Parameter Name (string)
            0.0                         # Default Value (float, radians)
        )
        self.declare_parameter(
            "svg_fixed_yaw",            # Parameter Name (string)
            0.0                         # Default Value (float, radians)
        )

        # Timing Params
        self.declare_parameter("start_delay_sec", 2.0)
        self.declare_parameter("time_to_reach_start", 1.0)
        self.declare_parameter("time_to_return_home", 4.0)

        # Read parameters (using self.get_param helper)
        arm_ctrl = self._get_param("arm_controller_name")
        gripper_ctrl = self._get_param("gripper_controller_name")
        diff_drive_ctrl = self._get_param("diff_drive_controller_name")
        self.joint_names_param = self._get_param("joints")
        self.base_link = self._get_param("base_link")
        self.tip_link = self._get_param("tip_link")
        self.spin_angular_vel = self._get_param("spin_angular_vel")
        self.gripper_open_pos = self._get_param("gripper_open_pos")
        self.gripper_closed_pos = self._get_param("gripper_closed_pos")

        self.ik_iterations = self._get_param("ik_iterations")
        self.ik_alpha = self._get_param("ik_alpha")
        self.ik_tolerance = self._get_param("ik_tolerance")
        self.ik_damping = self._get_param("ik_damping")

        self.svg_relative_path = self._get_param("svg_relative_path")
        self.svg_scale = self._get_param("svg_scale")
        self.svg_x_offset = self._get_param("svg_x_offset")
        self.svg_y_offset = self._get_param("svg_y_offset")
        self.svg_z_offset = self._get_param("svg_z_offset")
        self.svg_center = self._get_param("svg_center")
        self.svg_target_speed = self._get_param("svg_target_speed")
        self.svg_fixed_rpy = (self._get_param("svg_fixed_roll"),
                              self._get_param("svg_fixed_pitch"),
                              self._get_param("svg_fixed_yaw"))
        self.start_delay_sec = self._get_param("start_delay_sec")
        self.time_to_reach_start = self._get_param("time_to_reach_start")
        self.time_to_return_home = self._get_param("time_to_return_home")
        self.enable_smoothing = self._get_param("enable_cartesian_smoothing")
        self.smoothing_window = self._get_param("smoothing_window_length")
        self.smoothing_polyorder = self._get_param("smoothing_polyorder")
        self.max_joint_velocity = self._get_param("max_joint_velocity")
        self.min_segment_time = self._get_param("min_segment_time")

        # Construct absolute SVG path
        try:
            self.svg_absolute_path = os.path.join(
                get_package_share_directory(package_name_for_node), self.svg_relative_path
            )
            self.get_logger().info(f"Expecting SVG at: {self.svg_absolute_path}")
        except Exception as e:
            self.get_logger().error(f"Could not get SVG path for package '{package_name_for_node}'. Error: {e}")
            rclpy.try_shutdown()
            return

        # --- Kinematics Setup ---
        try:
            urdf_pkg = self._get_param("urdf_package")
            urdf_rel_path = self._get_param("urdf_relative_path")
            urdf_path = os.path.join(get_package_share_directory(urdf_pkg), urdf_rel_path)
            self.get_logger().info(f"Loading URDF from: {urdf_path}")
            self.robot = URDF.from_xml_file(urdf_path)
            # Call the (potentially unredacted) get_chain here during init to check structure
            chain = get_chain(self.robot, self.base_link, self.tip_link)
            self.ik_joint_names = [j.name for j in chain if j.type != 'fixed']
            self.get_logger().info(f"Kinematic chain joints (IK order): {self.ik_joint_names}")
            if set(self.joint_names_param) != set(self.ik_joint_names):
                 self.get_logger().error(f"Mismatch: 'joints' param {self.joint_names_param} != IK joints {self.ik_joint_names}.")
                 rclpy.try_shutdown(); return
            if list(self.joint_names_param) != list(self.ik_joint_names):
                 self.get_logger().warn(f"Order mismatch: param {self.joint_names_param} vs IK {self.ik_joint_names}. Assuming param order for controller.")
        except Exception as e:
             self.get_logger().error(f"Failed to setup kinematics: {e}\n{traceback.format_exc()}")
             # If get_chain fails due to redaction, this error might occur.
             # The node might still start, but functionality will be broken.
             self.get_logger().warn("Continuing initialization despite kinematics setup error (may be due to redactions)...")
             # rclpy.try_shutdown(); return # Don't shutdown, allow it to run partially so cadets can troubleshoot


        # --- State Variables ---
        self.current_joint_angles = None
        self.joint_state_msg_received = False
        self.latest_joint_state_msg = None
        self.trajectory_active = False
        self.desired_linear_vel = 0.0
        self.desired_angular_vel = 0.0
        self._last_generated_traj = None
        self._open_gripper_timer = None

        # --- Publishers ---
        self.arm_publisher = self.create_publisher(JointTrajectory, f"/{arm_ctrl}/joint_trajectory", 10)
        self._internal_cmd_pub = self.create_publisher(String, '~/internal_command', 10)
        cmd_vel_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.cmd_vel_pub_ = self.create_publisher(TwistStamped, f"/{diff_drive_ctrl}/cmd_vel", cmd_vel_qos)
        self.cmd_vel_msg = TwistStamped()

        # --- Subscribers ---
        joint_state_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, joint_state_qos)
        self._internal_cmd_sub = self.create_subscription(String, '~/internal_command', self._internal_command_callback, 10)

        # --- Action Clients ---
        self.gripper_client = ActionClient(self, GripperCommand, f'/{gripper_ctrl}/gripper_cmd')
        if not self.gripper_client.wait_for_server(timeout_sec=3.0): self.get_logger().warn("Gripper action server not available.")
        else: self.get_logger().info("Gripper action client connected.")

        # --- Timers ---
        self.velocity_publish_timer = self.create_timer(1.0 / self._get_param("velocity_publish_rate"), self.publish_velocity_callback)
        # Use a one-shot timer to start the process after initial delay and receiving joint states
        self.initiation_timer = self.create_timer(self.start_delay_sec, self.initiate_trajectory)

        self.get_logger().info("Integrated Manipulator Node initialized for SVG tracing.")

    def initiate_trajectory(self):
        """Called by timer to start SVG processing if prerequisites are met."""
        if self.initiation_timer: self.initiation_timer.cancel() # Cancel this timer

        if self.trajectory_active:
            self.get_logger().info("SVG trajectory already active or completed.")
            return

        if not self.joint_state_msg_received or self.current_joint_angles is None:
            self.get_logger().warn("Joint states not ready. Retrying SVG trajectory initiation in 5s.")
            # Reschedule the check
            self.initiation_timer = self.create_timer(5.0, self.initiate_trajectory)
            return


    def _get_param(self, name):
        """Helper to get parameter value."""
        return self.get_parameter(name).value

    
    def joint_state_callback(self, msg: JointState):
        """Stores the latest joint states needed for IK and initial state capture."""
        self.latest_joint_state_msg = msg # Store the whole message

        if self.trajectory_active:
             # Maybe add a log here if needed for debugging state updates
             # self.get_logger().debug("Trajectory active, not updating current_joint_angles seed from joint_state.")
             pass # Don't update the seed while a trajectory is planned/running

        # Check if ik_joint_names is available (might fail during init if get_chain was redacted/failed)
        if not hasattr(self, 'ik_joint_names') or not self.ik_joint_names:
            if not self.joint_state_msg_received: # Log once
                self.get_logger().warn("IK joint names not available (kinematics setup might have failed). Cannot process joint states.")
            self.joint_state_msg_received = True # Prevent repeated logging
            return

        required = set(self.ik_joint_names)
        available = set(msg.name)
        if not required.issubset(available):
            if not self.joint_state_msg_received: # Log once
                 self.get_logger().warn(f"Waiting for joint states. Missing: {required - available}")
            return

        current_states = {name: pos for name, pos in zip(msg.name, msg.position)}
        try:
            # Extract angles in the specific order needed by IK for seeding
            current_ik_angles = np.array([current_states[name] for name in self.ik_joint_names])

            # Only update the 'live' seed if trajectory isn't active
            if not self.trajectory_active:
                self.current_joint_angles = current_ik_angles

            if not self.joint_state_msg_received:
                self.get_logger().info(f"Received first joint states (IK order): {np.round(current_ik_angles, 3)}")
            self.joint_state_msg_received = True
        except KeyError as e:
             self.get_logger().error(f"Error extracting joint state for '{e}'.")
             self.current_joint_angles = None
             self.joint_state_msg_received = False

    def initiate_svg_trajectory(self):
        """Called by timer to start SVG processing if prerequisites are met."""
        if self.initiation_timer: self.initiation_timer.cancel() # Cancel this timer

        if self.trajectory_active:
            self.get_logger().info("SVG trajectory already active or completed.")
            return

        if not self.joint_state_msg_received or self.current_joint_angles is None:
            self.get_logger().warn("Joint states not ready. Retrying SVG trajectory initiation in 5s.")
            # Reschedule the check
            self.initiation_timer = self.create_timer(5.0, self.initiate_svg_trajectory)
            return

        # Add a check to see if kinematics setup likely failed
        if not hasattr(self, 'robot') or not hasattr(self, 'ik_joint_names'):
            self.get_logger().error("Kinematics (URDF/chain) not properly initialized. Cannot start SVG trajectory.")
            self.trajectory_active = False # Ensure flag is false
            return

        self.get_logger().info("Prerequisites met. Starting SVG trajectory generation...")
        self.generate_and_send_svg_trajectory()

    def generate_and_send_svg_trajectory(self):
        """Starts the SVG trajectory generation in a separate thread."""
        if self.trajectory_active:
            self.get_logger().warn("Trajectory generation already active. Ignoring request.")
            return

        self.trajectory_active = True
        self.get_logger().info("Starting SVG trajectory task in background thread...")
        thread = threading.Thread(target=self._execute_svg_trajectory_task, daemon=True)
        thread.start()

    # --- Helper Methods for _execute_svg_trajectory_task ---
    def _wait_for_ros_and_joints(self):
        """Waits for rclpy context and initial joint states."""
        loop_count = 0
        while not rclpy.ok() and loop_count < 100: # Timeout after ~10s
            if loop_count == 0: print("Waiting for rclpy context...")
            time.sleep(0.1); loop_count += 1
        if not rclpy.ok(): print("ERROR: Timed out waiting for rclpy context!"); return False
        self.get_logger().info("rclpy context ok.")
        time.sleep(0.5) # Extra safety pause

        max_wait = 5.0; waited_time = 0.0; wait_interval = 0.1
        while self.current_joint_angles is None and waited_time < max_wait:
            self.get_logger().warn(f"Waiting for initial joint angles... ({waited_time:.1f}s)")
            time.sleep(wait_interval); waited_time += wait_interval
        if self.current_joint_angles is None:
             self.get_logger().error("Failed to get initial joint angles. Aborting task.")
             return False
        return True

    def _load_and_transform_svg(self):
        """Loads SVG, parses paths, and transforms points."""
        try:
            self.get_logger().info(f"Parsing SVG: {self.svg_absolute_path}")
            points2D, listOfpaths2D = parse_svg_file_robust(self.svg_absolute_path)

            if listOfpaths2D: # Optional plotting
                try:
                    self.get_logger().info("Plotting extracted SVG paths (blocking)...")
                    plot_svg_paths(listOfpaths2D, self.svg_absolute_path)
                except Exception as plot_err:
                     self.get_logger().warn(f"Plotting failed (display might not be available): {plot_err}")
            else:
                self.get_logger().warn("No paths extracted or plotting failed.")

            if not points2D:
                self.get_logger().error(f"SVG file parsing failed or returned no points: {self.svg_absolute_path}")
                return None

            self.get_logger().info(f"Transforming {len(points2D)} SVG points to 3D...")
            points3D = transform_points_to_arm_vertical_plane(
                points2D, self.svg_scale, self.svg_x_offset,
                self.svg_y_offset, self.svg_z_offset, self.svg_center
            )
            if not points3D:
                 self.get_logger().error("Failed to transform SVG points to 3D.")
                 return None
            self.get_logger().error(f"\n\npoints3D: {points3D}.\n")
            
            return points3D
        except FileNotFoundError:
            self.get_logger().error(f"SVG file not found at {self.svg_absolute_path}.")
            return None
        except Exception as e:
            self.get_logger().error(f"Error during SVG processing: {e}\n{traceback.format_exc()}")
            return None

    # --- Helper for Smoothing ---
    def _smooth_cartesian_path(self, points_3d):
        """Applies Savitzky-Golay filter to smooth the 3D path."""
        if not SCIPY_AVAILABLE:
            self.get_logger().warn("Scipy not installed. Cannot perform Cartesian path smoothing.", throttle_duration_sec=30)
            return points_3d # Return original points

        if len(points_3d) < self.smoothing_window:
            self.get_logger().warn(f"Not enough points ({len(points_3d)}) for smoothing window ({self.smoothing_window}). Skipping smoothing.")
            return points_3d # Return original points

        if self.smoothing_polyorder >= self.smoothing_window:
            self.get_logger().error(f"Smoothing polyorder ({self.smoothing_polyorder}) must be less than window size ({self.smoothing_window}). Skipping smoothing.")
            return points_3d

        if self.smoothing_window % 2 == 0:
             self.get_logger().error(f"Smoothing window size ({self.smoothing_window}) must be odd. Skipping smoothing.")
             return points_3d

        self.get_logger().info(f"Applying Savitzky-Golay smoothing: window={self.smoothing_window}, order={self.smoothing_polyorder}")
        points_3d_np = np.array(points_3d)

        try:
            # Apply filter to each coordinate (X, Y, Z)
            # Note: mode='interp' might be slightly better at boundaries than default 'mirror' for paths
            smoothed_x = savgol_filter(points_3d_np[:, 0], self.smoothing_window, self.smoothing_polyorder, mode='interp')
            smoothed_y = savgol_filter(points_3d_np[:, 1], self.smoothing_window, self.smoothing_polyorder, mode='interp')
            smoothed_z = savgol_filter(points_3d_np[:, 2], self.smoothing_window, self.smoothing_polyorder, mode='interp')

            points_3d_smoothed = list(zip(smoothed_x, smoothed_y, smoothed_z))
            self.get_logger().info(f"Smoothing applied successfully to {len(points_3d_smoothed)} points.")
            return points_3d_smoothed

        except Exception as e:
            self.get_logger().error(f"Error during Savitzky-Golay smoothing: {e}")
            return points_3d

    # --- Helper to plot comparison...see how well they are overlaid---
    def _plot_smoothed_vs_raw(self, raw_points, smoothed_points):
        try:
            fig = plt.figure(figsize=(12, 6))

            # Plot YZ projection (since X is fixed)
            ax1 = fig.add_subplot(121)
            raw_np = np.array(raw_points)
            smooth_np = np.array(smoothed_points)
            ax1.plot(raw_np[:, 1], raw_np[:, 2], 'r.-', label='Raw', markersize=4, alpha=0.7)
            ax1.plot(smooth_np[:, 1], smooth_np[:, 2], 'b.-', label='Smoothed', markersize=2)
            ax1.set_xlabel("Y (m)")
            ax1.set_ylabel("Z (m)")
            ax1.set_title("Smoothed vs Raw (YZ Plane)")
            ax1.legend()
            ax1.grid(True)
            ax1.invert_yaxis() # Match SVG coords if needed
            ax1.set_aspect('equal', adjustable='box')

            # Plot Z coordinate vs point index
            ax2 = fig.add_subplot(122)
            ax2.plot(raw_np[:, 2], 'r.-', label='Raw Z', alpha=0.7)
            ax2.plot(smooth_np[:, 2], 'b.-', label='Smoothed Z')
            ax2.set_xlabel("Point Index")
            ax2.set_ylabel("Z Coordinate (m)")
            ax2.set_title("Z Coordinate Smoothing")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
        except Exception as plot_err:
            self.get_logger().warn(f"Plotting smoothed vs raw failed: {plot_err}")


    def _calculate_ik_and_build_trajectory(self,
                                           points3D,
                                           initial_joint_angles_ik_order,    # Use this for seeding IK
                                           initial_angles_controller_order # Use this for the return point
                                          ):
        """
        Calculates IK for 3D points, builds the JointTrajectory message with
        refined timing based on Cartesian and joint speeds, and adds estimated velocities.
        """
        target_poses_T = [homogeneous_transform(pt, self.svg_fixed_rpy) for pt in points3D]
        num_targets = len(target_poses_T)
        self.get_logger().info(f"Generated {num_targets} target poses.")

        if num_targets == 0: return None, 0.0, 0.0

        traj = JointTrajectory()
        traj.joint_names = self.joint_names_param # Controller joint order

        # State variables for loop
        current_ik_solution = np.copy(initial_joint_angles_ik_order)
        last_sent_joint_angles_ik = None
        last_time_from_start = 0.0
        last_target_point_3d = None

        successful_points = 0
        total_distance = 0.0

        # --- Move to the starting point (Point 0) ---
        self.get_logger().info("Calculating IK for the first point...")
        try:
            ik_solution_first, success_first = ik_solver(
                target_poses_T[0], current_ik_solution, self.robot, self.base_link, self.tip_link,
                self.ik_joint_names, self.ik_iterations, self.ik_alpha, self.ik_tolerance, self.ik_damping
            )
        except Exception as ik_err:
            self.get_logger().error(f"IK solver crashed for first point: {ik_err}\n{traceback.format_exc()}")
            self.get_logger().error("This might be due to un-filled redactions in kinematic functions.")
            return None, 0.0, 0.0

        if not success_first:
            self.get_logger().error(f"IK failed for the *first* point (Pose: {target_poses_T[0][:3,3]}). Cannot start.")
            return None, 0.0, 0.0

        # Map FIRST IK solution to controller joint order
        try:
            ik_dict_first = {name: angle for name, angle in zip(self.ik_joint_names, ik_solution_first)}
            point0_positions = [ik_dict_first[name] for name in self.joint_names_param]
        except KeyError as e:
            self.get_logger().error(f"Joint name mismatch mapping first IK solution: {e}")
            return None, 0.0, 0.0

        point0 = JointTrajectoryPoint()
        point0.positions = point0_positions
        current_time_from_start = self.time_to_reach_start
        point0.time_from_start = Duration(sec=int(current_time_from_start), nanosec=int((current_time_from_start % 1) * 1e9))
        point0.velocities = [0.0] * len(point0_positions)
        traj.points.append(point0)

        successful_points += 1
        current_ik_solution = ik_solution_first
        last_sent_joint_angles_ik = ik_solution_first
        last_time_from_start = current_time_from_start
        last_target_point_3d = np.array(points3D[0])
        self.get_logger().info(f"First point added. Time: {current_time_from_start:.2f}s")

        # --- Iterate through the rest of the points (Point 1 to N-1) ---
        self.get_logger().info(f"Calculating IK and timing for remaining {num_targets - 1} points...")
        for i in range(1, num_targets):
            T_desired = target_poses_T[i]
            current_target_point_3d = np.array(points3D[i])

            try:
                ik_solution, success = ik_solver(
                    T_desired, current_ik_solution, self.robot, self.base_link, self.tip_link,
                    self.ik_joint_names, self.ik_iterations, self.ik_alpha, self.ik_tolerance, self.ik_damping
                )
            except Exception as ik_err:
                self.get_logger().error(f"IK solver crashed for point {i+1}: {ik_err}")
                self.get_logger().error("This might be due to un-filled redactions in kinematic functions. Skipping point.")
                success = False # Treat crash as failure for this point
                ik_solution = current_ik_solution # Keep previous solution as seed maybe

            if success:
                # --- Timing Calculation ---
                distance = np.linalg.norm(current_target_point_3d - last_target_point_3d)
                total_distance += distance
                time_cartesian = (distance / self.svg_target_speed) if self.svg_target_speed > 1e-6 else self.min_segment_time

                delta_q = ik_solution - last_sent_joint_angles_ik
                max_joint_change = 0.0
                if len(delta_q) > 0: max_joint_change = np.max(np.abs(delta_q))

                time_joint_limited = (max_joint_change / self.max_joint_velocity) if self.max_joint_velocity > 1e-6 else self.min_segment_time

                time_increment = max(time_cartesian, time_joint_limited, self.min_segment_time)
                if time_increment <= 1e-9:
                     self.get_logger().warn(f"Calculated zero/negative time_increment ({time_increment:.4g}) for segment {i}. Using min_segment_time.")
                     time_increment = self.min_segment_time
                current_time_from_start = last_time_from_start + time_increment

                # --- Map IK Solution to Controller Order ---
                try:
                    ik_dict = {name: angle for name, angle in zip(self.ik_joint_names, ik_solution)}
                    point_positions = [ik_dict[name] for name in self.joint_names_param]
                except KeyError as e:
                    self.get_logger().error(f"Joint name mismatch mapping IK solution for point {i+1}: {e}. Skipping.")
                    continue

                # --- Velocity Calculation (Finite Difference) ---
                try:
                    last_ik_dict = {name: angle for name, angle in zip(self.ik_joint_names, last_sent_joint_angles_ik)}
                    last_point_positions_ctrl_order = [last_ik_dict[name] for name in self.joint_names_param]
                    velocities = [(p_curr - p_last) / time_increment for p_curr, p_last in zip(point_positions, last_point_positions_ctrl_order)]
                except Exception as e:
                     self.get_logger().error(f"Error getting previous point angles for velocity calc at point {i+1}: {e}. Setting zero velocity.")
                     velocities = [0.0] * len(point_positions)

                # --- Create and Add Trajectory Point ---
                point = JointTrajectoryPoint()
                point.positions = point_positions
                point.velocities = velocities
                point.time_from_start = Duration(sec=int(current_time_from_start), nanosec=int((current_time_from_start % 1) * 1e9))
                traj.points.append(point)

                # --- Update State for Next Iteration *ONLY ON SUCCESS* ---
                current_ik_solution = ik_solution
                last_sent_joint_angles_ik = ik_solution
                last_time_from_start = current_time_from_start
                last_target_point_3d = current_target_point_3d
                successful_points += 1

            else: # If IK failed or crashed
                 self.get_logger().warn(f"IK failed or skipped for point {i+1} (Pos: {np.round(T_desired[:3,3],3)}).")
                 current_ik_solution = ik_solution # Use the failed solution as seed for next attempt

            if (i+1) % 50 == 0 or i == num_targets - 1:
                 self.get_logger().info(f"Processed point {i+1}/{num_targets}. Success: {successful_points}. Traj Time: {last_time_from_start:.2f}s")

        # --- Add the return-to-home point ---
        if successful_points > 0 and last_sent_joint_angles_ik is not None:
            self.get_logger().info("Adding return-to-start point...")

            return_delta_q = initial_joint_angles_ik_order - last_sent_joint_angles_ik
            return_max_joint_change = np.max(np.abs(return_delta_q)) if len(return_delta_q) > 0 else 0.0
            return_time_joint = (return_max_joint_change / self.max_joint_velocity) if self.max_joint_velocity > 1e-6 else self.min_segment_time
            return_time_increment = max(return_time_joint, self.time_to_return_home, self.min_segment_time)
            return_time_abs = last_time_from_start + return_time_increment

            return_point = JointTrajectoryPoint()
            return_point.positions = initial_angles_controller_order.tolist()
            return_point.velocities = [0.0] * len(return_point.positions)
            return_point.time_from_start = Duration(sec=int(return_time_abs), nanosec=int((return_time_abs % 1) * 1e9))
            traj.points.append(return_point)

            current_time_from_start = return_time_abs
            self.get_logger().info(f"Return point added. New total trajectory time: {current_time_from_start:.2f}s")
        else:
            self.get_logger().warn("Skipping return point as no points were successfully added to trajectory.")

        if not traj.points:
            self.get_logger().error("No points successfully added to trajectory.")
            return None, 0.0, 0.0
        elif len(traj.points) < 2:
             self.get_logger().warn("Trajectory contains fewer than 2 points. Movement might be trivial or jerky.")

        return traj, current_time_from_start, total_distance


    def _perform_pre_trajectory_actions(self):
        """Handles gripper actions before starting the arm movement."""
        self.get_logger().info("Performing pre-trajectory gripper sequence...")
        self.send_gripper_goal(self.gripper_closed_pos); time.sleep(1.0) # Close (alert)
        self.send_gripper_goal(self.gripper_open_pos); time.sleep(1.5)   # Open
        self.get_logger().info(f"*/*//*/*/*//*/*/* Insert laser now... */*//*/*/*//*/*/*")
        self.send_gripper_goal(self.gripper_closed_pos); time.sleep(1.5) # Close (hold)
        self.get_logger().info("Pre-trajectory sequence complete.")

    # --- Main Task Execution (Runs in Thread) ---
    def _execute_svg_trajectory_task(self):
        """Orchestrates the entire SVG tracing process in a background thread."""
        if not self._wait_for_ros_and_joints():
            self.trajectory_active = False; return

        # Verify kinematics setup again before proceeding
        if not hasattr(self, 'robot') or not hasattr(self, 'ik_joint_names'):
            self.get_logger().error("Kinematics (URDF/chain) not properly initialized. Aborting trajectory task.")
            self.trajectory_active = False
            return

        # --- Capture Initial State ---
        if self.latest_joint_state_msg is None:
            self.get_logger().error("Cannot start trajectory: Latest joint state message not available.")
            self.trajectory_active = False
            return

        try:
            latest_joint_states = {name: pos for name, pos in zip(self.latest_joint_state_msg.name, self.latest_joint_state_msg.position)}
            initial_angles_controller_order = np.array([latest_joint_states[name] for name in self.joint_names_param])
            initial_angles_ik_order = np.array([latest_joint_states[name] for name in self.ik_joint_names])
            self.get_logger().info(f"Stored initial joints (Controller Order): {np.round(initial_angles_controller_order, 3)}")
            self.get_logger().info(f"Stored initial joints (IK Order): {np.round(initial_angles_ik_order, 3)}")
        except KeyError as e:
            self.get_logger().error(f"Failed to extract initial joint angles from stored state for joint '{e}'. Aborting.")
            self.trajectory_active = False
            return
        except Exception as e:
            self.get_logger().error(f"Error capturing initial joint states: {e}\n{traceback.format_exc()}")
            self.trajectory_active = False
            return

        # --- Start Main Logic ---
        try:
            # 1. Start Spinning Wheels (via internal command)
            self.safe_publish_velocity(0.0, self.spin_angular_vel)
            time.sleep(0.1)

            # 2. Load SVG and Transform Points
            points3D_raw = self._load_and_transform_svg()
            if points3D_raw is None:
                self.safe_publish_velocity(0.0, 0.0)
                self.trajectory_active = False; return

            # 2b. Smooth the Cartesian Path
            if self.enable_smoothing and SCIPY_AVAILABLE:
                points3D = self._smooth_cartesian_path(points3D_raw)
                # Optional plotting comparison
                # self._plot_smoothed_vs_raw(points3D_raw, points3D)
            else:
                points3D = points3D_raw

            if not points3D:
                 self.get_logger().error("Point list is empty after optional smoothing. Aborting.")
                 self.safe_publish_velocity(0.0, 0.0)
                 self.trajectory_active = False; return

            # 3. Calculate IK and Build Trajectory
            traj, total_time, total_dist = self._calculate_ik_and_build_trajectory(
                points3D,
                initial_angles_ik_order,
                initial_angles_controller_order
            )
            if traj is None:
                self.get_logger().error("Failed to calculate trajectory (check IK solver/kinematics).")
                self.safe_publish_velocity(0.0, 0.0)
                self.trajectory_active = False; return

            self.get_logger().info(f"Generated final trajectory with {len(traj.points)} points.")
            self.get_logger().info(f"Estimated drawing distance: {total_dist:.3f} m")
            self.get_logger().info(f"Estimated total time (incl. return): {total_time:.2f} s")

            # 4. Pre-Trajectory Actions (Gripper)
            self._perform_pre_trajectory_actions()

            # 5. Stop Spinning Wheels
            self.get_logger().info("Stopping wheels before arm trajectory.")
            self.safe_publish_velocity(0.0, 0.0)
            time.sleep(0.1)

            # 6. Publish Trajectory (via internal command)
            self.get_logger().info("Requesting trajectory publish via internal topic...")
            self._last_generated_traj = traj
            self._publish_internal_command({"action": "publish_trajectory"})

            # 7. Schedule Post-Trajectory Actions (via internal command)
            delay = total_time + 2.0 # Add buffer time
            self.get_logger().info(f"Requesting scheduling of final actions after {delay:.2f}s.")
            self._publish_internal_command({"action": "schedule_final_timer", "delay": delay})
            self.get_logger().info("Background trajectory task completed initiation.")

        except Exception as e:
             self.get_logger().error(f"Unhandled exception in trajectory task thread: {e}\n{traceback.format_exc()}")
             try: self.safe_publish_velocity(0.0, 0.0)
             except Exception as e2: self.get_logger().error(f"Failed to stop wheels during exception handling: {e2}")
             self.trajectory_active = False

    # --- Internal Command Handling ---
    def _publish_internal_command(self, command_dict):
        """Helper to publish commands to the internal topic."""
        try:
            msg = String(data=json.dumps(command_dict))
            if hasattr(self, '_internal_cmd_pub') and self._internal_cmd_pub:
                self._internal_cmd_pub.publish(msg)
            else: self.get_logger().error("Internal command publisher not available!")
        except Exception as e:
            self.get_logger().error(f"Failed to publish internal command {command_dict}: {e}\n{traceback.format_exc()}")

    def _internal_command_callback(self, msg: String):
        """Handles commands received on the internal topic (runs in executor thread)."""
        try:
            cmd = json.loads(msg.data)
            action = cmd.get("action")

            if action == "publish_velocity":
                linear_x = cmd.get("linear_x", 0.0); angular_z = cmd.get("angular_z", 0.0)
                # self.get_logger().info(f"Exec internal: Publish Velocity ({linear_x:.2f}, {angular_z:.2f})") # Debug
                self.desired_linear_vel = float(linear_x)
                self.desired_angular_vel = float(angular_z)
                # self._publish_velocity_now()

            elif action == "publish_trajectory":
                self.get_logger().info("Exec internal: Publish Trajectory")
                if self._last_generated_traj and self.arm_publisher:
                     self.arm_publisher.publish(self._last_generated_traj)
                     self._last_generated_traj = None
                else: self.get_logger().error("No trajectory data or arm publisher for internal command.")

            elif action == "schedule_final_timer":
                 delay = cmd.get("delay")
                 if delay is not None:
                      self.get_logger().info(f"Exec internal: Schedule Timer (delay={delay:.2f})")
                      self._schedule_final_actions(float(delay))
                 else: self.get_logger().error("Delay missing for schedule_final_timer cmd.")

            else: self.get_logger().warn(f"Unknown internal command action: {action}")

        except json.JSONDecodeError: self.get_logger().error(f"Failed to decode internal command JSON: {msg.data}")
        except Exception as e: self.get_logger().error(f"Error processing internal command: {e}\n{traceback.format_exc()}")

    # --- Velocity Control Helpers ---
    def safe_publish_velocity(self, linear_x, angular_z):
        """Requests immediate velocity publish via the internal command topic."""
        # self.get_logger().info(f"Requesting internal velocity publish: lin={linear_x:.2f}, ang={angular_z:.2f}") # Debug
        self._publish_internal_command({
            "action": "publish_velocity",
            "linear_x": linear_x,
            "angular_z": angular_z
        })

    def _publish_velocity_now(self):
        """Publishes the current desired velocity immediately (must run in executor)."""
        if not hasattr(self, 'cmd_vel_pub_') or self.cmd_vel_pub_ is None:
            self.get_logger().warn("Velocity publisher not ready for immediate publish.", throttle_duration_sec=5)
            return
        try:
            self.cmd_vel_msg.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_msg.header.frame_id = self.base_link
            self.cmd_vel_msg.twist.linear.x = self.desired_linear_vel
            self.cmd_vel_msg.twist.angular.z = self.desired_angular_vel
            self.cmd_vel_msg.twist.linear.y = 0.0; self.cmd_vel_msg.twist.linear.z = 0.0
            self.cmd_vel_msg.twist.angular.x = 0.0; self.cmd_vel_msg.twist.angular.y = 0.0
            self.cmd_vel_pub_.publish(self.cmd_vel_msg)
        except Exception as e:
            self.get_logger().error(f"Failed immediate velocity publish: {e}", exc_info=True)

    def publish_velocity_callback(self):
        """Called by timer to continuously publish the *desired* velocity."""
        self._publish_velocity_now()

    # --- Final Actions Scheduling and Callback ---
    def _schedule_final_actions(self, delay):
        """Creates the timer for final actions (runs in executor thread)."""
        self.get_logger().info(f"Creating final action timer with delay {delay:.2f}s.")
        if self._open_gripper_timer:
            try: self._open_gripper_timer.cancel()
            except Exception as e: self.get_logger().warn(f"Exception cancelling previous timer: {e}")
        try:
            self._open_gripper_timer = self.create_timer(delay, self._final_actions_callback)
            self.get_logger().info("Final action timer created.")
        except Exception as e:
            self.get_logger().error(f"Failed to create final action timer: {e}", exc_info=True)

    def _final_actions_callback(self):
        """Callback for final actions after trajectory completion."""
        self.get_logger().info("Final action timer expired. Opening gripper and resuming spin.")
        self.send_gripper_goal(self.gripper_open_pos) # Open gripper

        # Resume Spinning by setting desired state (timer will publish)
        self.desired_linear_vel = 0.0
        self.desired_angular_vel = self.spin_angular_vel

        if self._open_gripper_timer:
             try: self._open_gripper_timer.cancel()
             except Exception as e: self.get_logger().warn(f"Exception cancelling final action timer: {e}")
             self._open_gripper_timer = None
        self.trajectory_active = False
        self.get_logger().info("SVG tracing process considered complete.")


    # --- Gripper Action Goal Sender ---
    def send_gripper_goal(self, position, max_effort=0.0):
       if not self.gripper_client or not self.gripper_client.server_is_ready():
           self.get_logger().error('Gripper action server not available!')
           return False
       goal_msg = GripperCommand.Goal()
       goal_msg.command.position = position
       goal_msg.command.max_effort = max_effort
       self.get_logger().info(f'Sending gripper goal: pos={position:.3f}')
       send_goal_future = self.gripper_client.send_goal_async(goal_msg)
       send_goal_future.add_done_callback(self.gripper_goal_response_callback)
       return True

    def gripper_goal_response_callback(self, future):
        """Log goal acceptance/rejection."""
        try:
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().warn('Gripper goal rejected.')
                return
            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self.gripper_get_result_callback)
        except Exception as e:
             self.get_logger().error(f"Exception in gripper goal response: {e}")

    def gripper_get_result_callback(self, future):
        """Log final result of gripper action."""
        try:
            status = future.result().status
            result = future.result().result
            if status == 4: # SUCCEEDED (GoalStatus.SUCCEEDED)
                # self.get_logger().info(f"Gripper OK: Pos={result.position:.4f} Reached={result.reached_goal}") # Less Verbose
                pass
            else:
                self.get_logger().warn(f"Gripper action finished with status: {status} (4=Success)")
        except Exception as e:
             self.get_logger().error(f"Exception getting gripper result: {e}")

    # --- Node cleanup ---
    def destroy_node(self):
        self.get_logger().info("Destroying node...")
        if hasattr(self, 'velocity_publish_timer') and self.velocity_publish_timer: self.velocity_publish_timer.cancel()
        if hasattr(self, '_open_gripper_timer') and self._open_gripper_timer: self._open_gripper_timer.cancel()
        if hasattr(self, 'initiation_timer') and self.initiation_timer: self.initiation_timer.cancel()
        if hasattr(self, 'cmd_vel_pub_') and self.cmd_vel_pub_ and rclpy.ok():
            try:
                stop_msg = TwistStamped()
                stop_msg.header.stamp = self.get_clock().now().to_msg()
                stop_msg.header.frame_id = self.base_link
                self.cmd_vel_pub_.publish(stop_msg)
                self.get_logger().info("Sent stop command to wheels.")
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().warn(f"Could not send stop command on shutdown: {e}")
        super().destroy_node()
        self.get_logger().info("Node destroyed.")

# --- Main function ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        print("Creating IntegratedManipulatorNode...")
        node = IntegratedManipulatorNode()
        if node and rclpy.ok():
             print("Node created. Creating MultiThreadedExecutor...")
             executor = MultiThreadedExecutor()
             executor.add_node(node)
             print(f"Node '{node.get_name()}' added. Spinning executor...")
             executor.spin()
        else:
             print("Node initialization failed or shutdown requested early.")

    except KeyboardInterrupt: print("Keyboard interrupt received.")
    except ExternalShutdownException: print("External shutdown request received.")
    except Exception as e: print(f"Unhandled exception in main: {e}\n{traceback.format_exc()}")
    finally:
        print("Executing finally block...")
        if executor:
            print("Shutting down executor...")
            executor.shutdown()
            print("Executor shutdown complete.")
        # Ensure node destruction happens *after* executor shutdown if possible
        if node:
            print("Destroying node...")
            try: node.destroy_node()
            except Exception as e: print(f"Error during node destruction: {e}")
            print("Node destruction finished.")

        if rclpy.ok():
            print("Shutting down rclpy...")
            rclpy.shutdown()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()

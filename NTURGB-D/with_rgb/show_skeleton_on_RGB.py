import numpy as np
import cv2
import argparse
import os
import sys


# --- Copied and adapted from Python/txt2npy.py ---
def read_skeleton_file(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=False):
    """
    Reads a skeleton file (.skeleton).

    Args:
        file_path (str): Path to the skeleton file.
        save_skelxyz (bool): Whether to save 3D coordinates.
        save_rgbxy (bool): Whether to save 2D RGB coordinates.
        save_depthxy (bool): Whether to save 2D depth coordinates.

    Returns:
        dict: A dictionary containing skeleton data.
              Structure:
              {
                  'file_name': str,
                  'nbodys': list[int], # Number of bodies in each frame
                  'njoints': int,      # Number of joints (e.g., 25)
                  'skel_body{i}': np.array(N, njoints, 3), # 3D coords for body i
                  'rgb_body{i}': np.array(N, njoints, 2),  # 2D RGB coords for body i
                  'depth_body{i}': np.array(N, njoints, 2) # 2D depth coords for body i
              }
              where N is the number of frames.
              Keys for coordinates only exist if the corresponding save_ flag is True.
              Body indices {i} only exist for bodies present in the data.
    """
    try:
        f = open(file_path, "r")
        datas = f.readlines()
    except FileNotFoundError:
        print(f"Error: Skeleton file not found at {file_path}")
        return None
    finally:
        if "f" in locals() and not f.closed:
            f.close()

    max_body = 2  # NTU RGB+D max 2 bodies
    njoints = 25

    nframe = int(datas[0].strip())
    bodymat = dict()
    bodymat["file_name"] = os.path.basename(file_path)
    bodymat["nbodys"] = []
    bodymat["njoints"] = njoints
    bodymat["nframes"] = nframe

    # Initialize data holders
    if save_skelxyz:
        for b in range(max_body):
            bodymat[f"skel_body{b}"] = np.zeros(shape=(nframe, njoints, 3))
    if save_rgbxy:
        for b in range(max_body):
            bodymat[f"rgb_body{b}"] = np.zeros(shape=(nframe, njoints, 2))
    if save_depthxy:
        for b in range(max_body):
            bodymat[f"depth_body{b}"] = np.zeros(shape=(nframe, njoints, 2))

    cursor = 0
    actual_max_bodies = 0  # Track the max bodies actually found
    for frame_idx in range(nframe):
        cursor += 1
        if cursor >= len(datas):
            break  # Avoid index error if file is truncated
        bodycount = int(datas[cursor].strip())
        bodymat["nbodys"].append(bodycount)
        actual_max_bodies = max(actual_max_bodies, bodycount)

        for body_idx in range(bodycount):
            cursor += 1
            if cursor >= len(datas):
                break
            # bodyinfo = datas[cursor].strip().split(' ') # Body info line (we don't use it here)
            cursor += 1
            if cursor >= len(datas):
                break
            num_joints_in_frame = int(datas[cursor].strip())  # Should be 25

            if body_idx >= max_body:  # Skip extra bodies if more than max_body defined
                cursor += num_joints_in_frame
                continue

            for joint_idx in range(num_joints_in_frame):
                cursor += 1
                if cursor >= len(datas):
                    break
                jointinfo = datas[cursor].strip().split(" ")
                jointinfo = np.array(list(map(float, jointinfo)))

                # Ensure jointinfo has enough elements (at least 7 for RGB)
                if len(jointinfo) < 7 and save_rgbxy:
                    print(f"Warning: Incomplete joint data at frame {frame_idx}, body {body_idx}, joint {joint_idx}")
                    continue  # Skip this joint if data is missing

                if save_skelxyz and len(jointinfo) >= 3:
                    bodymat[f"skel_body{body_idx}"][frame_idx, joint_idx] = jointinfo[:3]
                if save_depthxy and len(jointinfo) >= 5:
                    bodymat[f"depth_body{body_idx}"][frame_idx, joint_idx] = jointinfo[3:5]
                if save_rgbxy and len(jointinfo) >= 7:
                    bodymat[f"rgb_body{body_idx}"][frame_idx, joint_idx] = jointinfo[5:7]
            if cursor >= len(datas):
                break
        if cursor >= len(datas):
            break

    # Remove placeholders for bodies that never appeared
    for b in range(actual_max_bodies, max_body):
        if save_skelxyz and f"skel_body{b}" in bodymat:
            del bodymat[f"skel_body{b}"]
        if save_rgbxy and f"rgb_body{b}" in bodymat:
            del bodymat[f"rgb_body{b}"]
        if save_depthxy and f"depth_body{b}" in bodymat:
            del bodymat[f"depth_body{b}"]

    # Update max body count based on actual data
    bodymat["max_bodies"] = actual_max_bodies

    return bodymat


# --- Load NPY skeleton file ---
def load_skeleton_npy(npy_path):
    """
    Loads skeleton data from a .npy file.

    Args:
        npy_path (str): Path to the .npy skeleton file.

    Returns:
        dict: A dictionary containing skeleton data.
    """
    try:
        skeleton_data = np.load(npy_path, allow_pickle=True).item()

        # Basic validation
        if "njoints" not in skeleton_data or "rgb_body0" not in skeleton_data:
            print(f"Error: Loaded .npy file {npy_path} seems incomplete or has unexpected structure.")
            return None

        # Add nframes field if missing (computed from rgb_body0 shape)
        if "nframes" not in skeleton_data and "rgb_body0" in skeleton_data:
            skeleton_data["nframes"] = skeleton_data["rgb_body0"].shape[0]
            print(f"Added missing 'nframes' field: {skeleton_data['nframes']}")

        # Count how many bodies we have (needed for max_bodies)
        max_body = 0
        while f"rgb_body{max_body}" in skeleton_data:
            max_body += 1

        # Always add max_bodies field
        skeleton_data["max_bodies"] = max_body
        if max_body > 0:
            print(f"Added 'max_bodies' field with value: {max_body}")

        # Add nbodys field if missing
        if "nbodys" not in skeleton_data:
            # Create nbodys list with constant value (assuming all bodies present in all frames)
            if "nframes" in skeleton_data and max_body > 0:
                skeleton_data["nbodys"] = [max_body] * skeleton_data["nframes"]
                print(f"Added missing 'nbodys' field assuming {max_body} bodies per frame")

        return skeleton_data
    except FileNotFoundError:
        print(f"Error: Skeleton NPY file not found at {npy_path}")
        return None
    except Exception as e:
        print(f"Error loading skeleton NPY file {npy_path}: {e}")
        return None


# --- Main plotting function ---
def show_skeleton_on_rgb(skeleton_data, rgb_video_path, output_video_path=None, display=True):
    """
    Draws skeleton data onto RGB video frames.

    Args:
        skeleton_data (dict): Skeleton data dictionary from read_skeleton_file.
        rgb_video_path (str): Path to the RGB video file.
        output_video_path (str, optional): Path to save the output video. Defaults to None.
        display (bool): Whether to display the video frames in a window. Defaults to True.
    """
    if skeleton_data is None:
        return

    # Define connections between joints (0-based index)
    # Matches the structure in the Matlab code
    connecting_joint = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
    if len(connecting_joint) != skeleton_data["njoints"]:
        print(f"Warning: connecting_joint array length ({len(connecting_joint)}) " f"does not match njoints ({skeleton_data['njoints']})")
        # Adjust or handle error as needed

    # Colors (BGR format for OpenCV)
    joint_color = (0, 255, 0)  # Green
    bone_color = (0, 0, 255)  # Red
    joint_radius = 7
    bone_thickness = 5  # Adjusted thickness similar to Matlab's 7x7 patch

    # --- Video Reading ---
    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {rgb_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_skeleton = skeleton_data["nframes"]

    print(f"Video Info: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames_video} frames")
    print(f"Skeleton Info: {total_frames_skeleton} frames")

    if total_frames_video != total_frames_skeleton:
        print("Warning: Video frame count and skeleton frame count differ.")
        # Decide how to handle: use min count, skip, etc.
        # Using min count for now.
        num_frames_to_process = min(total_frames_video, total_frames_skeleton)
    else:
        num_frames_to_process = total_frames_video

    # --- Video Writing (Optional) ---
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Or use 'XVID', 'MJPG', etc.
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            print(f"Error: Could not open video writer for path: {output_video_path}")
            writer = None  # Disable writing if opening failed

    # --- Process Frames ---
    for frame_idx in range(num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} from video.")
            break

        num_bodies_in_frame = skeleton_data["nbodys"][frame_idx]

        for body_idx in range(min(num_bodies_in_frame, skeleton_data["max_bodies"])):
            if f"rgb_body{body_idx}" not in skeleton_data:
                continue  # Skip if body data doesn't exist

            joints = skeleton_data[f"rgb_body{body_idx}"][frame_idx]  # Shape: (njoints, 2)

            # Draw bones (connections)
            for joint_idx in range(skeleton_data["njoints"]):
                connected_idx = connecting_joint[joint_idx]

                # Get coordinates (ensure they are valid)
                x1, y1 = joints[joint_idx]
                x2, y2 = joints[connected_idx]

                # Check for zero coordinates (often indicate untracked joints)
                if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                    continue

                # Convert to integer pixel coordinates
                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))

                # Check if points are within frame boundaries (optional but good practice)
                if 0 <= pt1[0] < frame_width and 0 <= pt1[1] < frame_height and 0 <= pt2[0] < frame_width and 0 <= pt2[1] < frame_height:
                    cv2.line(frame, pt1, pt2, bone_color, bone_thickness)

            # Draw joints
            for joint_idx in range(skeleton_data["njoints"]):
                x, y = joints[joint_idx]

                # Check for zero coordinates
                if x == 0 and y == 0:
                    continue

                # Convert to integer pixel coordinates
                center = (int(round(x)), int(round(y)))

                # Check if point is within frame boundaries
                if 0 <= center[0] < frame_width and 0 <= center[1] < frame_height:
                    cv2.circle(frame, center, joint_radius, joint_color, -1)  # -1 fills the circle

        # Write frame to output video
        if writer:
            writer.write(frame)

        # Display frame
        if display:
            cv2.imshow("Skeleton Visualization", frame)
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # --- Cleanup ---
    cap.release()
    if writer:
        writer.release()
        print(f"Output video saved to: {output_video_path}")
    if display:
        cv2.destroyAllWindows()


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw skeleton data on RGB video frames.")
    parser.add_argument("skeleton_file", help="Path to the skeleton file (.skeleton or .npy).")
    parser.add_argument("rgb_video", help="Path to the corresponding RGB video file.")
    parser.add_argument("-o", "--output", help="Path to save the output video file (e.g., output.mp4).")
    parser.add_argument("--no-display", action="store_true", help="Do not display the video while processing.")

    args = parser.parse_args()

    print(f"Reading skeleton file: {args.skeleton_file}")

    # Determine file type and use appropriate loading function
    if args.skeleton_file.endswith(".npy"):
        skeleton_data = load_skeleton_npy(args.skeleton_file)
    else:
        skeleton_data = read_skeleton_file(args.skeleton_file, save_rgbxy=True, save_skelxyz=False, save_depthxy=False)

    if skeleton_data:
        print(f"Processing video: {args.rgb_video}")
        show_skeleton_on_rgb(skeleton_data, args.rgb_video, args.output, display=not args.no_display)
        print("Processing finished.")
    else:
        print("Could not read skeleton data. Exiting.")
        sys.exit(1)

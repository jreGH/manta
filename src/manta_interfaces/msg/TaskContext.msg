# What the focused actor (diver / robot) is currently doing.
# Published by the focused actor's system or loaded from a pre-dive mission plan.

string task_id
string task_type             # ordnance_disposal | survey | transit | ...
geometry_msgs/PoseStamped target_pose
float32 safety_radius        # meters — anything inside this radius is a danger
string[] priority_actor_types  # labels the task is specifically sensitive to

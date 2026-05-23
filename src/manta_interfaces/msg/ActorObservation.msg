# Raw report about an actor from any source.
# Classification axes are strings so new actor types require no schema change.

string actor_id
string label           # human-readable type: "shark", "torpedo", "helicopter", etc.
string domain          # surface | subsurface | aerial | seabed
string agency          # human | autonomous | biological | passive
string cooperativity   # cooperative | uncooperative | adversarial

geometry_msgs/PoseWithCovariance pose
geometry_msgs/Twist velocity
builtin_interfaces/Time observed_at
string source          # direct | gateway | inference

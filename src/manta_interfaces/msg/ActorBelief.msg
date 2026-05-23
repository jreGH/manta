# World-model estimate for a single actor, projected forward to the current time.

string actor_id
string label
string domain
string agency
string cooperativity

geometry_msgs/PoseWithCovariance projected_pose   # forward-projected to now
float32 staleness_seconds
float32 confidence
string[] intent_labels    # e.g. ["approaching_diver", "foraging", "inbound"]
float32[] intent_probs

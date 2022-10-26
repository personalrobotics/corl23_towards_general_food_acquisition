#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import tf_conversions.posemath as pm
import tf.transformations

class ForqueBodyTransformer:
    def __init__(self):
        # Initialize the subscriber and publishers
        self.sub = rospy.Subscriber("vrpn_client_node/ForqueBody/pose", PoseStamped, self.forque_body_callback)
        self.pub = rospy.Publisher('forque_tip', PoseStamped, queue_size=1)

        # self.transform times a vector in forktip frame will give a vector in ForqueBody frame
        self.transform = np.linalg.inv(np.array(tf.transformations.rotation_matrix(np.pi, (0,0,1)) @ [
            [0.7242579781,	 0.4459343445,	-0.5259778548,	-0.003609486659],
            [-0.3048250761,	 0.8912113169,	 0.3359484526,	 0.00490096214 ],
            [0.6185878081,	-0.08298122464,	 0.7813997912,	-0.2091574497  ],
            [0,	             0,	             0,	             1             ],
        ]))
        print(tf.transformations.translation_from_matrix(self.transform), tf.transformations.quaternion_from_matrix(self.transform))

    def forque_body_callback(self, msg):
        """
        Takes in the PoseStamped msg for the center of the Forque's rigid body
        in Motive, and publishes the PoseStamped msg for the forque_tip.
        """
        B = tf.transformations.quaternion_matrix(np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]))
        B[0][3] = msg.pose.position.x
        B[1][3] = msg.pose.position.y
        B[2][3] = msg.pose.position.z
        forktip_pose = np.matmul(B, self.transform)
        q = tf.transformations.quaternion_from_matrix(forktip_pose)

        # frame = pm.fromMsg(msg.pose)
        # transformed_pose_stamped = PoseStamped()
        # transformed_pose_stamped.header = msg.header
        # transformed_pose_stamped.pose = pm.toMsg(frame * self.static_transform)

        transformed_pose_stamped = PoseStamped()
        transformed_pose_stamped.header = msg.header
        transformed_pose_stamped.pose.position.x = forktip_pose[0,3]
        transformed_pose_stamped.pose.position.y = forktip_pose[1,3]
        transformed_pose_stamped.pose.position.z = forktip_pose[2,3]
        transformed_pose_stamped.pose.orientation.x = q[0]
        transformed_pose_stamped.pose.orientation.y = q[1]
        transformed_pose_stamped.pose.orientation.z = q[2]
        transformed_pose_stamped.pose.orientation.w = q[3]

        self.pub.publish(transformed_pose_stamped)

if __name__ == '__main__':

    rospy.init_node('publish_forque_transform')

    forque_body_transformer = ForqueBodyTransformer()

    rospy.spin()

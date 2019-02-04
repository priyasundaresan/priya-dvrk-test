import dvrk
import PyKDL
import numpy as np

def radians_to_degrees(rad):
	return 180. / np.pi * np.array(rad)

def degrees_to_radians(deg):
	return np.pi / 180. * np.array(deg)

class robot(dvrk.psm):

	def move_to_pos_rot(self, pos, rot):
		"""
		Moves the PSM to the pose specified by (pos, rot).
		"""
		px, py, pz = pos
		z, y, x = degrees_to_radians(rot)
		pos = PyKDL.Vector(px, py, pz)
		rot = PyKDL.Rotation.EulerZYX(z, y, x)
		pose = PyKDL.Frame(rot, pos)
		self.move_to_pose(pose)

	def move_to_pose(self, pose):
		"""
		Moves the PSM to the pose specified by the supplied PyKDL Frame.
		TODO(bthananjeyan): add workspace limits
		"""
		self.move(pose)

	@property
	def position(self):
		pos = self.get_current_position().p
		return np.array((pos.x(), pos.y(), pos.z()))

	@property
	def rotation(self):
		return radians_to_degrees(np.array(self.get_current_position().M.GetEulerZYX()))

	@property
	def pose(self):
		return self.position, self.rotation

if __name__ == '__main__':
	import IPython; IPython.embed()

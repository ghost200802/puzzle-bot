import sys
sys.path.insert(0, 'src')
from common.vector import Vector, Candidate
from common import util
import math

pid = 2
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()

vec_offset = 1 if p.scalar < 2 else 2
vec_len_for_angle = round(3 * p.scalar)
n = len(p.vertices)
i = 2367

a_ih, _ = util.colinearity(from_point=p.vertices[i], to_points=util.slice(p.vertices, i-vec_len_for_angle-vec_offset, i-vec_offset-1))
a_ij, _ = util.colinearity(from_point=p.vertices[i], to_points=util.slice(p.vertices, i+vec_offset+1, i+vec_len_for_angle+vec_offset))
a_ih_orig = a_ih
a_ih = a_ih + math.pi

p_h = (p.vertices[i][0] + 10 * math.cos(a_ih), p.vertices[i][1] + 10 * math.sin(a_ih))
p_j = (p.vertices[i][0] + 10 * math.cos(a_ij), p.vertices[i][1] + 10 * math.sin(a_ij))

angle_hij = util.counterclockwise_angle_between_vectors(p_h, p.vertices[i], p_j)
a_ic = util.angle_between(p.vertices[i], p.centroid)
midangle = util.angle_between(p.vertices[i], util.midpoint(p_h, p_j))
offset_from_center = util.compare_angles(midangle, a_ic)

is_pointed_toward_center = offset_from_center < angle_hij / 2
if not is_pointed_toward_center and angle_hij < 90 * math.pi/180:
    is_pointed_toward_center = abs(offset_from_center) <= (45 * math.pi/180)

print(f"Point: ({p.vertices[i][0]},{p.vertices[i][1]}) i={i}")
print(f"a_ih (seg dir) = {a_ih_orig*180/math.pi:.1f}°, after +180 = {a_ih*180/math.pi:.1f}°")
print(f"a_ij (seg dir) = {a_ij*180/math.pi:.1f}°")
print(f"angle_hij = {angle_hij*180/math.pi:.1f}°")
print(f"offset_from_center = {offset_from_center*180/math.pi:.1f}°")
print(f"is_pointed_toward_center = {is_pointed_toward_center}")
print(f"midangle = {midangle*180/math.pi:.1f}°, a_ic = {a_ic*180/math.pi:.1f}°")

import freenect
import cv2
import frame_convert
import numpy as np
# cv2.namedWindow('Depth')
# cv2.namedWindow('Video')
# print('Press ESC in window to stop')


def get_depth():
    return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(freenect.sync_get_video()[0])

def getDepthMap():	
	depth, timestamp = freenect.sync_get_depth()
 
	np.clip(depth, 0, 2**10 - 1, depth)
	depth >>= 2
	depth = depth.astype(np.uint8)
 
	return depth


# while 1:
#     depth = get_depth()
    
#     print("done")
#     cv2.imshow('Depth', 255-depth)
#     cv2.imshow('Video', get_video())
#     if cv2.waitKey(10) == 27:
#         break

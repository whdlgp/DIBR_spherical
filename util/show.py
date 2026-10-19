import numpy as np
import cv2


def depth_to_show(depth):
    valid = np.isfinite(depth) & (depth > 0)
    depth_show = np.zeros_like(depth)
    depth_show[valid] = depth[valid]
    depth_show = ((depth_show - depth_show[valid].min()) / (depth_show[valid].max() - depth_show[valid].min()) * 255).astype(np.uint8)
    return cv2.cvtColor(depth_show, cv2.COLOR_GRAY2BGR)


def disp_to_show(disp):
    valid = np.isfinite(disp) & (disp > 0)
    disp_show = np.zeros_like(disp)
    disp_show[valid] = disp[valid]
    disp_show = ((disp_show - disp_show[valid].min()) / (disp_show[valid].max() - disp_show[valid].min()) * 255).astype(np.uint8)
    return cv2.cvtColor(disp_show, cv2.COLOR_GRAY2BGR)


def show(img, show_width=1280, title="Show"):
    show_scale = show_width / img.shape[1]
    show_img = cv2.resize(img, None, fx=show_scale, fy=show_scale)

    cv2.imshow(title, show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
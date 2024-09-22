import random
import numpy as np
import cv2

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import layout
from bokeh.models import Div


def draw_matches(img1, kp1, img2, kp2, matches, orientation='vertical'):
    # Convert images to BGR format if they are grayscale
    if len(img1.shape) == 2:  # Grayscale to BGR
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:  # Grayscale to BGR
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    img1 = put_text(img1, "top_center", "Previous Frame")
    img2 = put_text(img2, "top_center", "Current Frame")

    # Create a new image for stacking
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    if orientation == 'vertical':
        combined_height = height1 + height2
        combined_width = max(width1, width2)
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_img[:height1, :width1] = img1
        combined_img[height1:combined_height, :width2] = img2

        # Adjust keypoints for the second image
        adjusted_kp2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + height1, kp.size) for kp in kp2]
    elif orientation == 'horizontal':
        combined_height = max(height1, height2)
        combined_width = width1 + width2
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_img[:height1, :width1] = img1
        combined_img[:height2, width1:combined_width] = img2

        # Adjust keypoints for the second image
        adjusted_kp2 = [cv2.KeyPoint(kp.pt[0] + width1, kp.pt[1], kp.size) for kp in kp2]
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")

    # Draw matches on the combined image
    for match in matches:
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(adjusted_kp2[match.trainIdx].pt[0]), int(adjusted_kp2[match.trainIdx].pt[1]))

        # Draw lines between the matched keypoints
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(combined_img, pt1, pt2, color, 2)

        # Draw circles around keypoints
        cv2.circle(combined_img, pt1, 5, (255, 255, 255), -1)  # Keypoint in first image
        cv2.circle(combined_img, pt2, 5, (255, 255, 255), -1)  # Keypoint in second image

    return combined_img


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def visualize_paths(gt_path, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)
    source = ColumnDataSource(data=dict(gtx=gt_path[:, 0], gty=gt_path[:, 1],
                                        px=pred_path[:, 0], py=pred_path[:, 1],
                                        diffx=np.arange(len(diff)), diffy=diff,
                                        disx=xs, disy=ys))

    fig1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                  x_axis_label="x", y_axis_label="y")
    fig1.circle("gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT")
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle("px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed")
    fig1.legend.click_policy = "hide"

    fig2 = figure(title="Error", tools=tools, width_policy="max", toolbar_location="above",
                  x_axis_label="frame", y_axis_label="error")
    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [fig1, fig2],
                 ], sizing_mode='scale_width'))


def play_trip(l_frames, r_frames=None, lat_lon=None, timestamps=None, color_mode=False, waite_time=100, win_name="Trip"):
    l_r_mode = r_frames is not None

    if not l_r_mode:
        r_frames = [None]*len(l_frames)

    frame_count = 0
    for i, frame_step in enumerate(zip(l_frames, r_frames)):
        img_l, img_r = frame_step

        if not color_mode:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            if img_r is not None:
                img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)

        if img_r is not None:
            img_l = put_text(img_l, "top_center", "Left")
            img_r = put_text(img_r, "top_center", "Right")
            show_image = np.vstack([img_l, img_r])
        else:
            show_image = img_l

        show_image = put_text(show_image, "top_left", "Press ESC to stop")
        show_image = put_text(show_image, "top_right", f"Frame: {frame_count}/{len(l_frames)}")

        if timestamps is not None:
            time = timestamps[i]
            show_image = put_text(show_image, "bottom_right", f"{time}")

        if lat_lon is not None:
            lat, lon = lat_lon[i]
            show_image = put_text(show_image, "bottom_left", f"{lat}, {lon}")

        cv2.imshow(win_name, show_image)

        key = cv2.waitKey(waite_time)
        if key == 27:  # ESC
            break
        frame_count += 1
    cv2.destroyWindow(win_name)


def put_text(image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    if not isinstance(org, tuple):
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        org_w = 0
        org_h = 0

        h, w, *_ = image.shape

        place_h, place_w = org.split("_")

        if place_h == "top":
            org_h = label_height
        elif place_h == "bottom":
            org_h = h
        elif place_h == "center":
            org_h = h // 2 + label_height // 2

        if place_w == "left":
            org_w = 0
        elif place_w == "right":
            org_w = w - label_width
        elif place_w == "center":
            org_w = w // 2 - label_width // 2

        org = (org_w, org_h)

    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image



import os
import svgwrite  # conda install -c omnia svgwrite=1.1.6
from IPython.display import SVG, display
import numpy as np
def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)
    return (min_x, max_x, min_y, max_y)


def draw_strokes(data,
                 svg_filename='sample.svg',
                 factor=0.2,
                 show_pen_sequence=False,
                 who_draw_the_stroke=None):
    if not os.path.exists('./sketch_rnn/svg/'):
        os.makedirs(os.path.dirname(svg_filename))
        
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing('./sketch_rnn/svg/'+svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "M"
    xs = []
    ys = []
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "M"
        elif (command != "L"):
            command = "L"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        xs.append(abs_x)
        ys.append(abs_y)
        lift_pen = data[i, 2]
        p += command + str(abs_x) + "," + str(abs_y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    color = 'black'
    if show_pen_sequence:
        turn = 0
        for i in range(1, len(xs)):
            dwg.add(
                dwg.text(
                    '{}'.format(i),
                    insert=(xs[i], ys[i]),
                    font_size="10px",
                    fill=color))
            if who_draw_the_stroke is not None:
                if data[i, 2] == 1:
                    color = 'red' if who_draw_the_stroke[
                        turn] == 0 else 'black'
                    turn += 1
    display(SVG(dwg.tostring()))
    return SVG(dwg.tostring())
def preprocess(sketches, limit=1000):
    raw_data = []
    seq_len = []

    for i in range(len(sketches)):
        data = sketches[i]
        # removes large gaps from the data
        data = np.minimum(data, limit)
        data = np.maximum(data, -limit)
        data = np.array(data, dtype=np.float32)
        raw_data.append(data)
        seq_len.append(len(data))
    seq_len = np.array(seq_len)  # nstrokes for each sketch
    idx = np.argsort(seq_len)
    sketches = []
    for i in range(len(seq_len)):
        sketches.append(raw_data[idx[i]])
    return raw_data


def calculate_normalizing_scale_factor(sketches):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(sketches)):
        for j in range(len(sketches[i])):
            data.append(sketches[i][j, 0])
            data.append(sketches[i][j, 1])
    data = np.array(data)
    return np.std(data)


def normalize(sketches, scale_factor):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    result = []
    for i in range(len(sketches)):
        tmp = sketches[i]
        tmp[:, 0:2] /= scale_factor
        result.append(tmp)
    return result


def to_big_strokes(strokes):
    """Converts from stroke-3 to stroke-5 format and pads to given length, but does not insert special start token)."""

    result = np.zeros((len(strokes), 5), dtype=float)
    l = len(strokes)
    result[0:l, 0:2] = strokes[:, 0:2]
    result[0:l, 3] = strokes[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def to_big_sketches(sketches):
    result = []
    for i in range(len(sketches)):
        sketch = to_big_strokes(sketches[i])
        result.append(sketch)
    return result


def to_normal_strokes(big_strokes):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_strokes)):
        if big_strokes[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_strokes)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_strokes[0:l, 0:2]
    result[:, 2] = big_strokes[0:l, 3]
    return result
# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):

    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end

    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)
def render_imgs(strokes, img_size, max_seq_len):
    """ 
      convert 5-strokes format to image
          args:
              sketches: shape(data_size, 5, max_seq_len)
      """

    xy = np.cumsum(strokes[:, :, 0:2], axis=1)  # (data_size, max_seq, 2)
    min_xy = np.min(xy, axis=(1), keepdims=True)  # (data_size, 1, 2)
    xy = xy - min_xy  # (data_size, max_seq, 2)
    max_xy = np.max(xy, axis=(1), keepdims=True)  # (data_size, 1, 2)
    max_xy = np.where(max_xy == 0, np.ones([len(strokes), 1, 2]),
                      max_xy)  # avoid divide by 0
    xy = xy / max_xy  # (data_size, max_seq, 2)
    xy = xy * (img_size - 1)  # (data_size, max_seq, 2)

    strokes_idx = np.tile(np.arange(len(strokes))[:, None],
                          [1, img_size])  # (data_size, img_size)
    interpolate_line = np.tile(
        np.reshape(
            np.arange(img_size).astype(np.float32) / (img_size - 1),
            [1, img_size, 1]), [len(strokes), 1, 2])

    def interpolate(p1, p2):
        p1 = np.reshape(p1, [-1, 1, 2])
        p2 = np.reshape(p2, [-1, 1, 2])
        return (1 - interpolate_line
                ) * p1 + interpolate_line * p2  # (data_size, img_size, 2)

    images = np.zeros([len(strokes), img_size, img_size])
    render_next = np.ones(len(images), dtype=np.bool)
    for idx in range(max_seq_len - 1):
        p1 = xy[:, idx]
        p2 = xy[:, idx + 1]
        # if p1 is connect to p2, draw a line between them
        connect = np.where(
            np.logical_and(strokes[:, idx, 3] > strokes[:, idx, 2],
                           strokes[:, idx, 3] > strokes[:, idx, 4]),
            np.zeros(len(images), dtype=np.bool),
            np.ones(len(images), dtype=np.bool))

        p_interpolate_line = interpolate(p1, p2).astype(
            np.int32)  # (data_size, img_size, 2)
        x_idx = np.where(connect[:, None], p_interpolate_line[:, :, 0],
                         np.tile(xy[:, idx, None, 0], [1,
                                                       img_size]).astype(np.int32))
        y_idx = np.where(connect[:, None], p_interpolate_line[:, :, 1],
                         np.tile(xy[:, idx, None, 1], [1,
                                                       img_size]).astype(np.int32))
        images[strokes_idx, x_idx, y_idx] = 1
    images = np.rot90(images, -1, axes=(1, 2))
    return images

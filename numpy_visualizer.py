import numpy as np
import cv2
import tkinter as tk

def draw_cube(img, start_point, size, depth, color, thickness):
    x, y = start_point
    l = size
    d = depth

    front_face = np.array([[x, y], [x + l, y], [x + l, y + l], [x, y + l]], np.int32)
    top_face = np.array([[x, y],
                         [x + l // 2, y - d // 2],
                         [x + l + l // 2, y - d // 2],
                         [x + l, y]], np.int32)
    side_face = np.array([[x + l, y],
                          [x + l + l // 2, y - d // 2],
                          [x + l + l // 2, y + l - d // 2],
                          [x + l, y + l]], np.int32)

    cv2.fillPoly(img, [front_face], color)
    cv2.fillPoly(img, [top_face], (max(color[0] - 50, 0), max(color[1] - 50, 0), max(color[2] - 50, 0)))
    cv2.fillPoly(img, [side_face], (max(color[0] - 100, 0), max(color[1] - 100, 0), max(color[2] - 100, 0)))

    cv2.polylines(img, [front_face], isClosed=True, color=(0, 0, 0), thickness=thickness)
    cv2.polylines(img, [top_face], isClosed=True, color=(0, 0, 0), thickness=thickness)
    cv2.polylines(img, [side_face], isClosed=True, color=(0, 0, 0), thickness=thickness)


def crop_to_content(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        x = max(0, x - 20)
        y = max(0, y - 20)
        w = min(img.shape[1] - x, w + 60)
        h = min(img.shape[0] - y, h + 60)
        return img[:y + h, :x + w]
    return img


def visualize_array(array, slice_dims=None):
    dims = array.ndim

    img = np.ones((2000, 2000, 3), dtype=np.uint8) * 255

    size = 50
    depth = 20
    offset_x = 50
    offset_y = 50
    spacing = 30

    if dims == 1:
        # 1D: array[x]
        sx = array.shape[0]
        for x in range(sx):
            start_x = offset_x + x * (size + spacing)
            start_y = offset_y

            color = (255, 200, 150)
            if slice_dims and (x in slice_dims[0]):
                color = (150, 255, 150)

            draw_cube(img, (start_x, start_y), size, depth, color, 2)

            text = str(array[x])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = start_x + (size - text_size[0]) // 2
            text_y = start_y + (size + text_size[1]) // 2
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)

    elif dims == 2:
        # 2D: array[x, y]
        sx, sy = array.shape
        for x in range(sx):
            for y in range(sy):
                start_x = offset_x + x * (size + spacing)
                start_y = offset_y + y * (size + spacing)

                color = (255, 200, 150)
                if slice_dims and (x in slice_dims[0]) and (y in slice_dims[1]):
                    color = (150, 255, 150)

                draw_cube(img, (start_x, start_y), size, depth, color, 2)

                # Исправлено индексирование: теперь array[x, y], а не array[y, x]
                text = str(array[x, y])
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = start_x + (size - text_size[0]) // 2
                text_y = start_y + (size + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)

    elif dims == 3:
        # 3D: array[x, y, z]
        sx, sy, sz = array.shape
        for x in range(sx):
            for y in range(sy-1, -1, -1):
                for z in range(sz-1, -1, -1):
                    start_x = offset_x + x * (size + spacing) + z * (depth // 2 + spacing // 2)
                    start_y = offset_y + y * (size + spacing) - z * (depth // 2 + spacing // 2) + sz*30

                    color = (255, 200, 150)
                    if slice_dims and (x in slice_dims[0]) and (y in slice_dims[1]) and (z in slice_dims[2]):
                        color = (150, 255, 150)

                    draw_cube(img, (start_x, start_y), size, depth, color, 2)

                    # Исправлено индексирование: теперь array[x, y, z], а не array[z, y, x]
                    text = str(array[x, y, z])
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = start_x + (size - text_size[0]) // 2
                    text_y = start_y + (size + text_size[1]) // 2
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cropped_img = crop_to_content(img)

    h, w = cropped_img.shape[:2]
    max_dim = max(h, w)
    scale = 600 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    y_offset = (600 - new_h) // 2
    x_offset = (600 - new_w) // 2
    final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    cv2.namedWindow('Array', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Array', 600, 600)
    cv2.imshow('Array', final_img)
    cv2.waitKey(20)


def update_slices(*args):
    dim = dim_var.get()
    if dim == 1:
        slice_2_frame.grid_remove()
        slice_3_frame.grid_remove()
    elif dim == 2:
        slice_2_frame.grid()
        slice_3_frame.grid_remove()
    elif dim == 3:
        slice_2_frame.grid()
        slice_3_frame.grid()
    start_visualization()


def start_visualization(*args):
    dim = dim_var.get()
    s1 = size_1.get()
    s2 = size_2.get() if dim > 1 else 1
    s3 = size_3.get() if dim > 2 else 1
    sizes = [s1, s2, s3][:dim]

    s1_start_val = slice_1_start.get()
    s1_end_val = slice_1_end.get()
    s1_step_val = slice_1_step.get() if slice_1_step.get() > 0 else 1

    s2_start_val = slice_2_start.get() if dim > 1 else 0
    s2_end_val = slice_2_end.get() if dim > 1 else 0
    s2_step_val = slice_2_step.get() if (dim > 1 and slice_2_step.get() > 0) else 1

    s3_start_val = slice_3_start.get() if dim > 2 else 0
    s3_end_val = slice_3_end.get() if dim > 2 else 0
    s3_step_val = slice_3_step.get() if (dim > 2 and slice_3_step.get() > 0) else 1

    slices = [range(s1_start_val, s1_end_val, s1_step_val)]
    if dim > 1:
        slices.append(range(s2_start_val, s2_end_val, s2_step_val))
    if dim > 2:
        slices.append(range(s3_start_val, s3_end_val, s3_step_val))

    if dim == 1:
        array = np.arange(sizes[0])
    elif dim == 2:
        array = np.arange(sizes[0] * sizes[1]).reshape(sizes[0], sizes[1])
    elif dim == 3:
        array = np.arange(sizes[0] * sizes[1] * sizes[2]).reshape(sizes[0], sizes[1], sizes[2])

    visualize_array(array, slices)


root = tk.Tk()
root.title("Визуализация массива")

dim_var = tk.IntVar(value=1)

tk.Label(root, text="Выберите размерность:").grid(row=0, column=0, sticky="w")
for i in range(1, 4):
    tk.Radiobutton(root, text=f"{i}D", variable=dim_var, value=i, command=update_slices).grid(row=0, column=i, sticky="w")

size_1 = tk.IntVar(value=5)
size_2 = tk.IntVar(value=5)
size_3 = tk.IntVar(value=5)

tk.Label(root, text="Размер по оси X:").grid(row=1, column=0, sticky="w")
tk.Scale(root, from_=1, to=20, orient="horizontal", variable=size_1, command=start_visualization).grid(row=1, column=1, columnspan=3, sticky="w")

tk.Label(root, text="Размер по оси Y:").grid(row=2, column=0, sticky="w")
tk.Scale(root, from_=1, to=20, orient="horizontal", variable=size_2, command=start_visualization).grid(row=2, column=1, columnspan=3, sticky="w")

tk.Label(root, text="Размер по оси Z:").grid(row=3, column=0, sticky="w")
tk.Scale(root, from_=1, to=20, orient="horizontal", variable=size_3, command=start_visualization).grid(row=3, column=1, columnspan=3, sticky="w")

slice_1_start = tk.IntVar(value=0)
slice_1_end = tk.IntVar(value=5)
slice_1_step = tk.IntVar(value=1)

slice_2_start = tk.IntVar(value=0)
slice_2_end = tk.IntVar(value=5)
slice_2_step = tk.IntVar(value=1)

slice_3_start = tk.IntVar(value=0)
slice_3_end = tk.IntVar(value=5)
slice_3_step = tk.IntVar(value=1)

slice_1_frame = tk.LabelFrame(root, text="Срез по оси X")
slice_1_frame.grid(row=4, column=0, columnspan=4, sticky="w")
tk.Label(slice_1_frame, text="Начало:").grid(row=0, column=0, sticky="w")
tk.Scale(slice_1_frame, from_=0, to=20, orient="horizontal", variable=slice_1_start, command=start_visualization).grid(row=0, column=1, sticky="w")
tk.Label(slice_1_frame, text="Конец:").grid(row=0, column=2, sticky="w")
tk.Scale(slice_1_frame, from_=0, to=20, orient="horizontal", variable=slice_1_end, command=start_visualization).grid(row=0, column=3, sticky="w")
tk.Label(slice_1_frame, text="Шаг:").grid(row=1, column=0, sticky="w")
tk.Scale(slice_1_frame, from_=1, to=10, orient="horizontal", variable=slice_1_step, command=start_visualization).grid(row=1, column=1, sticky="w")

slice_2_frame = tk.LabelFrame(root, text="Срез по оси Y")
slice_2_frame.grid(row=5, column=0, columnspan=4, sticky="w")
tk.Label(slice_2_frame, text="Начало:").grid(row=0, column=0, sticky="w")
tk.Scale(slice_2_frame, from_=0, to=20, orient="horizontal", variable=slice_2_start, command=start_visualization).grid(row=0, column=1, sticky="w")
tk.Label(slice_2_frame, text="Конец:").grid(row=0, column=2, sticky="w")
tk.Scale(slice_2_frame, from_=0, to=20, orient="horizontal", variable=slice_2_end, command=start_visualization).grid(row=0, column=3, sticky="w")
tk.Label(slice_2_frame, text="Шаг:").grid(row=1, column=0, sticky="w")
tk.Scale(slice_2_frame, from_=1, to=10, orient="horizontal", variable=slice_2_step, command=start_visualization).grid(row=1, column=1, sticky="w")

slice_3_frame = tk.LabelFrame(root, text="Срез по оси Z")
slice_3_frame.grid(row=6, column=0, columnspan=4, sticky="w")
tk.Label(slice_3_frame, text="Начало:").grid(row=0, column=0, sticky="w")
tk.Scale(slice_3_frame, from_=0, to=20, orient="horizontal", variable=slice_3_start, command=start_visualization).grid(row=0, column=1, sticky="w")
tk.Label(slice_3_frame, text="Конец:").grid(row=0, column=2, sticky="w")
tk.Scale(slice_3_frame, from_=0, to=20, orient="horizontal", variable=slice_3_end, command=start_visualization).grid(row=0, column=3, sticky="w")
tk.Label(slice_3_frame, text="Шаг:").grid(row=1, column=0, sticky="w")
tk.Scale(slice_3_frame, from_=1, to=10, orient="horizontal", variable=slice_3_step, command=start_visualization).grid(row=1, column=1, sticky="w")

slice_2_frame.grid_remove()
slice_3_frame.grid_remove()

start_visualization()

root.mainloop()

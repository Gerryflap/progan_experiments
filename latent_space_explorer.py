"""
    The latent space explorer loads a generator and allows the user to explore the latent space of this model.

"""

import os
import torch
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
import tkinter.filedialog

image = None
orig_img = None
should_update = True

root = tk.Tk()
filename = tk.filedialog.askopenfilename(initialdir="./saved_models", title="Select generator",
                                           filetypes=(("Pytorch model", "*.pt"), ("all files", "*.*")))
root.destroy()
generator = torch.load(filename, map_location=torch.device('cpu'))
generator.eval()

print(generator.inp_layer)

z_shape = int(generator.inp_layer.weight.data.size(1))
max_phase = generator.n_upscales
phase = max_phase
z = np.zeros((1, z_shape), dtype=np.float32)


def set_should_update():
    global should_update
    should_update = True


def randomize():
    global should_update
    should_update = False
    for slider in sliders:
        slider.set(np.random.normal(0, 1))



def reset():
    global should_update
    should_update = False
    for slider in sliders:
        slider.set(0)


def dankify():
    global should_update
    should_update = False
    for slider in sliders:
        slider.set(4.20)


def load_z():
    global should_update
    should_update = False
    z = np.load("z.npy")
    for i, slider in enumerate(sliders):
        slider.set(z[0, i])


def update_and_enable_updates(iets):
    global should_update
    should_update = True
    update_canvas(iets)


def update_canvas(iets):
    global image
    global orig_img
    global should_update

    if not should_update:
        return

    for i in range(len(sliders)):
        z[0, i] = sliders[i].get()
    # array = G(z).eval(session=K.get_session())[0]
    array = generator(torch.from_numpy(z), phase=phase)[0]

    array = torch.clamp(array, 0, 1)

    array *= 255.0
    array = array.detach().numpy().astype(np.uint8)

    if array.shape[0] == 1:
        array = array[0, :, :]
    else:
        array = np.stack(list(array), axis=2)

    img = Image.fromarray(array)
    orig_img = img
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(image=img)
    image = img
    canvas.create_image(0, 0, anchor="nw", image=image)

def phase_switch(iets):
    global phase
    phase = phase_slider.get()
    update_canvas(iets)


root = tk.Tk()
root.title("GAN tool")
root.attributes('-type', 'dialog')

left_frame = tk.Frame()
canvas = tk.Canvas(left_frame, width=200, height=200)
canvas.pack()

phase_slider = tk.Scale(root, from_=0, to_=max_phase, resolution=0.1, length=290, orient=tk.HORIZONTAL, command=phase_switch)
phase_slider.pack()
phase_slider.set(max_phase)

scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


def scroll_set(*args):
    print("Setting scrollbar ", args)
    scrollbar.set(*args)

subcanvas = tk.Canvas(root, bd=0, highlightthickness=0, yscrollcommand=scroll_set, width=500, height=500)
subframe = tk.Frame(subcanvas)

# for i, slider in enumerate(sliders):
#     subframe.insert(tk.END, slider)
# subframe.pack(side=tk.LEFT, fill=tk.BOTH)
subcanvas.pack(side=tk.RIGHT)
scrollbar.config(command=subcanvas.yview)
subframe_id = subcanvas.create_window((0, 0), window=subframe, anchor='nw')
sliders = [tk.Scale(subframe, from_=-5.0, to_=5.0, resolution=0.05, length=290, orient=tk.HORIZONTAL,
                    command=update_canvas if i != z_shape - 1 else update_and_enable_updates) for i in range(z_shape)]
sliders = list(sliders)
list(map(lambda s: s.pack(), sliders))


def _configure_subframe(event):
    # update the scrollbars to match the size of the inner frame
    size = (subframe.winfo_reqwidth(), subframe.winfo_reqheight())
    subcanvas.config(scrollregion="0 0 %s %s" % size)
    if subframe.winfo_reqwidth() != subcanvas.winfo_width():
        # update the canvas's width to fit the inner frame
        subcanvas.config(width=subframe.winfo_reqwidth())


subframe.bind('<Configure>', _configure_subframe)


def _configure_canvas(event):
    if subframe.winfo_reqwidth() != subcanvas.winfo_width():
        # update the inner frame's width to fill the canvas
        subcanvas.itemconfigure(subframe_id, width=subcanvas.winfo_width())


subcanvas.bind('<Configure>', _configure_canvas)
update_canvas(None)

reset_button = tk.Button(left_frame, text="Reset", command=reset)
reset_button.pack()

save_button = tk.Button(left_frame, text="Save", command=lambda: orig_img.save("saved_image.png"))
save_button.pack()

randomize_button = tk.Button(left_frame, text="Randomize", command=randomize)
randomize_button.pack()

load_z_button = tk.Button(left_frame, text="Load_z", command=load_z)
load_z_button.pack()

left_frame.pack(side=tk.LEFT)

tk.mainloop()

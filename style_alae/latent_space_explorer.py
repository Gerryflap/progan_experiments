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
filenameF = tk.filedialog.askopenfilename(initialdir="./results", title="Select F",
                                           filetypes=(("Pytorch model", "*.pt"), ("all files", "*.*")))
init_dir = os.path.split(filenameF)[0]
filenameG = tk.filedialog.askopenfilename(initialdir=init_dir, title="Select G",
                                           filetypes=(("Pytorch model", "*.pt"), ("all files", "*.*")))
root.destroy()
Fnet = torch.load(filenameF, map_location=torch.device("cuda"))
generator = torch.load(filenameG, map_location=torch.device("cuda"))
Fnet.eval()
generator.eval()

print([param.size() for param in generator.init_layer.parameters()])


z_shape = Fnet.latent_size
max_phase = generator.n_upscales

phase = max_phase
mean_w = None
truncation_psi = 1.0
z = np.zeros((1, z_shape), dtype=np.float32)
current_w = None


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


def load_w():
    global should_update
    global current_w
    should_update = False
    w = torch.from_numpy(np.load("w.npy")).cuda()
    current_w = w
    set_img_from_w()
    should_update = True

def truncate_w(w):
    global mean_w
    global truncation_psi
    if mean_w is None or truncation_psi == 1.0:
        return w
    else:
        wp = mean_w + truncation_psi * (w - mean_w)
        print((w - wp).sum())
        return wp


def update_and_enable_updates(iets):
    global should_update
    should_update = True
    update_canvas(iets)

def set_img_from_w():
    global image
    global orig_img
    global current_w
    w = current_w
    w = truncate_w(w)
    array = generator(w, phase=phase)[0]

    array = torch.clamp(array, 0, 1)

    array *= 255.0
    array = array.detach().cpu().numpy().astype(np.uint8)

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

def update_canvas(iets):
    global image
    global orig_img
    global should_update
    global current_w

    if not should_update:
        return

    for i in range(len(sliders)):
        z[0, i] = sliders[i].get()
    # array = G(z).eval(session=K.get_session())[0]
    w = Fnet(torch.from_numpy(z).cuda())
    current_w = w
    set_img_from_w()

def phase_switch(iets):
    global phase
    phase = phase_slider.get()
    set_img_from_w()

def truncation_switch(iets):
    global truncation_psi
    global mean_w
    if mean_w is None:
        samples = torch.normal(0, 1, (100, z_shape), device="cuda")
        mean_w = Fnet(samples).mean(dim=0, keepdim=True)
    truncation_psi = truncation_slider.get()
    set_img_from_w()


root = tk.Tk()
root.title("GAN tool")
root.attributes('-type', 'dialog')

left_frame = tk.Frame()
canvas = tk.Canvas(left_frame, width=200, height=200)
canvas.pack()

phase_slider = tk.Scale(root, from_=0, to_=max_phase, resolution=0.1, length=290, orient=tk.HORIZONTAL, command=phase_switch)
phase_slider.pack()
phase_slider.set(max_phase)

truncation_slider = tk.Scale(root, from_=-1.0, to_=1.5, resolution=0.1, length=290, orient=tk.HORIZONTAL, command=truncation_switch)
truncation_slider.pack()
truncation_slider.set(1.0)

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

load_z_button = tk.Button(left_frame, text="Load_z", command=load_w)
load_z_button.pack()

left_frame.pack(side=tk.LEFT)

tk.mainloop()

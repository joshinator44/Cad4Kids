import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Cad4Kids")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
canvas_width = int(screen_width * 0.75)
canvas_height = int(screen_height * 0.75)
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack(side=tk.LEFT)

def load_image(button_number):
    file_path = filedialog.askopenfilename(title=f"Select Image {button_number}", filetypes=[("Image files", "*.png")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((100, 100))  # Resize the image to fit the button
        button_img = ImageTk.PhotoImage(image)
        if button_number == 1:
            button1.config(image=button_img)
            button1.image = button_img  # Keep a reference to avoid garbage collection issues
        elif button_number == 2:
            button2.config(image=button_img)
            button2.image = button_img
        elif button_number == 3:
            button3.config(image=button_img)
            button3.image = button_img

def clear_image(button_number):
    if button_number == 1:
        button1.config(image="", text="Image 1")
    elif button_number == 2:
        button2.config(image="", text="Image 2")
    elif button_number == 3:
        button3.config(image="", text="Image 3")

def quit_app():
    root.quit()

button_frame = tk.Frame(root)
button_frame.pack(side=tk.RIGHT)

button1 = tk.Button(button_frame, text="Image 1", command=lambda: load_image(1))
button1.pack(pady=10)

button2 = tk.Button(button_frame, text="Image 2", command=lambda: load_image(2))
button2.pack(pady=10)

button3 = tk.Button(button_frame, text="Image 3", command=lambda: load_image(3))
button3.pack(pady=10)

clear_button1 = tk.Button(button_frame, text="Clear Image 1", command=lambda: clear_image(1))
clear_button1.pack(pady=10)

clear_button2 = tk.Button(button_frame, text="Clear Image 2", command=lambda: clear_image(2))
clear_button2.pack(pady=10)

clear_button3 = tk.Button(button_frame, text="Clear Image 3", command=lambda: clear_image(3))
clear_button3.pack(pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_image)
clear_button.pack(pady=10)

quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(pady=10)

root.mainloop()



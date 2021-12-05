
import tkinter as tk   
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import asksaveasfile, askopenfilename

import io
import os
from PIL import Image, ImageOps, ImageTk

from keras.models import load_model
from keras import backend 
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2   
import random
import math

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

import time #for testing
def resize_to_28x28(img):
    img_h, img_w = img.shape
    dim_size_max = max(img.shape) 
    im_h = 28
    out_img = cv2.resize(img, (28,im_h),0,0,cv2.INTER_NEAREST)  
    return out_img

 
class App(tk.Tk):  
    def __init__(self):
        #model setup
        letters = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.labels_to_letters = list(letters)
        self.model = load_model('models/model_dl_balanced_2conv1dense.h5')
        
    
        #tkinter startup
        self.root = tk.Tk.__init__(self)
        
        self.predicted = StringVar()
        self.predicted.set(" - ") 

        #create drawing canvas
        self.line_start = None
        self.canvas = tk.Canvas(self, width=220, height=220, bg="white",bd=2,
            highlightthickness=1, highlightbackground="black")

        # self.canvas.bind("<Button-1>", lambda e: self.draw(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.draw(e))
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords) 
        self.canvas.old_coords=None
        self.canvas_image = None



        #create buttons and labels 
        self.pred_label = ttk.Label(self,textvariable=self.predicted,font=('Segoe UI',15),borderwidth=2, relief="groove")
        self.pred_button = ttk.Button(self,text="Predict",command=self.predict)
        self.clear_button = ttk.Button(self, text="Clear", command=self.clear)
        self.save_canvas_button = ttk.Button(self, text="Save Canvas", command=self.save_canvas)
        self.save_feature_button = ttk.Button(self, text="Save Features", command=self.save_feature)
        self.help_button = ttk.Button(self, text="Help/Info",command=self.help)
        self.upload_button = ttk.Button(self,text='Upload',command=self.upload)
        self.canvas_label = ttk.Label(text="Draw a letter/digit here!",font=('Segoe UI',11))
        self.prob_label = ttk.Label(text="Probabilities")
        self.graph_label = ttk.Label(text="Feature Map",font=('Segoe UI',15)) 
        
        self.conv1_button =ttk.Button(self,text='Conv1',
            command=lambda num=0:self.replot(num))
        self.mp1_button =ttk.Button(self,text='Max_pooling1',
            command=lambda num=1:self.replot(num))
        self.conv2_button =ttk.Button(self,text='Conv2',
            command=lambda num=2:self.replot(num))
        self.mp2_button =ttk.Button(self,text='Max_pooling2',
            command=lambda num=3:self.replot(num))

        self.about_button = ttk.Button(self, text='About the Model',
            command=self.about)

        #figure  
        self.display_grids = None
        self.img_fig, self.img_ax = plt.subplots(1, figsize=(8,6))
        self.img_fig.subplots_adjust(wspace=0, hspace=0.1)
        self.img_ax.set_xticklabels([])
        self.img_ax.set_yticklabels([]) 
        self.img_ax.set_axis_off()
        # [axi.set_axis_off() for axi in self.img_ax.ravel()]
        self.img_fig.subplots_adjust(bottom=0.05, top=0.95, left=0.18, right=0.82)
        self.layer_names = ['Conv2d','Max_pooling2d','Conv2d_1','Max_pooling2d_1']

        #barplot
        self.prob_fig, self.prob_ax = plt.subplots(1,figsize=(2.2,2.2))
        self.prob_fig.tight_layout() 
        self.prob_ax.set_axis_off()  
        # arrange buttons 
        # grid(row=0, column=0, columnspan=4,padx=self.pad)

        self.help_button.grid(row=0, column=0, columnspan=1)
        self.upload_button.grid(row=0,column=1,columnspan=1)
        self.canvas_label.grid(row=1, column=0, columnspan=2)
        self.canvas.grid(row=2, column=0,columnspan=2, padx=5, pady=10) 

        self.pred_button.grid(row=3, column=0)
        self.clear_button.grid(row=3,column=1)
        self.save_canvas_button.grid(row=4, column=0)
        self.save_feature_button.grid(row=4,column=1)

        #arrange predicted
        self.pred_label.grid(row=5, column=0,columnspan=2, 
            padx=5,ipadx=5,ipady=5 ) 
        self.prob_label.grid(row=6,column=0, columnspan=2)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig , master=self) 
        self.prob_canvas.get_tk_widget().grid(row=7, column=0, columnspan=2  ) 

        #arrange fig 
        self.graph_label.grid(row=0,column=2)
        self.conv1_button.grid(row=0,column=3)
        self.mp1_button.grid(row=0,column=4)
        self.conv2_button.grid(row=0,column=5)
        self.mp2_button.grid(row=0,column=6)
        self.about_button.grid(row=0,column=7)
        self.fig_canvas = FigureCanvasTkAgg(self.img_fig , master=self) 
        self.fig_canvas.get_tk_widget().grid(row=1,column=2,rowspan=7,columnspan=6,padx=8,pady=2, sticky=tk.E)

        #load texts for about page
        f=open('model_visual/texts.txt')
        self.texts = list()
        temp=''
        for l in f.readlines(): 
            if(l=='|\n' or l=="|"):
                self.texts.append(temp)
                temp=''
            else:
                temp+=l 
        f.close()
 
    def draw(self,event):
        w = 25
        x, y = event.x, event.y
        if self.canvas.old_coords:
            x1, y1 = self.canvas.old_coords
            self.canvas.create_line(x, y, x1, y1, smooth=TRUE, capstyle=ROUND, width=w,activewidth=w)
        self.canvas.old_coords = x, y
    

    def place_text(self,title, text, frame, wrapw):
        ab_title = ttk.Label(frame, text=title, font=('Segoe UI',18, 'bold'), justify='left')
        ab_title.pack()
        ab_text = ttk.Label(frame, text=text, wraplength=wrapw,justify='left',anchor='w')
        ab_text.pack(anchor='w')

    def place_image(self,path,frame,iw):
        img=Image.open(path)
        w,h=img.size
        img = img.resize((iw, int(iw*h/w)), Image.ANTIALIAS)  
        img_wrap = ImageTk.PhotoImage(img)
        label_img = ttk.Label(frame, image = img_wrap) 
        label_img.image = img_wrap
        label_img.pack(padx=20)
    def help(self): 
        win = tk.Toplevel()
        win.wm_title("Help")
 
        help_text = ''' 
        Help: Shows the help interface
        Upload: upload an image of letter/digit for prediction
        Predict: Predict the letter/digit drawn in canvas 
        Clear: Clears the canvas and the graphs
        Save Canvas: Save the current canvas as an image
        Save Features: Save the current feature map as an image
        Probabilities graph:  Shows how confident the classifier is with its top 5 predictions
        Conv1/Max_pooling1/Conv2/Max_pooling2: Shows the features that the classifier extracts to help identify what letter/digit is drawn in the canvas 
        ''' 
        l = tk.Label(win, text=help_text,anchor='w',wraplength=500,justify=LEFT)
        l.pack()
        self.place_image('model_visual/help.png', win, 600) 
 
 


    def about(self):
        def myfunction(event):
            canvas.configure(scrollregion=canvas.bbox("all"),width=cw,height=ch)
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        win_w,win_h=750, 550
        cw, ch = win_w-30,win_h-30
        win = tk.Toplevel()
        win.wm_title("About the Model")
        win.geometry(str(win_w)+"x"+str(win_h)) 

        myframe=tk.Frame(win,relief=GROOVE,width=cw,height=ch,bd=1)
        myframe.place(x=10,y=10)

        canvas=tk.Canvas(myframe)
        frame=tk.Frame(canvas)
        myscrollbar=tk.Scrollbar(myframe,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=myscrollbar.set) 
        myscrollbar.pack(side="right",fill="y")
        canvas.pack(side="left")
        canvas.create_window((0,0),window=frame,anchor='nw')
        frame.bind("<Configure>",myfunction) 

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        #load from file
        about_cnn, about_input, about_conv, about_pool, about_activation, about_fc, about_model = self.texts
        
  
        #image and label params
        iw,ih = win_w-30,win_h-30 
        wrapw = cw-50 

        #placing text widgets
        self.place_text("About CNN", about_cnn, frame, wrapw)
        self.place_text("About This Model", about_model, frame, wrapw)

        #about model images
        self.place_image("model_visual/cnn_structure1.png", frame,iw-20)
        self.place_image("model_visual/cnn_structure2.png",frame, iw-20)
        
        #about kernel images, show kernels  
        self.place_image("model_visual/kernel1.png ", frame,iw//3)
        self.place_image("model_visual/kernel2.png",frame, iw//3)
         

        #about input 
        self.place_text("Input Layer", about_input, frame, wrapw)
        #about conv
        self.place_text("Convolutional Layer", about_conv, frame, wrapw)
        #show kernel ex 1
        self.place_image("model_visual/kernel_ex1.gif", frame,(iw-100))
        #about pool, show pool
        self.place_text("Pooling Layer", about_pool,frame, wrapw)
        self.place_image("model_visual/pooling.png", frame,(iw-100))
        #about activation #about fc
        self.place_text("Activation", about_activation,frame, wrapw)
        self.place_text("Fully Connected Layer", about_fc,frame, wrapw)
 
    def upload(self):
        #resize into square -> canvas size (if needed)
        #auto predict
        filename=askopenfilename(filetypes=[('JPG Images','.jpg'),
            ('JPEG Images','.jpeg'),('PNG Images','.png')])
        if filename is None:
            return 
        print(filename)
        try:
            im=Image.open(filename)
            im = im.resize((220,220)) #canvas size
            im_cv = ImageTk.PhotoImage(im) 
            self.canvas_image = im_cv 
            self.canvas.create_image(5,5,image=self.canvas_image, anchor=NW) 
            # self.predict()

        except Exception as e:  
            print(e) 
        

    def clear(self):
        self.canvas.delete('all') 
        self.predicted.set(" - ")

        #clearing canvas  
        self.img_ax.clear()
        self.img_ax.axis('off')
        # [axi.clear() for axi in self.img_ax.ravel()]
        # [axi.axis("off") for axi in self.img_ax.ravel()]

        self.prob_ax.clear()
        self.prob_ax.axis('off') 
        #try .remove()
        # self.img_fig.clear()
        self.fig_canvas.draw() 
        self.prob_canvas.draw()

    def clear_graphs(self):
        # [axi.clear() for axi in self.img_ax.ravel()]
        # [axi.axis("off") for axi in self.img_ax.ravel()] 
        self.img_ax.clear()
        self.img_ax.axis("off")
        self.prob_ax.clear()
        self.prob_ax.axis('off')  
        self.fig_canvas.draw() 
        self.prob_canvas.draw()
    def reset_coords(self,event):
        self.canvas.old_coords = None

    def save_feature(self):
        #check current fig_canvas
        files = [('All Files', '*.*'),  ('PNG images', '*.png') ] 
        filename = asksaveasfile(mode='w', defaultextension = files) 
        if filename is None:
            return   
        self.img_fig.savefig(filename) 
        tk.messagebox.showinfo(message="Plots saved to "+filename)
        filename.close()
    def save_canvas(self): 
        #save current canvas 
        files = [('All Files', '*.*'),  ('JPG images', '*.jpg') ]
        filename = asksaveasfile(mode='w', defaultextension = files) 
        if filename is None:
            return   
        ps = self.canvas.postscript(colormode='gray') 
        img = Image.open(io.BytesIO(ps.encode('utf-8'))) 
        img = ImageOps.grayscale(img) 
        img.save(filename,"jpeg")
        tk.messagebox.showinfo(message="Image saved to "+filename)
        filename.close()
    
    def replot(self,num): 
        if(self.display_grids):
            self.img_ax.set_title(self.layer_names[num],size=5)
            self.img_ax.imshow(self.display_grids[num],aspect='auto',cmap='viridis')
            if(num<=1): #32 and 32 
                self.img_fig.subplots_adjust(bottom=0.05, top=0.95, left=0.18, right=0.82)
            else: #64 and 64
                self.img_fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)

            self.fig_canvas.draw() 

    def predict(self): 
        self.display_grids = [0]*4
        #Get image
        ps = self.canvas.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8'))) 
        img = ImageOps.grayscale(img)

        #convert to np arr and resize
        img = np.array(img) 
        img = resize_to_28x28(img) 
        resize_img = img
 
        
        #NOTE color of input is inverted vs training data 
        #invert color and normalize
        inv_img = 255 - img    
        img = inv_img/255 
        #apply transformation to convert to same format 
        img = np.flip(np.rot90(img,3), axis=1) 
        #reshape into 1,28,28,1 
        input_data = np.copy(img).reshape(1,28,28,1)  
        #convert to tensor
        input_data = tf.convert_to_tensor(input_data, np.float32)

        #clear graphs
        self.clear_graphs()

        #make prediction
        pred = self.model.predict(input_data)  
        pred_idx =np.argmax(pred[0])
        #show top 5 predictions in barplot
        top = np.argsort(pred[0])[::-1][0:5]  
        top_pred = [self.labels_to_letters[j] for j in top]
        top_pred_prob = [pred[0][j] for j in top]
        print(top_pred, top_pred_prob)

        #insert a plot
        self.prob_ax.set_axis_on() 
        self.prob_ax.barh(top_pred, top_pred_prob)
        self.prob_ax.invert_yaxis() 
        self.prob_ax.set_xlim([0,1.05])

        #set predicted output
        self.predicted.set("Predicted: "+self.labels_to_letters[pred_idx])
 
        #set up for layers' output
        layer_outputs = [layer.output for layer in self.model.layers]
        visual_model = tf.keras.models.Model(inputs = self.model.input, outputs = layer_outputs) 
        feature_maps = visual_model.predict(input_data)      
        # Collect the names of each layer except the first one for plotting
        layer_names = [layer.name for layer in self.model.layers]


        j=0
        # Plotting intermediate representation images layer by layer
        num_disp = 7 #randomly selects  count
        for layer_name, feature_map in zip(layer_names, feature_maps):  
            if len(feature_map.shape) == 4: # skip FC layers 
                #Note: 32,32,64,64 ->  4:3 ratio
                # number of features in an individual feature map
                n_features = feature_map.shape[-1]   
                num_disp = n_features #decide display count
                feature_indexes = random.sample(range(n_features), num_disp) 
                # The feature map is in shape of (1, size, size, n_features)
                size = feature_map.shape[1]  

                # Tile our feature images in matrix display_grid   
                # display_grid = np.zeros((size, size * num_disp))
                sq=8
                if(num_disp==32):
                    display_grid = np.zeros((size*8, size*4))
                else: 
                    display_grid = np.zeros((size*sq, size*sq))
                
  
                for i in range(num_disp):
                    rem = i//sq
                    mod = i%sq
                    # Process img
                    x = feature_map[0, :, :, feature_indexes[i]]
                    x = np.rot90(np.flip(x, axis=1))  
                    x -= x.mean()
                    if(x.std() !=0.0): 
                        x /= x.std()   
                    x *= 64
                    x += 128 
                    x = np.clip(x, 0, 255).astype('uint8')  
                    
                    display_grid[mod*size:(mod+1)*size, rem*size : (rem+1) * size] = x
                  
                # self.img_ax[j].set_title(layer_name,size=5)
                # self.img_ax[j].imshow(display_grid, aspect='auto', cmap='viridis')
                self.display_grids[j]=display_grid
                j=j+1
        
        #draw figures (default to first feature map)
        self.img_ax.set_title(self.layer_names[0],size=5)
        self.img_ax.imshow(self.display_grids[0],aspect='auto',cmap='viridis')
        self.fig_canvas.draw() 
        #draw probability graph
        self.prob_canvas.draw()
     


app = App()
app.mainloop()
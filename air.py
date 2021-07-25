
#pyinstaller.exe --onefile --icon=icon.ico air.py
import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
import numpy as np
# importing required libraries
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
import pickle
# import filedialog module
from tkinter import filedialog
import time
import datetime

import tkinter.scrolledtext as st
import tkinter as tk

root = Tk(className='Air Quality test')
root.geometry("1000x500")

csv_bool = False

# call function when we click on entry box
def clear_entry(event, entry):
    entry.delete(0, END)
    #entry.unbind('<Button-1>', click_event)

# call function when we leave entry box
def leave(event, entry, text): #*args
    if entry.get() == "" or entry.get() == text:
        entry.delete(0, 'end')
        entry.insert(0, text)
        root.focus()

# Function for opening the
# file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("all files",
                                                        "*.*"),
                                                        ("Text files",
                                                        "*.txt*")
                                                       ))
      
    # Change label contents
    #test_data = pd.read_csv('Data Sheet - Buvana - Trial 1.csv')
    #df = DataFrame(read_sales,columns=['TGS 826', 'TGS 822', 'TGS 2600', 'TGS 2602', 'Temperature', 'Humidity', 'Class'])  # assign column names
    #print(test_data.head(5))
    csv_bool = True
    print(csv_bool)
    label_file_explorer.configure(text=filename)#"File Opened: "+filename
  
e0 = Entry(root, width=50)
e0.pack()
e0.insert(0, "Enter TGS 826: ")
e0.bind("<Button-1>", lambda event: clear_entry(event, e0))
e0.bind("<Leave>", lambda event: leave(event, e0, 'Enter TGS 826: ')) #leave

e1 = Entry(root, width=50)
e1.pack()
e1.insert(0, "Enter TGS 822: ")
e1.bind("<Button-1>", lambda event: clear_entry(event, e1))
e1.bind("<Leave>", lambda event: leave(event, e1, 'Enter TGS 822: ')) #leave

e2 = Entry(root, width=50)
e2.pack()
e2.insert(0, "Enter TGS 2600: ")
e2.bind("<Button-1>", lambda event: clear_entry(event, e2))
e2.bind("<Leave>", lambda event: leave(event, e2, 'Enter TGS 2600: ')) #leave

e3 = Entry(root, width=50)
e3.pack()
e3.insert(0, "Enter TGS 2602: ")
e3.bind("<Button-1>", lambda event: clear_entry(event, e3))
e3.bind("<Leave>", lambda event: leave(event, e3, 'Enter TGS 2602: ')) #leave

e4 = Entry(root, width=50)
e4.pack()
e4.insert(0, "Enter Temperature: ")
e4.bind("<Button-1>", lambda event: clear_entry(event, e4))
e4.bind("<Leave>", lambda event: leave(event, e4, 'Enter Temperature: ')) #leave

e5 = Entry(root, width=50)
e5.pack()
e5.insert(0, "Enter Humidity: ")
e5.bind("<Button-1>", lambda event: clear_entry(event, e5))
e5.bind("<Leave>", lambda event: leave(event, e5, 'Enter Humidity: ')) #leave

def click():
    hello ="Hello " + e0.get()
    label = Label(root, text=hello)
    label.pack()

class AirData:
  def __init__(self, title, no):
        self.title = title
        self.no = no
        self.trainning = False
        self.csv = False

  def analysis(self,para):
    # read the train and test dataset
    train_data = pd.read_csv('./datasets/Data Sheet - Buvana - Trial 1.csv')
    #test_data = pd.read_csv('test_data.csv')
    print(para)
    if para=="csv":
        self.csv=True
    else:
        self.csv=False
    # shape of the dataset
    print('Shape of training data :',train_data.shape)
    # print('Shape of testing data :',test_data.shape)

    # Now, we need to predict the missing target variable in the test data
    # target variable - Survived

    train_data=train_data.dropna(subset=['TGS 826', 'TGS 822', 'TGS 2600', 'TGS 2602', 'Temperature', 'Humidity', 'Class'])
    print('Shape of training data after dropping :',train_data.shape)
    # seperate the independent and target variable on training data
    train_x = train_data.drop(columns=['Class'],axis=1)
    train_y = train_data['Class']
     
    # seperate the independent and target variable on testing data
    # test_x = test_data.drop(columns=['Class'],axis=1)
    # test_y = test_data['Class']

    train_x[280:]
    # train_data[280:]
    train_y[:281]

    #train_x.dropna(how='all')
    train_x=train_x.fillna(0)
    train_x[280:]
    # print(csv_bool)
    # if csv_bool==True:
    #     self.csv = True

    model = XGBClassifier()
    

    if self.trainning:
        # fit the model with the training data
        model.fit(train_x,train_y)
        # save the model to disk
        filename = 'finalized_model.sav' #.sav, .pkl
        pickle.dump(model, open(filename, 'wb'))
    else:
        # some time later...
        # load the model from disk
        filename = './models/finalized_model.sav'
        model = pickle.load(open(filename, 'rb')) #loaded_model
        # result = loaded_model.score(X_test, Y_test)
        # print(result)
     
    # predict the target on the train dataset
    predict_train = model.predict(train_x)
    print('\nTarget on train data',predict_train) 
     
    # Accuray Score on train dataset
    accuracy_train = accuracy_score(train_y,predict_train)
    print('\naccuracy_score on train dataset : ', accuracy_train)
    #print(self.csv)
    # # predict the target on the test dataset
    if self.csv:
        print("Predictions through CSV file")
        print(label_file_explorer.cget("text"))
        test_data = pd.read_csv(label_file_explorer.cget("text"))
        test_data=test_data.dropna(subset=['TGS 826', 'TGS 822', 'TGS 2600', 'TGS 2602', 'Temperature', 'Humidity', 'Class'])
        print('Shape of testing data after dropping :',train_data.shape)
        # seperate the independent and target variable on training data
        test_data_x = test_data.drop(columns=['Class'],axis=1)
        test_data_y = test_data['Class']
        predict_test = model.predict(test_data_x)
        #print('\nTarget on test data',predict_test)
        output= " \n " + "[Logs]: " + 'Target on test data: '+ predict_test + " \n "
        #print(type(predict_test))
        # convert n_array into dataframe
        DF = pd.DataFrame(predict_test)
        dtime=str(datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")).replace(" ", "_")
        filename=f"./logs/testing_data_logs_{dtime}.csv"#str()#.\logs\
        #print(filename)
        #print(type(filename))
        DF.to_csv(str(filename))
        # label = Label(root, text=output)
        # label.pack()
        # output+="\n [Logs]: "
        output+="[Logs]: "
        textWid.insert(END, output)
        #INPUT = textWid.get("1.0", "end-1c")
        #text_area.insert(tk.INSERT,output)
    else:
        print("Predictions through input entries")
        # test_x=np.array([[0,0.04,0.19,0.2,0.17,37,40]])
        test_x=np.array([[0,float(e0.get()),float(e1.get()),float(e2.get()),float(e3.get()),float(e4.get()),float(e5.get())]])
        df_test = pd.DataFrame(data=test_x, index=[0], columns=["ID", "TGS 826", "TGS 822", "TGS 2600", "TGS 2602", "Temperature", "Humidity"])
        print(df_test)
        predict_test = model.predict(df_test)
        print('\nTarget on test data',predict_test)
        output='Target on test data: '+predict_test
        # label = Label(root, text=output)
        # label.pack()
        output+="\n [Logs]: "
        textWid.insert(END, output)
        #text_area.insert(tk.INSERT,output)
     
    # Accuracy Score on test dataset
    # accuracy_test = accuracy_score(test_y,predict_test)
    # print('\naccuracy_score on test dataset : ', accuracy_test)

p1 = AirData("quality", 0)

# print(p1.name)
# print(p1.age)


button = Button(root, text="Enter", command=lambda: p1.analysis("entries"))
button.pack()

seperator = Label(root,
                            text = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                            width = 100, height = 4,
                            fg = "black")
seperator.pack()

button_explore = Button(root,
                        text = "Browse Files",
                        command = browseFiles)
button_explore.pack()

# Create a File Explorer label
label_file_explorer = Label(root,
                            text = "File Name",
                            width = 100, height = 2,
                            fg = "blue")
label_file_explorer.pack()

button2 = Button(root, text="Upload File", command=lambda: p1.analysis("csv"))
button2.pack()

seperator2 = Label(root,
                            text = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                            width = 100, height = 2,
                            fg = "black")
seperator2.pack()

button_exit = Button(root,
                     text = "Exit",
                     command = exit)
button_exit.pack()

textWid = Text(root, height = 5,
                width = 120,
                bg = "light yellow")
textWid.pack()

# Creating scrolled text area
# widget with Read only by
# disabling the state
text_area = st.ScrolledText(root,
                            width = 30, 
                            height = 8, 
                            font = ("Times New Roman",
                                    15))
text_area.pack()
# text_area.grid(column = 0, pady = 10, padx = 10)
  
# Inserting Text which is read only
text_area.insert(tk.INSERT,#import tkinter as tk->tk.INSERT, root.INSERT
"""\
This is a scrolledtext widget to make tkinter text read only.
Hi
""")
#adjustable text box tkinter


# Placing cursor in the text area
text_area.focus()


# Making the text read only
# text_area.configure(state ='disabled')

root.mainloop()
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 06:34:15 2021

@author: olrez
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
from scipy.constants import convert_temperature
import psychrolib as ps
import math as m
from threading import Thread

root = tk.Tk() 
root.title("Get DBT Tool")  # to define the title
root.geometry("300x200")


#myFont = tkfont.Font(family='Arial',size=10)

topFrame = tk.Frame(root)

# Tell the Frame to fill the whole window
topFrame.pack(fill=tk.BOTH, expand=1)

# Make the Frame grid contents expand & contract with the window
topFrame.columnconfigure(0, weight=1)
for i in range(4):
    topFrame.rowconfigure(i, weight=1)
    
def open_file():
    global measured
    filename = fd.askopenfilename()
    measured = pd.read_excel(filename,sheet_name = None)
    return measured

def export_folder():
    global folder
    folder= fd.askdirectory()
    return folder
    
def close_root_window():
    root.destroy()
    
def run_dbt():
    
    ps.SetUnitSystem(ps.SI) #SI or IP

    Min_DBT=255#273.15 was the original
    # Maximum dry bulb temperature
    Max_DBT=473.15
    # Convergence tolerance
    TOL=0.0005
    
    def __Pws(DBT):
        if __valid_DBT(DBT):
            C8=-5.8002206*10**3
            C9=1.3914993
            C10=-4.8640239*10**-2
            C11=4.1764768*10**-5
            C12=-1.4452093*10**-8
            C13=6.5459673
            return m.exp(C8/DBT+C9+C10*DBT+C11*DBT**2+C12*DBT**3+C13*m.log(DBT))
    
    def __is_positive(x):
        if x>0:
            return True
        else:
            return False
        
    # ASHRAE 2009 Chapter 1 Equation 22 and Equation 24
    def __W_DBT_RH_P(DBT, RH, P):
        if __valid_DBT(DBT):
            Pw=RH*__Pws(DBT)
            return 0.621945*Pw/(P-Pw)
    
    # ASHRAE 2009 Chapter 1 Equation 35
    def __W_DBT_WBT_P(DBT, WBT, P):
        if __valid_DBT(DBT):
            DBT=DBT-273.15
            WBT=WBT-273.15
            return ((2501-2.326*WBT)*__W_DBT_RH_P(WBT+273.15,1,P)-1.006*(DBT-WBT))/\
                   (2501+1.86*DBT-4.186*WBT)
    def __valid_DBT(DBT):
        if Min_DBT<=DBT<=Max_DBT:
            return True
        else:
            return False
        
    def __DBT_RH_WBT_P(RH, WBT, P):
        [DBTa, DBTb]=[Min_DBT, Max_DBT]
        DBT=(DBTa+DBTb)/2
        while DBTb-DBTa>TOL:
            ya=__W_DBT_WBT_P(DBTa, WBT, P)-__W_DBT_RH_P(DBTa, RH, P)
            y=__W_DBT_WBT_P(DBT, WBT, P)-__W_DBT_RH_P(DBT, RH, P)
            if __is_positive(y)==__is_positive(ya):
                DBTa=DBT
            else:
                DBTb=DBT
            DBT=(DBTa+DBTb)/2
        return DBT
    
    for sheet in measured.keys():
        
        P_sea = 101.325*1000 #Pa 14.69594878#Psi
        wbt_f = measured[sheet]['Web Bulb Temp (Deg F)'] #°F
        wbt_c = convert_temperature(wbt_f,'Fahrenheit','Celsius') #°C
        wbt_k = wbt_c + 273.15 #°K
        rh = measured[sheet]['Relative humidity (0-1)'] #from 0 to 1 percent
        dbt_c = []
        
        for i in range(len(wbt_k)):
            DBT=__DBT_RH_WBT_P(rh[i], wbt_k[i], P_sea)
            dbt_c.append(DBT - 273.15) #°C
        dbt_f = convert_temperature(dbt_c,'Celsius','Fahrenheit') #°F
        dbt_series = pd.Series(dbt_f, name = 'Dry Bulb Temp (Deg F)')
        #savetxt(folder+'/'+sheet+'-Dry Bulb data.csv', dbt_f, delimiter=',')
        dbt_series.to_csv(folder+'/'+sheet+'-Dry Bulb data.csv')
    return
    
def show_and_run(func, btn):
    # Save current button color and change it to green
    oldcolor = btn['bg']
    btn['bg'] = 'green'

    # Call the function
    func()

    # Restore original button color
    btn['bg'] = oldcolor

def run_function(func, btn):
    # Disable all buttons
    for b in buttons.values():
        b['state'] = 'disabled'

    processing_bar.start(interval=10)
    show_and_run(func, btn)
    processing_bar.stop()

    # Enable all buttons
    for b in buttons.values():
        b['state'] = 'normal'

def clicked(func, btn):
    Thread(target=run_function, args=(func, btn)).start()


button_data = (
    ('Open raw data', open_file),
    ('Select export data path', export_folder),
    ('Run', run_dbt),
    ('Close', close_root_window),
)

# Make all the buttons and save them in a dict
buttons = {}
for row, (name, func) in enumerate(button_data):
    btn = tk.Button(topFrame, text=name)
    btn.config(command=lambda f=func, b=btn: clicked(f, b))
    btn.grid(row=row, column=0, columnspan=1, sticky='EWNS')
    buttons[name] = btn
row += 1

processing_bar = ttk.Progressbar(topFrame, 
    orient='horizontal', mode='indeterminate')
processing_bar.grid(row=row, column=0, columnspan=1, sticky='EWNS')
    
root.mainloop()
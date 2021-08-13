# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 06:34:15 2021
Last modified on Mon Feb 15 14:47:00 2021

@author: olrez, Eng. Oldemar Ramirez
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog as fd
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
from math import sqrt
from itertools import groupby
from scipy.optimize import curve_fit
from math import ceil
from scipy.optimize import least_squares
from scipy.constants import convert_temperature
import psychrolib as ps
import math as m

root = tk.Tk() 
root.title("Model Fitting Tool")  # to define the title
root.geometry("300x200")


myFont = tkfont.Font(family='Arial',size=10)

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
    
def close_window():
    secondary.destroy()
    root.deiconify()
    
def close_root_window():
    root.destroy()
    
def run_fitting():
    if eq_name == 'p':
        for pump in measured.keys():
        
            hd = measured[pump]['Design Head [ft]'][0] #design head ft
            vd = measured[pump]['Design Flow [gpm]'][0] #design volume/flow gpm
            #pump_name = measured[pump]['Pump Name'][0]
            
            #head vs flow equation
            def head(X, c4, c5, c6):
                c,v = X
                return c4*hd*np.power(c,2)*(1-np.power(v/(c5*c*vd),c6))
            
            # measured data
            c = measured[pump]['Command Speed']
            v = measured[pump]['Flow [gpm]']
            h = measured[pump]['Head [ft]']
            
            #curve fit
            p0 = [1., 1., 1.] # initial guesses for coefficients
            coef_456, pcov = curve_fit(head, (c,v), h, p0)
            hpred = head((c,v),coef_456[0],coef_456[1],coef_456[2])
            
            #error
            hresid = h - hpred
            SSresid_h = sum(pow(hresid,2))
            SStotal_h = len(h)*np.var(h)
            Rsq_h = 1-SSresid_h/SStotal_h #se recomienda mayor a .85
            rmse_h = sqrt(mean_squared_error(h, hpred))
            mean_h=np.mean(h)
            cvrmse_h = rmse_h/mean_h*100 #menor a 6% infica que el modelo es confiable
            
            #power vs flow equation
            def eff(Y, c1, c2, c3):
                c,v = Y
                return c1+c2*v/(coef_456[1]*c*vd)+c3*np.power(v/(coef_456[1]*c*vd),2)
            
            # measured data
            n = measured[pump]['Efficiency']
            
            #curve fit
            p0_2 = [0.5, 1., -1.] # initial guesses for coefficients
            coef_123, pcov2 = curve_fit(eff, (c,v), n, p0_2)
            npred = eff((c,v),coef_123[0],coef_123[1],coef_123[2])
            def power(Z):
                v,hpred,npred = Z
                return v*hpred/(3956*npred)
            p = v*h/(3965*n)
            ppred = power((v,hpred,npred))
            
            #error
            presid = p - ppred
            SSresid_p = sum(pow(presid,2))
            SStotal_p = len(p)*np.var(p)
            Rsq_p = 1-SSresid_p/SStotal_p
            rmse_p = sqrt(mean_squared_error(p, ppred))
            mean_p=np.mean(p)
            cvrmse_p = rmse_p/mean_p*100
            
            #% Plot Section
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,10))
            f.suptitle(pump+' Model Fit: $c_1$ = '+str(round(coef_123[0],6))+'; $c_2$ = '+str(round(coef_123[1],6))+'; $c_3$ = '+str(round(coef_123[2],6))+'; $c_4$ = '+str(round(coef_456[0],6))+'; $c_5$ = '+str(round(coef_456[1],6))+'; $c_6$ = '+str(round(coef_456[2],6)))
            cmd_speeds = [x[0] for x in groupby(c)] #revome duplicates
            #cont = [len(list(group)) for key, group in groupby(c)] #count of curves points
            split_v = np.array_split(v, len(cmd_speeds)) #curves must have the same number of points, or the first can have one point in addition
            split_c = np.array_split(c, len(cmd_speeds))
            legend1 = []
            legend2 = []
            
            #plots
            for i in range(len(cmd_speeds)):
                colors = np.random.rand(3,)
                head_i = head((split_c[i],split_v[i]),coef_456[0],coef_456[1],coef_456[2])
                eff_i = eff((split_c[i],split_v[i]),coef_123[0],coef_123[1],coef_123[2])
                power_i = power((split_v[i],head_i,eff_i))
                ax1.plot(split_v[i],head_i,'--',color=colors)
                legend1.append('Predicted Head ('+str(round(cmd_speeds[i]*100,2))+'% Speed)')
                ax2.plot(split_v[i],power_i,'--',color=colors)
                legend2.append('Predicted Power ('+str(round(cmd_speeds[i]*100,2))+'% Speed)')
                
            legend1.append('Actual Head')
            ax1.scatter(v, h, color='black', marker='o')
            ax1.grid(which='major', linestyle=':', linewidth='0.5', color='black')
            ax1.set_title('Pump Head Model Fit \n CV(RMSE) = '+str(round(cvrmse_h,4))+'%; $R^2$ = '+str(round(Rsq_h,4))) 
            ax1.legend(legend1, loc='upper right', numpoints=1)
            ax1.set_xlabel("Capacity (gpm)")
            ax1.set_ylabel("Head (ft)")
            
            legend2.append('Actual Power')
            ax2.scatter(v, p, color='black', marker='o')
            ax2.grid(which='major', linestyle=':', linewidth='0.5', color='black')
            ax2.set_title('Pump Power Model Fit \n CV(RMSE) = '+str(round(cvrmse_p,4))+'%; $R^2$ = '+str(round(Rsq_p,4))) 
            ax2.legend(legend2, loc='upper left', numpoints=1)
            ax2.set_xlabel("Capacity (gpm)")
            ax2.set_ylabel("Power (bhp)")
            
            #f.show()
            f.savefig(folder+'/'+pump+" curve.png", bbox_inches='tight') #figure export
            
            
            #% Export Section
            coefficients = np.concatenate([coef_123,coef_456])
            index = []
            
            for i in range(6):
                index.append('Pump Model Coefficient '+str(i+1)+'(NaU)') #Planning Tool template format
            model_coefficients = pd.DataFrame({'Values':coefficients},index=index)
            model_coefficients = model_coefficients.rename_axis('Coefficients')
            model_coefficients = model_coefficients.T
            model_coefficients.to_csv(folder+'/'+pump+'_coefficients.csv', index = False)
            
    if eq_name == 'lc':
        for load in measured.keys():
            
            measured[load] = measured[load][measured[load]['Status On/Off'] != 0] #ignore off plant values
            measured[load] = measured[load].reset_index()
            
            #flow vs load equation
            def flow(c,X,v):
                Q,T = X
                return v - np.exp(c[0]+c[1]*np.power(np.log(Q),2)+c[2]*np.exp(c[3]*T))
            
            # measured data
            Q = measured[load]['CHW Load [ton]']*3.5168528421 #from ton to kW
            T = convert_temperature(measured[load]['CHWST [F]'],'Fahrenheit','Celsius') #°C
            V_gpm = measured[load]['CHW Flow [gpm]']
            V = V_gpm*0.000063 #gpm converted to m3/s
            
            max_flow = max(V_gpm)
            def roundup(x):
                return int(ceil(x / 1000.0)) * 1000
            
            axes_limit = roundup(max_flow)
            
            #curve fit
            c0 = [0.01, 0.01, 0.01, 0.01] # initial guesses for coefficients
            
            res_lsq = least_squares(flow, c0, args=((Q,T), V))
            c = res_lsq.x
        
            Vpred = np.exp(c[0]+c[1]*np.power(np.log(Q),2)+c[2]*np.exp(c[3]*T))/0.000063 #gpm
            
            #error
            Vresid = V_gpm - Vpred
            SSresid = sum(pow(Vresid,2))
            SStotal = len(V_gpm)*np.var(V_gpm)
            Rsq = 1-SSresid/SStotal
            rmse = sqrt(mean_squared_error(V_gpm, Vpred))
            mean=np.mean(V_gpm)
            cvrmse = rmse/mean*100
            
            #plots
            plt.figure(figsize=(10,10))
            plt.scatter(V_gpm,Vpred,facecolors='none', edgecolors=[0.1,0.5,1]) 
            plt.plot(V_gpm, V_gpm,color='r') #mean function, linear y=x
            plt.title(load+' Model Fit: $c_1$ = '+str(round(c[0],6))+'; $c_2$ = '+str(round(c[1],6))+'; $c_3$ = '+str(round(c[2],6))+'; $c_4$ = '+str(round(c[3],6))+'\n CV(RMSE) = '+str(round(cvrmse,4))+'%; $R^2$ = '+str(round(Rsq,4))) 
            plt.legend(['Mean Function Regression Fit','Fitted data'], loc='upper left', numpoints=1)
            plt.xlabel('Measured Flow (gpm)')
            plt.ylabel('Fitted Flow (gpm)')
            plt.gca().set_xlim([0,axes_limit])
            plt.gca().set_ylim([0,axes_limit])
            plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
            plt.savefig(folder+'/'+load+"_plot.png", bbox_inches='tight') #figure export
            #plt.show()
            
            #export
            index = ['Coil C1(NaU)','Coil C2(NaU)','Coil C3(NaU)','Coil C4(NaU)']
            model_coefficients = pd.DataFrame({'Values':c},index=index)
            model_coefficients = model_coefficients.rename_axis('Coefficients')
            model_coefficients = model_coefficients.T
            model_coefficients.to_csv(folder+'/'+load+'_coefficients.csv', index = False)
            
    if eq_name == 'ch':
        for chiller in measured.keys():
    
            measured[chiller] = measured[chiller][measured[chiller]['Status'] != 0] #ignore off values
            measured[chiller] = measured[chiller].reset_index()
            qevap = measured[chiller]['CHW Power [ton]']*3.5168528421 #from ton to kW
            p = measured[chiller]['Electric Power [kW]'] #kW
            tchwo = convert_temperature(measured[chiller]['ELFT [F]'],'Fahrenheit','Kelvin') #°K
            tcwo = convert_temperature(measured[chiller]['CLFT [F]'],'Fahrenheit','Kelvin') #°K
            
            y = (p/qevap+1)*tchwo/tcwo-1
            x1 = tchwo/qevap
            x2 = (tcwo-tchwo)/(tcwo*qevap)
            x3 = (p/qevap+1)*qevap/tcwo
            
            
            #curve fit method 
            #linear function
            def func(X, c1, c2, c3):
                x1, x2, x3 = X
                return c1*x1 + c2*x2 + c3*x3
            
            coef_0 = [0.1, 1.,0.1]
            coef, pcov = curve_fit(func, (x1,x2,x3), y, coef_0, maxfev=5000)
            
            c1 = coef[0]
            c2 = coef[1]
            c3 = coef[2]
            p_pred = qevap*(tcwo*(1+c1*tchwo/qevap+c2*(tcwo-tchwo)/(tcwo*qevap))+c3*qevap-tchwo)/(tchwo-c3*qevap) # kW
            
            #error calculation
            p_resid = p - p_pred
            SSresid = sum(pow(p_resid,2))
            SStotal = len(p)*np.var(p)
            Rsq = 1-SSresid/SStotal #coefficient of determination
            rmse = sqrt(mean_squared_error(p, p_pred)) #error rms
            mean=np.mean(p)
            cvrmse = rmse/mean*100 #Coefficient of Variation of Root-Mean Squared Error
            
            #plots
            plt.figure(figsize=(10,10))
            plt.scatter(p,p_pred,facecolors='none', edgecolors=[0.1,0.5,1])
            plt.plot(p, p,color='r') #Linear trendline
            plt.title(chiller+' Model Fit: $GnC_1$ = '+str(round(c1,6))+'; $GnC_2$ = '+str(round(c2,6))+'; $GnC_3$ = '+str(round(c3,6))+'\n CV(RMSE) = '+str(round(cvrmse,4))+'%; $R^2$ = '+str(round(Rsq,4))) 
            plt.legend(['Mean Function Regression Fit','Fitted data'], loc='upper left', numpoints=1)
            plt.xlabel('Measured Electric Power (kW)')
            plt.ylabel('Fitted Electric Power (kW)')
            plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
            plt.savefig(folder+'/'+chiller+"_plot.png", bbox_inches='tight') #image export
            #plt.show()
            
            
            #% Data Export
            index = ['Gn C1(kW/degK)','Gn C2(kW)','Gn C3(degK/KW)']
            model_coefficients = pd.DataFrame({'Values':coef},index=index)
            model_coefficients = model_coefficients.rename_axis('Coefficients')
            model_coefficients = model_coefficients.T
            model_coefficients.to_csv(folder+'/'+chiller+'_coefficients.csv', index = False)
    if eq_name == 'ct':
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
        
        for tower in measured.keys():
            
            P_sea = 101.325*1000 #Pa 14.69594878#Psi
            wbt_f = measured[tower]['Web Bulb Temp (Deg F)'] #°F
            wbt_c = convert_temperature(wbt_f,'Fahrenheit','Celsius') #°C
            wbt_k = wbt_c + 273.15 #°K
            cwrt_f = measured[tower]['Hot/Ent Water Temp (Deg F)'] #°F
            cwrt_c = convert_temperature(cwrt_f,'Fahrenheit','Celsius') #°C
            cwst_f = measured[tower]['Cold/Lvg Water Temp (Deg F)'] #°F
            tower_approach = cwst_f - wbt_f #°F
            water_flow = measured[tower]['Flow Rate (GPM)'] #gpm
            air_flow = measured[tower]['Air Flow Rate (CFM)']*0.00047194745 #cfm converted to m3/s
            
            Q_actual = 3.5168528421*500*water_flow*(cwrt_f-cwst_f)/12000 #condenser water capacity converted from ton to kW
            air_density = 1.225# kg/m³
            m_air = air_density*air_flow #kg/s
            rh = measured[tower]['Relative humidity (0-1)'] #from 0 to 1 percent
            altitude = measured[tower]['Altitude (m)'][0] #m
            P, dbt_c, hr, h_air, h_air_sat = ([] for i in range(5))
            
            for i in range(len(Q_actual)):
                DBT=__DBT_RH_WBT_P(rh[i], wbt_k[i], P_sea)
                dbt_c.append(DBT - 273.15) #°C
                P.append(ps.GetStationPressure(P_sea, altitude, dbt_c[i])) #Pa
                hr.append(ps.GetHumRatioFromTWetBulb(dbt_c[i], wbt_c[i], P[i])) #kg_water/kg_air
                h_air.append(ps.GetMoistAirEnthalpy(dbt_c[i], hr[i])) #J/kg
                h_air_sat.append(ps.GetSatAirEnthalpy(cwrt_c[i], P[i])) #J/kg
            C_air = m_air*(pd.Series(h_air_sat)-pd.Series(h_air))/1000 #air (cold fluid) heat capacity in kJ/s or kW
            C_water = 3.5168528421*500*water_flow*(cwrt_f-wbt_f)/12000 #water (hot fluid) heat capacity converted from ton to kW
            
            C_min = []
            C_r = []
            for i in range(len(C_air)):
                C_min.append(min(C_air[i],C_water[i]))
                C_r.append(min(0.999,C_min[i]/max(C_air[i],C_water[i])))
            C_r = pd.Series(C_r) #heat capacity ratio from 0 to 1
            C_min = pd.Series(C_min) #min heat capacity kW
            #ntu_actual = -np.log((Q_actual/C_min-1)/(Q_actual*C_r/C_min-1))/(1-C_r) #number of transfer units (for reference only)
            #epsilon_actual = Q_actual/C_min #eff from 0 to 1 (for reference only)
            
            #condenser water equiation (epsilon*C_min)
            def power(x, c1, c2): #max(1,C_min) = C_min
                C_min,C_air,C_r = x
                return C_min*((1-np.exp(-(c1*np.power(C_air,c2)/C_min)*(1-C_r)))/(1-C_r*np.exp(-(c1*np.power(C_air,c2)/C_min)*(1-C_r))))
            
            #% curve fit
            coef_0 = [0.1, 1.]
            coef, pcov = curve_fit(power, (C_min,C_air,C_r), Q_actual, coef_0, maxfev=5000)
            Q_pred = power((C_min,C_air,C_r),coef[0],coef[1])
            #ntu_pred = coef[0]*np.power(C_air,coef[1])/C_min #number of transfer units (for reference only)
            #epsilon_pred = Q_pred/C_min #eff from 0 to 1 (for reference only)
            cwst_pred = cwrt_f - Q_pred*12000/(3.5168528421*500*water_flow) #°F
            tower_approach_pred = cwst_pred - wbt_f  #°F
            
            #error calculation
            Qresid = Q_actual - Q_pred
            SSresid = sum(pow(Qresid,2))
            SStotal = len(Q_actual)*np.var(Q_actual)
            Rsq = 1-SSresid/SStotal #coefficient of determination
            rmse = sqrt(mean_squared_error(Q_actual, Q_pred)) #error rms
            mean=np.mean(Q_actual)
            cvrmse = rmse/mean*100 #Coefficient of Variation of Root-Mean Squared Error
            
            plt.figure(figsize=(10,10))
            plt.scatter(tower_approach,tower_approach_pred,facecolors='none', edgecolors=[0.1,0.5,1])
            m_a, b_a = np.polyfit(tower_approach, tower_approach_pred, 1) #Linear trendline
            plt.plot(tower_approach, m_a*tower_approach + b_a,color='r') #Linear trendline
            plt.xlabel('Tower Approach (Manufacturer)')
            plt.ylabel('Tower Approach (CPO NTU-eff)')
            plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
            plt.show()
            
            #plots
            plt.figure(figsize=(10,10))
            plt.scatter(Q_actual,Q_pred,facecolors='none', edgecolors=[0.1,0.5,1])
            plt.plot(Q_actual, Q_actual,color='r') #Linear trendline
            plt.title(tower+' Model Fit: $c_1$ = '+str(round(coef[0],6))+'; $c_2$ = '+str(round(coef[1],6))+'\n CV(RMSE) = '+str(round(cvrmse,4))+'%; $R^2$ = '+str(round(Rsq,4))) 
            plt.legend(['Mean Function Regression Fit','Fitted data'], loc='upper left', numpoints=1)
            plt.xlabel('Measured Condenser Power (kW)')
            plt.ylabel('Fitted Condenser Power (kW)')
            plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
            plt.savefig(folder+'/'+tower+"_plot.png", bbox_inches='tight') #image export
            #plt.show()
            
            #% Data Export
            index = ['c1','c2']
            model_coefficients = pd.DataFrame({'Values':coef},index=index)
            model_coefficients = model_coefficients.rename_axis('Coefficients')
            model_coefficients.to_csv(folder+'/'+tower+'_coefficients.csv')
        
    
def open_pump_window():
    global eq_name
    eq_name = 'p'
    global secondary
    root.withdraw()
    secondary = tk.Toplevel(root)
    secondary.title('Pump Model Fit')
    secondary.geometry("300x100")
    
    topFrameS = tk.Frame(secondary)
    
    # Tell the Frame to fill the whole window
    topFrameS.pack(fill=tk.BOTH, expand=1)
    
    # Make the Frame grid contents expand & contract with the window
    topFrameS.columnconfigure(0, weight=1)
    for i in range(4):
        topFrameS.rowconfigure(i, weight=1)
    
    #new_canvas = tk.Canvas(secondary, width=100, height=100)  # define the size
    #new_canvas.pack()
    
    open_btn = tk.Button(topFrameS, text = 'Open raw data', command = open_file, activebackground = 'white')
    open_btn['font'] = myFont
    open_btn.grid(row=0, column=0, columnspan=1, sticky='EWNS')
    #open_btn.pack(side='left')
    
    exp_btn = tk.Button(topFrameS, text = 'Select export data path', command = export_folder, activebackground = 'white')
    exp_btn['font'] = myFont
    exp_btn.grid(row=1, column=0, columnspan=1, sticky='EWNS')
    #exp_btn.pack(side='left')
    
    run_btn = tk.Button(topFrameS, text = 'Run', command = run_fitting, activebackground = 'white')
    run_btn['font'] = myFont
    #run_btn['status'] = 'disabled'
    run_btn.grid(row=2, column=0, columnspan=1, sticky='EWNS')
    #run_btn.pack(side='left')
    
    close_btn = tk.Button(topFrameS, text = 'Close', command = close_window, activebackground = 'white')
    close_btn['font'] = myFont
    close_btn.grid(row=3, column=0, columnspan=1, sticky='EWNS')
    #close_btn.pack(side='bottom')
    
    secondary.protocol("WM_DELETE_WINDOW", close_window)
    

def open_lc_window():
    global eq_name
    eq_name = 'lc'
    global secondary
    root.withdraw()
    secondary = tk.Toplevel(root)
    secondary.title('Load Coil Model Fit')
    secondary.geometry("300x100")

    topFrameS = tk.Frame(secondary)
    
    # Tell the Frame to fill the whole window
    topFrameS.pack(fill=tk.BOTH, expand=1)
    
    # Make the Frame grid contents expand & contract with the window
    topFrameS.columnconfigure(0, weight=1)
    for i in range(4):
        topFrameS.rowconfigure(i, weight=1)
        
    open_btn = tk.Button(topFrameS, text = 'Open raw data', command = open_file, activebackground = 'white')
    open_btn['font'] = myFont
    open_btn.grid(row=0, column=0, columnspan=1, sticky='EWNS')
    #open_btn.pack(side='left')
    
    exp_btn = tk.Button(topFrameS, text = 'Select export data path', command = export_folder, activebackground = 'white')
    exp_btn['font'] = myFont
    exp_btn.grid(row=1, column=0, columnspan=1, sticky='EWNS')
    #exp_btn.pack(side='left')
    
    run_btn = tk.Button(topFrameS, text = 'Run', command = run_fitting, activebackground = 'white')
    run_btn['font'] = myFont
    run_btn.grid(row=2, column=0, columnspan=1, sticky='EWNS')
    #run_btn.pack(side='left')
    
    close_btn = tk.Button(topFrameS, text = 'Close', command = close_window, activebackground = 'white')
    close_btn['font'] = myFont
    close_btn.grid(row=3, column=0, columnspan=1, sticky='EWNS')
    #close_btn.pack(side='bottom')
    
    secondary.protocol("WM_DELETE_WINDOW", close_window)
    

def open_ch_window():
    global eq_name
    eq_name = 'ch'
    global secondary
    root.withdraw()
    secondary = tk.Toplevel(root)
    secondary.title('Chiller Model Fit')
    secondary.geometry("300x100")

    topFrameS = tk.Frame(secondary)
    
    # Tell the Frame to fill the whole window
    topFrameS.pack(fill=tk.BOTH, expand=1)
    
    # Make the Frame grid contents expand & contract with the window
    topFrameS.columnconfigure(0, weight=1)
    for i in range(4):
        topFrameS.rowconfigure(i, weight=1)
        
    open_btn = tk.Button(topFrameS, text = 'Open raw data', command = open_file, activebackground = 'white')
    open_btn['font'] = myFont
    open_btn.grid(row=0, column=0, columnspan=1, sticky='EWNS')
    #open_btn.pack(side='left')
    
    exp_btn = tk.Button(topFrameS, text = 'Select export data path', command = export_folder, activebackground = 'white')
    exp_btn['font'] = myFont
    exp_btn.grid(row=1, column=0, columnspan=1, sticky='EWNS')
    #exp_btn.pack(side='left')
    
    run_btn = tk.Button(topFrameS, text = 'Run', command = run_fitting, activebackground = 'white')
    run_btn['font'] = myFont
    run_btn.grid(row=2, column=0, columnspan=1, sticky='EWNS')
    #run_btn.pack(side='left')
    
    close_btn = tk.Button(topFrameS, text = 'Close', command = close_window, activebackground = 'white')
    close_btn['font'] = myFont
    close_btn.grid(row=3, column=0, columnspan=1, sticky='EWNS')
    #close_btn.pack(side='bottom')
    
    secondary.protocol("WM_DELETE_WINDOW", close_window)
    

def open_ct_window():
    global eq_name
    eq_name = 'ct'
    global secondary
    root.withdraw()
    secondary = tk.Toplevel(root)
    secondary.title('Cooling Tower Model Fit')
    secondary.geometry("300x100")
    
    topFrameS = tk.Frame(secondary)
    
    # Tell the Frame to fill the whole window
    topFrameS.pack(fill=tk.BOTH, expand=1)
    
    # Make the Frame grid contents expand & contract with the window
    topFrameS.columnconfigure(0, weight=1)
    for i in range(4):
        topFrameS.rowconfigure(i, weight=1)
        
    open_btn = tk.Button(topFrameS, text = 'Open raw data', command = open_file, activebackground = 'white')
    open_btn['font'] = myFont
    open_btn.grid(row=0, column=0, columnspan=1, sticky='EWNS')
    #open_btn.pack(side='left')
    
    exp_btn = tk.Button(topFrameS, text = 'Select export data path', command = export_folder, activebackground = 'white')
    exp_btn['font'] = myFont
    exp_btn.grid(row=1, column=0, columnspan=1, sticky='EWNS')
    #exp_btn.pack(side='left')
    
    run_btn = tk.Button(topFrameS, text = 'Run', command = run_fitting, activebackground = 'white')
    run_btn['font'] = myFont
    run_btn.grid(row=2, column=0, columnspan=1, sticky='EWNS')
    #run_btn.pack(side='left')
    
    close_btn = tk.Button(topFrameS, text = 'Close', command = close_window, activebackground = 'white')
    close_btn['font'] = myFont
    close_btn.grid(row=3, column=0, columnspan=1, sticky='EWNS')
    #close_btn.pack(side='bottom')
    
    secondary.protocol("WM_DELETE_WINDOW", close_window)


equipment_text = tk.Text(topFrame, height=2, width=30, font=myFont)
equipment_text.grid(row=0, column=0, columnspan=1, sticky='EWNS')
equipment_text.insert(tk.END, "Select the equipment type")

pump_btn = tk.Button(topFrame, text = 'Pump', command = open_pump_window)
pump_btn['font'] = myFont
pump_btn.grid(row=1, column=0, columnspan=1, sticky='EWNS')
#pump_btn.pack(side='left')

loadcoil_btn = tk.Button(topFrame, text = 'Load Coil', command = open_lc_window)
loadcoil_btn['font'] = myFont
loadcoil_btn.grid(row=2, column=0, columnspan=1, sticky='EWNS')
#loadcoil_btn.pack(side='left')

chiller_btn = tk.Button(topFrame, text = 'Chiller', command = open_ch_window)
chiller_btn['font'] = myFont
chiller_btn.grid(row=3, column=0, columnspan=1, sticky='EWNS')
#chiller_btn.pack(side='left')

coolingtower_btn = tk.Button(topFrame, text = 'Cooling Tower', command = open_ct_window)
coolingtower_btn['font'] = myFont
coolingtower_btn.grid(row=4, column=0, columnspan=1, sticky='EWNS')
#coolingtower_btn.pack(side='left')

close_root_btn = tk.Button(topFrame, text = 'Close', command = close_root_window, activebackground = 'white')
close_root_btn['font'] = myFont
close_root_btn.grid(row=5, column=0, columnspan=1, sticky='EWNS')
#close_root_btn.pack(side='bottom')
    
root.mainloop()
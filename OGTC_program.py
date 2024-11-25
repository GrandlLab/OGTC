#################### PACKAGES NEEDED TO RUN TENSION CLAMP ##############################
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import threading
import statistics
import time
import re
import serial
from tkinter import *
from tkinter import ttk
from pycromanager import Core
import logging
import collections
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
logging.getLogger('PIL').setLevel(logging.WARNING)# Suppress PIL debug messages


#################### VARIABLES THAT NEED DEFINING BEFORE THE PROTOCOL STARTS THAT CAN BE ALTERED ####################

volts_per_bit=0.0158 #slope of volts over bits calibration with ocilliscope; teensy to HEKA
volts_offset=-1.9171 #y intercept of volts over bits calibration with ocilliscope; teensy to HEKA

bits_per_volt=231.26 #slope of bits over volts calibration with ocilliscope; HEKA to teensy
bits_offset=500.38 #y intercept of volts over bits calibration with ocilliscope; HEKA to teensy

command_voltage = ((0-volts_offset)/volts_per_bit) # voltage program will begin at (0mmHg and 0V)


total_commanded_sweep_time=20 # Total time you want each sweep to last; dictates how much rest will be given from sweep to sweep
target_tension_flex=0.1 #setting how flexible tension measurements from target tension can be before pressure is changed
hist_limits=[559, 922] #manually determining what the upper and lower bound on the pixel historgrams
gaussian_column_num=2 #teeing the program how many columns to skip when performing gaussian fitting
file_path='' #File pacth to save images and end CSV
image_saving_frequency = 10 #saving an image of the membrane and fit every x frames


#################### FUNCTIONS THAT WILL NOT BE THEIR OWN THREAD ##############################
def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def circle_fit_convex(x, r, h, k):
    return k - np.sqrt(r**2 - (x - h)**2)

def circle_fit_concave(x, r, h, k):
    return k + np.sqrt(r**2 - (x - h)**2)

def crop_membrane(avgpix_array, cropped_image, _1gaussian, gaussian_column_num):
    for i in range(cropped_image.shape[1]):
        pix_start = 0

        #get the average pixel intensity for five pixels at a time in each column
        while pix_start + 5 <= cropped_image.shape[0]:
            pix_to_avg = np.average(cropped_image[pix_start:pix_start+5, i])
            pix_start += 5
            avgpix_array[pix_start-6, i] = pix_to_avg
            
        #find darkest pixel (center pixel of 5 pixel chunk) for each column and generate x coordinates to graph with, convert to dataframe
        y_coordinates = np.argmin(avgpix_array, axis = 0)
        mid_x_loc=int(len(y_coordinates)/2)
        
    mid_x_loc=int(len(y_coordinates)/2)

    #moving left and right until the pipette wall is hit
    left_x=mid_x_loc
    right_x=mid_x_loc

    #moving point by point in the left until direction there is a shift in thew y location that is greater that 15; characteristic of when membrane switches to pipette
    while y_coordinates[left_x]-15 <= y_coordinates[mid_x_loc] <= y_coordinates[left_x]+15:
        left_x-=1
    
    #moving point by point in the right direction until there is a shift in thew y location that is greater that 15; characteristic of when membrane switches to pipette
    while y_coordinates[right_x]-15 <= y_coordinates[mid_x_loc] <=  y_coordinates[right_x]+15:
        right_x+=1

    #cutting the original "darkest piexl" df to now only havinbg the "darkest pixels" for the membrane
    mem_pix_array=y_coordinates[left_x+1:right_x]
    
    #creating a list to add the x, y coordinates for the gaussian fit darkest pixel
    mem_gaussian_list_y=[]
    mem_gaussian_list_x=[]
    
    #creating a list to add the sigma for each gaussian fit
    sigma_list=[]
        
    for i in range(int(len(mem_pix_array)/gaussian_column_num)): 
        
        column_id=(i*gaussian_column_num)+left_x #identifiying which column you want to add a gaussian fit to
        darkest_pixel = np.argmin(avgpix_array[:, column_id]) #determining where the darkest pixel from pixel fit was located
        sliced_column=cropped_image[darkest_pixel-10:darkest_pixel+15, column_id].tolist() #slicing the origina image array to have 10 points above and below the darkest pixels
        
        # Flatten the list by taking only the first element of each nested list
        flattened_sliced_column = [pixel[0] for pixel in sliced_column]
        
        sliced_column_flipped=[abs(x-max(flattened_sliced_column)) for x in flattened_sliced_column]
        xdata=np.linspace(0, len(sliced_column_flipped), len(sliced_column_flipped)) #generating x data to be used for gaussian fit

        try:
            initial_guesses=[max(sliced_column_flipped), len(sliced_column_flipped)/2, 10]
            popt, pcov=curve_fit(_1gaussian, xdata, sliced_column_flipped, p0=initial_guesses)
            sigma_amp, sigma_cen, sigma_sigma = np.sqrt(np.diag(pcov))

            if sigma_amp < 10 or sigma_amp > 100: #triaging out fits that are incorrect and will skew overall circle fit
                continue
            else:        
                sigma_list.append(sigma_amp)
                mem_gaussian_list_y.append(popt[1]+darkest_pixel-10) #adding the centerpoint of the gaussian to the gaussian_array_y
                mem_gaussian_list_x.append(column_id) #adding the column id as x corrdinates for the darkest pixel
        except:
            continue
    
    #removing the first fit since this, many times, will fail, and skew the overall circle fit
    mem_gaussian_list_y.pop(0)
    mem_gaussian_list_x.pop(0)
    sigma_list.pop(0)
    
    return mem_gaussian_list_y, mem_gaussian_list_x, sigma_list

def determine_radii(popt):
    true_radius = popt[0] * 0.03225 #each pixel resolution/size post binning
    return true_radius

def micro_to_yolo_converter(original_array, hist_min, hist_max):
    # Remove the first dimension and leave height and width; array from micromanager has an extra dimension in the start
    flattened_array = original_array[0, :, :]

    # Clip the values to the histogram limits
    flattened_array = np.clip(flattened_array, hist_min, hist_max)

    # Rescale the array values to be normalized between 0-255; typoe of images model was trained on
    normalized_array = (flattened_array - hist_min) * (255 / (hist_max - hist_min))
    normalized_array = normalized_array.astype(np.uint8)  # Convert to uint8

    return normalized_array


def pick_circle_fit(pcov_convex, pcov_concave, popt_convex, popt_concave):
    #determining the varance to see which circle fit to use
    diagonal_convex = np.diag(pcov_convex)
    diagonal_concave = np.diag(pcov_concave)

    #if circle is convex (bent towards the cell...)
    if abs(diagonal_convex[0])<abs(diagonal_concave[0]) and abs(diagonal_convex[1])<abs(diagonal_concave[1]) and abs(diagonal_convex[2])<abs(diagonal_concave[2]):
        popt, pcov, circle_class= popt_convex, pcov_convex, 'convex'

    #if the circle is concave (bent towards pressure clamp)...
    elif abs(diagonal_convex[0])>abs(diagonal_concave[0]) and abs(diagonal_convex[1])>abs(diagonal_concave[1]) and abs(diagonal_convex[2])>abs(diagonal_concave[2]):
        popt, pcov, circle_class= popt_concave, pcov_concave, 'concave'

    #if the circle fit parameter is not working properly...    
    else:
        print('circle fitting error')
    
    return popt, pcov, circle_class


def plot_fit(circle_class, circle_fit_convex, circle_fit_concave, mem_gaussian_list_x, mem_gaussian_list_y, popt, cropped_image, fit_color, plot_text_list, file_path, count):
    #determining which fit to use for plotting
    if circle_class=='convex':
        circle_fit=circle_fit_convex
    elif circle_class=='concave':
        circle_fit=circle_fit_concave
        
    # Plot the circular fit along with the original heatmap, scatter of points used, and line graph of connected dots
    fitline_xdata = np.linspace(mem_gaussian_list_x[0], mem_gaussian_list_x[-1], num=1000)
    fitline_ydata = circle_fit(fitline_xdata, *popt)

    
    # Plot the circular fit along with the original heatmap, scatter of points used, and line graph of connected dots
    fig_plot, ax_plot = plt.subplots(figsize=(20,16))

    #plot the circular fit over the original yolo cut image
    ax_plot.plot(fitline_xdata, fitline_ydata, color=fit_color)
    ax_plot.scatter(mem_gaussian_list_x, mem_gaussian_list_y)
    ax_plot.imshow(cropped_image)

    #adding a textbox in the top right corner with relevant information for each image
    text_content = (
        f'Command Bits: {plot_text_list[0]}\n'
        f'Monitor Bits: {plot_text_list[1]}\n'
        f'Monitor Pressure: {plot_text_list[2]}\n'
        f'Target tension: {plot_text_list[3]}\n'
        f'Measured Tension: {plot_text_list[4]}\n'
        f'Instant Radius: {plot_text_list[5]}\n'
        f'Avg Radius: {plot_text_list[6]}\n'
        f'Protocol Phase: {plot_text_list[7]}\n'
        f'Membrane fit duration: {plot_text_list[8]}'
    )
    
    ax_plot.text(0, 0, text_content, fontsize=15, ha='left', va='top', color='white', bbox=dict(facecolor='black')) #adding a text box to the upper left corner
    fig_plot.savefig(file_path+'/frame_'+str(count)+'.JPG') #save the figure
    plt.close(fig_plot)  # Close the figure to release memory


##################### INITIAL GUI SETUP ############################################################

#function called to start thread zero
def begin_protocol_GUI():
    global command_voltage
    global emergency_end_protocol

    # global is_protocol_written #setting that the protocol key is written to global
    root = Tk()
    root.title('Interactive Table')
    root.geometry('600x400') #set the size of the window that initially appears
    # root.resizable(True,)

    #setting up notebooks/tabs for ramp and tension protocols
    pulse_generator=ttk.Notebook(root) #defining the window for the GUI
    pulse_generator.pack(pady=15)
    
    tension_step_notebook=Frame(pulse_generator, width=500, height=500) #setting characteristics of first notebook/tabe
    tension_step_notebook.pack(fill='both', expand=1)

    pulse_generator.add(tension_step_notebook, text='Tension Step Protocol Generator') #generating  tension step (first) notebook/tab and labeling it

    # Create a Canvas widget for the vertical line for step tab
    canvas_step = Canvas(tension_step_notebook, width=2, height=200, bg='black')
    canvas_step.grid(row=0, column=4, rowspan=6, sticky='ns')
    
    # Draw the vertical line
    canvas_step.create_line(1, 0, 1, 200, fill='black')


    #used to terminat all threads manually while saving the dataframe that was being generated
    def forced_protocol_end():
        global emergency_end_protocol
        emergency_end_protocol='yes'
        print('Forced Ending')


    ##################### TENSION STEP GUI TAB SETUP ############################################################

    #function that will save vales in the GUI to be used for commanding pressure clamp
    def save_tension_step_values():
        global values
        global is_protocol_written
        for i in range(2):
            for j in range(4):
                values[i][j] = pressure_entry_boxes[i][j].get() #need to jump column that has dividing line on it

        values.append(['step']) #add marker indicating pressure step protocol in last spot in values list
        print('Pressure step values: ', values)
        is_protocol_written='yes' #signals that the protocol you wrote is complete


    # Initialize pressure_step values list
    values = [[None]*4 for _ in range(2)]
    values.append([None])  # Adding one more list with one element initialized with None


    # Initialize pressure step entry boxes
    pressure_entry_boxes = [[None]*4 for _ in range(2)]
    pressure_entry_boxes.append([None])  # Adding one more list with one element initialized with None

    # Create and place entry boxes
    for i in range(2):
        for j in range(4):
            pressure_entry_boxes[i][j] = Entry(tension_step_notebook, width=10)
            if j!=3:
                pressure_entry_boxes[i][j].grid(row=i+1, column=j+1, padx=5, pady=5)
            if j==3:
                pressure_entry_boxes[i][j].grid(row=i+1, column=j+2, padx=5, pady=5)#needing extra step to bypass line divider on gui
               
    #define headers for rows and columns
    row_titles=['Time (ms)', 'Tension (mN/m)']
    column_titles=['Pre_tension', 'Tension', 'Post_tension', 'Step size']
    for i in range(len(row_titles)):
        label=Label(tension_step_notebook, text=row_titles[i])
        label.grid(row=i+1, column=0, padx=5, pady=5)

    for i in range(len(column_titles)):
        label=Label(tension_step_notebook, text=column_titles[i])
        if i!=3:
            label.grid(row=0, column=i+1, padx=5, pady=5)
        if i==3:
            label.grid(row=0, column=i+2, padx=5, pady=5) #needing extra step to bypass line divider on gui

    # Add labels to the right side of the last column
    label_step_size = Label(tension_step_notebook, text="Starting\n Tension (mN/m)")
    label_step_size.grid(row=1, column=6, padx=5, pady=5)

    label_step_direction = Label(tension_step_notebook, text="Step \n Size (mN/m)")
    label_step_direction.grid(row=2, column=6, padx=5, pady=5)

    # Create a button to save values
    save_button = Button(tension_step_notebook, text="Save Values", command=save_tension_step_values)
    save_button.grid(row=5, column= 1, pady=10, sticky='nsew')

    # Create a button to close program when something occurs (patch ruptures, yolo fails, etc.)
    eject_button = Button(tension_step_notebook, text="End Protocol", command=forced_protocol_end)
    eject_button.grid(row=5, column= 3, pady=10, sticky='nsew')


    ##################### START GUI ############################################################

    # Start the Tkinter event loop
    root.mainloop()

    while emergency_end_protocol!='yes' and command_voltage != 'end': #continously check to see if the protocol has ended or be manually stopped
        time.sleep(0.1) #pause to limit CPU usage

##################### CAMERA COMMUNICATION ############################################################

#function called to start thread one
def run_camera():

    global image_array #array of the imcroscope image
    global emergency_end_protocol
    
    #not letting the program run until the protocol has been written
    while is_protocol_written=='no': 
        continue
    
    #open communication with micromanager (software to control camera)
    core = Core() 

    while True: #once the protocol is started, the camera will just keep running   
        core.snap_image() #take image using microcontroller
        tagged_image = core.get_tagged_image() #retrurn pixel and metadata from image
        image_array = np.reshape(tagged_image.pix,newshape=[-1, tagged_image.tags["Height"], tagged_image.tags["Width"]]) #rehape to have a useable numpy array for the image taken

        if emergency_end_protocol=='yes': #end recording if the user manually stops the protocol
            print('run_camera thread terminated')
            break

        if command_voltage == 'end': #end recording if the pressure protocol has completed
            print('run_camera thread has ended')
            break

##################### YOLO AND TENSION CALCULATION ############################################################

#function called to start thread two
def calc_radius(circle_fit_convex, circle_fit_concave, crop_membrane, determine_radii, micro_to_yolo_converter, pick_circle_fit, _1gaussian, plot_fit):
    global is_protocol_written 
    global true_radius  
    global prepulse_difference
    global image_array
    global hist_limits
    global command_voltage
    global measured_tension
    global protocol_phase
    global monitor_pressure
    global target_tension
    global avg_radi
    global rolling_avg_radi_list
    global gaussian_column_num
    global mem_fit_time
    global image_saving_frequency
    global emergency_end_protocol

    # load yolov8 model from trained data
    model = YOLO('C:/Users/mjs164/Desktop/final_tc/train6/weights/last.pt')

    #starting count at zero; used for frame naming/numbering for saved images
    count=0

    #not letting the program run until the protocol has been written
    while is_protocol_written=='no': 
        continue
    
    time.sleep(0.5)#letting the camera take an image before the loop starts

    while True: #continously iterate through loop to acruire image and calculate tension

        radius_calc_start_time=time.time() #starting a timer to see how long radius cal takes

        #convert array from microscope image to somehting yolo can recognize
        normalized_array=micro_to_yolo_converter(image_array, hist_limits[0], hist_limits[1])
        
        # Convert to 3-channel image; take each value and repeat is three times since all images are greyscale
        formatted_array = np.repeat(normalized_array[:, :, np.newaxis], 3, axis=-1)
        
        try: #try using yolo to find membrae; if it does not detected mem, use previous bounding box since mem location should not change that much
            
            # results = model(formatted_array, verbose=False,  device='cuda') #verbose stops yolov8 from printing every loop
            results = model(formatted_array, verbose=False) #verbose stops yolov8 from printing every loop
                        
        except: #assinging the previous bounding box the new image since the membrane should remain relativly stable from image to image
            print('Yolo did not find the membrane. Using previous boudning box coordinates') #since previous xyxys is never updated, it will automatically be used in the next loop
        
        boxes = results[0].boxes.cpu().numpy() #take the bounding box info from GPU and converts it to be useable in the CPU
    
        xyxys=boxes.xyxy #Pulls the bounding box informaiton from the boxes object

        for xyxy in xyxys: #since there is only one label, xyxys will only have one xyxy which is the membrane
            cropped_image=formatted_array[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] #cropping the array from the yolo bounding box; int since xyxys are float
            
            #making an array of repeating large number. This way later argmin won't pick up zeros like it would with np.zero. 1000000 is arbitrary
            avgpix_array = np.full((cropped_image.shape[0], cropped_image.shape[1]), 10000)

            #generating the x and y lists for the membrane locations based on gaussian fitting
            mem_gaussian_list_y, mem_gaussian_list_x, sigma_list=crop_membrane(avgpix_array, cropped_image, _1gaussian, gaussian_column_num)
        
        try:
            # determining difference in membrane/pipette wall position and the center of the membrane position; used for prepulse
            pipette_wall_center=(mem_gaussian_list_y[0]+mem_gaussian_list_y[-1])/2 #taking the average y position
            central_membrane_point=mem_gaussian_list_y[round(len(mem_gaussian_list_y)/2)]
            prepulse_difference=central_membrane_point-pipette_wall_center
        
        except: #when the prepulse fails, use previous value and do not shut down loop
            print('prepulse fail, using previous fit')

        # Guess initial values for the circle fit parameters
        r_guess = 100
        h_guess = 80
        k_guess = 20
        guess_para = [r_guess, h_guess, k_guess]
    
        try: #try to fit the membrane with both circle fits

            # Perform curve with both concave and convex fitting
            popt_concave, pcov_concave = curve_fit(circle_fit_concave, mem_gaussian_list_x, mem_gaussian_list_y,  guess_para, sigma=sigma_list, maxfev=8000)

            popt_convex, pcov_convex = curve_fit(circle_fit_convex, mem_gaussian_list_x, mem_gaussian_list_y, guess_para, sigma=sigma_list, maxfev=8000)

            #function to determine which curcle fit to use
            popt, pcov, circle_class=pick_circle_fit(pcov_convex, pcov_concave, popt_convex, popt_concave)
            
            #calculate the radius from the current fit
            new_radius=determine_radii(popt) 

            if protocol_phase=='Pressure step': #only relevent for pressure step since prepulse radii will be extremely large
                    true_radius=new_radius #calculate the radius to be sent to PID control
                    rolling_avg_radi_list.append(true_radius) #adding the most recent radi fit to the rolling average radi list
                    avg_radi=statistics.mean(list(rolling_avg_radi_list)) #calculating the average radi for five previous successful measurements
                    fit_color='red'
            else:
                true_radius=new_radius #calculate the radius to be sent to PID control
                fit_color='red'
                
        except: #if the membrane fits do not work, use the previous radius fit
            print('Membrane radius fit was not updated. Previousl radius fit used') #since the popt would not be updated, not other command is needed
            fit_color='blue'
    
        mem_fit_time=time.time()-radius_calc_start_time #calculation how long each fit took
        
        #saving the plotted image every x number of frames
        if count%image_saving_frequency==0:

            #making a list of text to be shown on the plot rather than importing every variable into the plot_fit function
            plot_text_list=[command_voltage, int(monitor_bits), monitor_pressure, target_tension, measured_tension, true_radius, avg_radi, protocol_phase, mem_fit_time]
            
            #function to plot the fit. No variable is returned since the plot is saved within the plot_fit function
            plot_fit(circle_class, circle_fit_convex, circle_fit_concave, mem_gaussian_list_x, mem_gaussian_list_y, popt, cropped_image, fit_color, plot_text_list, file_path, count)

        #increase count by one for saving frames
        count+=1
        
        # Set the event to signal that radius has been updated
        new_radius_calculated_updated_event.set()

        if emergency_end_protocol=='yes': #end recording if the user manually stops the protocol
            print('calc_radius thread terminated')
            break

        if command_voltage == 'end': #end recording if the pressure protocol has completed
            print('calc_radius thread has ended')
            break

##################### START PRESSURE PROTOCOL ############################################################

# Define the run_pressure_protocol function to run in thread two
def run_pressure_protocol():
    global command_voltage
    global record_data
    global is_protocol_written
    global true_radius
    global monitor_bits
    global measured_tension
    global protocol_phase
    global monitor_pressure
    global target_tension
    global sweep_start_time
    global volts_offset
    global volts_per_bit
    global avg_radi
    global target_tension_flex
    global rolling_avg_radi_list
    global on_off_signal
    global total_commanded_sweep_time
    global bits_offset
    global bits_per_volt
    global emergency_end_protocol
    
    while is_protocol_written=='no': #not letting the program run until the protocol has been written
        continue
    
    #running protocol if the command is for a pressure step
    if values[-1][0] == 'step': #begin protocol if key from values list == tension
       
        #Assingning variables from the values list to use for tension step protocol
        prepressure_duration=float(values[0][0])/1000
        pressure_duration=float(values[0][1])/1000
        post_pressure_duraiton=float(values[0][2])/1000
        
        prepulse_tension=float(values[1][0])
        tension_stimuli=int(values[1][1])
        post_tension=float(values[1][2])
        starting_tension=int(values[0][3])
        step_size=int(values[1][3])
        
        #go through each commanded tension
        for i in range(starting_tension, tension_stimuli, step_size):
            sweep_start_time = time.time() #start timing to know how long the entire sweep took
            record_data = 'yes' #start recording data (activate teensy recording code) because the protocol has started
            target_tension=i

#############PREPULSE PROTOCOL
            protocol_phase='Prepulse'
            on_off_signal=0
            command_voltage = int((float(2*0.02)-volts_offset)/volts_per_bit) #initial prepulse step is +2mmHg
            prepulse_start_time=time.time() #setting the start of the prepulse time

            while (time.time() - prepulse_start_time) < prepressure_duration:

                # Only procede is a new radius has been calculated 
                if new_radius_calculated_updated_event.is_set():

                    try: #only change command if a new radius is calculated
                        if prepulse_difference>0: #if the membrane is curved toward the pressure clamp (under corrected)
                            command_voltage+=1 #increasing pressure set by pressure clamp by one bit
                        if prepulse_difference<0: #if the membrane is curved toward the cell (over corrected)
                            command_voltage-=1 #decrease pressure set by pressure clamp by one bit
                    except:
                        print('Prepulse not changed, original fitting did not provide new position values')

                    # Reset the event to wait for the next update of the radius
                    new_radius_calculated_updated_event.clear()

                else: #if a new radius value has not come in, wait 5 ms and check again
                    time.sleep(0.005) 
                
                if emergency_end_protocol=='yes': #end recording if the user manually stops the protocol
                    print('run_pressure_protocol thread terminated')
                    break


#############PRESSURE PROTOCOL
            protocol_phase='Pressure step'
            pressure_start_time=time.time() #setting the start of the pressure time
    
            
            first_pressure_jump= -7.83*target_tension+3.56 #setting the initial pressure value to get to  

            command_voltage=int(((first_pressure_jump*0.02)-volts_offset)/volts_per_bit) #jumping to the first approximate tension and setting working command voltage
            on_off_signal=1 #telling HEKA that a pressure pulse has begun

            time.sleep(0.05) #pausing for 50ms to let the first pressure step achieve the correct pressure and for the membrane to flex to its desired shape
           
            # Reset the event to wait for the next update of the radius
            new_radius_calculated_updated_event.clear()

            target_tension_achieved=False #setting a variable that remans off until the target tension has been achieved
            
            while (time.time() - pressure_start_time) < pressure_duration: #continue to loop for commanded pressure duration

                # Only procede is a new radius has been calculated 
                if new_radius_calculated_updated_event.is_set():

                    #calculating the tension that is currently at the membrane
                    monitor_pressure= ((monitor_bits-bits_offset)/bits_per_volt)*(10/0.2) # Converting volts from HSPC to pressure based on voltage calibration
                    
                    #mmHg to F/m^2 conversion factor = 133.322
                    measured_tension = ((-1*monitor_pressure* 133.322)*avg_radi*10**-6)/2*10**3 #LaPlace's equation
                    print('measured tension: ', measured_tension)
                    
                    #cycling through to get measured tension to equal the comanded tension; addition of 1 is one bit (0-255 for teensy board)
                    if measured_tension < (target_tension-target_tension_flex): #Adding target_tension_flex to prevent oscillations. Allowing for an acceptable tension window
                        command_voltage -= 1
                    elif measured_tension > (target_tension+target_tension_flex):#Adding target_tension_flex to prevent oscillations. Allowing for an acceptable tension window.
                        command_voltage += 1
                    elif measured_tension >= target_tension: #When the measured tension finally reaches the target tension, make the target_tension_achieved True
                        target_tension_achieved==True
                    elif target_tension_achieved==False: #target tension has not been reached and if not corrected, will hover below target tension and in the tension error window
                        command_voltage -= 1 #keep applying negative pressure until the target tension is acheived, reguardless of the tension flex window

                    # Reset the event to wait for the next update of the radius
                    new_radius_calculated_updated_event.clear()
                
                else: #if a new radius value has not come in, wait 1 ms and check again
                    time.sleep(0.001)
                
                if emergency_end_protocol=='yes': #end recording if the user manually stops the protocol
                    print('run_pressure_protocol thread terminated')
                    break

#############POSTPRESSURE PROTOCOL
            protocol_phase='Post Pressure'
            postpressure_start_time=time.time() #setting the start of the pressure time
            command_voltage = int((float(0*0.02)-volts_offset)/volts_per_bit) 
            monitor_pressure= ((monitor_bits-bits_offset)/bits_per_volt)*(10/0.2) # Converting volts from HSPC to pressure based on voltage calibration
            measured_tension=None
            on_off_signal=0 #telling HAKE the pressure pulse is over
            while time.time() - postpressure_start_time < post_pressure_duraiton: #continue to loop for commanded postpressure duration
                monitor_pressure= ((monitor_bits-bits_offset)/bits_per_volt)*(10/0.2) # Converting volts from HSPC to pressure based on voltage calibration
                continue
            if emergency_end_protocol=='yes': #end recording if the user manually stops the protocol
                    print('run_pressure_protocol thread terminated')
                    break

            record_data = 'no' #tell teensy to stop recording data for that sweep
            rolling_avg_radi_list=collections.deque([1], 5) #clearing the radii list for next pressure stimulus
            time.sleep(total_commanded_sweep_time - (time.time() - sweep_start_time)) #Holding the sweep for the remainder of the total time commanded
            
            
            print('Sweep completed: ', i)

        command_voltage = 'end'
        
        print('run_pressure_protocol thread has ended')

##################### START ARDUINO COMMUNICATION AND DATA ACQUISITION ############################################################

# Define the run_teensy_communication function to run in thread three
def run_teensy_communication():
    global sweep_start_time
    global start_time
    global is_protocol_written
    global monitor_bits
    global monitor_pressure
    global target_tension
    global true_radius
    global protocol_phase
    global command_voltage
    global avg_radi
    global emergency_end_protocol
    global on_off_signal
    global mem_fit_time
    global emergency_end_protocol           

    #lists for plotting; all get added at each teensy recording loop
    monitor_list = []  # Monitor pressure in bits from teensy
    command_list = []  # Command pressure in bits to teensy
    monitor_pressure_list = [] #Monitoring pressure based on conversion values; need to add calibration
    target_tension_list = [] #Monitoring the target tension being commanded
    measured_tension_list = [] #Monitoring what the calculated tension is based on the radius fit and pressure
    instant_radius_list = [] #Monitoring the recorded radius fits
    avg_radius_list=[] #Monitoring the average radius fits; averages the most recent 10 recorded radi
    protocol_phase_list = [] #Monitoring what phase of the sweep the protocol is in
    time_list = [] #Monitoring how far into the protocol the patch is
    sweep_time_list = [] #Monitoring the time that has passed during each sweep
    voltage_sent_to_heka_list = [] #Monitoring if an "on" or "off" signal is sent to HEKA
    mem_fit_time_list = [] #monitoring how much time each membrane fit (calc_radius) loop takes
        

    while is_protocol_written=='no': #not letting the program run until the protocol has been written
        continue

    ser = serial.Serial('COM3', 115200)
    start_time=time.time() #when the recording/protocol began

    while True:

        # Sending command voltage and on'off signal for HEKA to teensy
        command = str(command_voltage)
        teensy_message=f"{command_voltage},{on_off_signal}\n"
        ser.write(teensy_message.encode())

        #Receiving informaiton from teensy board
        line = ser.readline().decode().strip()
        monitor_bits=float(line)

        if record_data == 'yes':
            
            #adding variables to each list
            monitor_list.append(monitor_bits)
            command_list.append(command)
            time_list.append(time.time()-start_time)
            monitor_pressure_list.append(monitor_pressure) 
            target_tension_list.append(target_tension)
            measured_tension_list.append(measured_tension)
            instant_radius_list.append(true_radius)
            avg_radius_list.append(avg_radi)
            protocol_phase_list.append(protocol_phase)
            sweep_time_list.append(time.time()-sweep_start_time)
            voltage_sent_to_heka_list.append(on_off_signal)
            mem_fit_time_list.append(mem_fit_time)

        if record_data == 'no':
            sweep_start_time = 0

        if command_voltage == 'end': #end recording if the pressure protocol has completed
            print('Completed!')
            break

        if emergency_end_protocol=='yes': #end recording if the user manually stops the protocol
            print('run_teensy_communication thread terminated')
            break

    #adding all of the list data into a dataframe which can be exported
    df_protocol_data=pd.DataFrame({'monitor_bits': monitor_list,
                                   'command_bits': command_list,
                                   'monitor_pressure': monitor_pressure_list,
                                   'target_tension': target_tension_list,
                                   'measured_tension': measured_tension_list,
                                   'instant_radius': instant_radius_list,
                                   'avg_radius': avg_radius_list,
                                   'protocol_phase': protocol_phase_list,
                                   'time': time_list,
                                   'sweep_time':sweep_time_list,
                                   'on_off_heka':voltage_sent_to_heka_list,
                                   'mem_fit_time':mem_fit_time_list})
    
    #creating a dataframe of the protocol data
    df_protocol_data.to_csv(file_path+'/protocol_data.csv', index=False)

    print('run_teensy_communication thread has ended')


#################### VARIABLES THAT NEED DEFINING BEFORE THE PROTOCL STARTS THAT ARE NOT ALTERED ####################

true_radius=None #setting the radius value to none
prepulse_difference=None #setting the initial prepulse difference to none
monitor_bits=None #setting the initial monitor bits to zero
image_array=None #setting the image array variable to None
measured_tension=None #setting the original measured tension variable to None
protocol_phase=None #identification for what step in the protocol each sweep is in
monitor_pressure=None #setting the initial monitor pressure to None
target_tension=None #setting the target tension to None
sweep_start_time = None #initializing start time to make sure each sweep runs the same time
avg_radi=1 #setting to one so first frame has a number to average if yolo/fitting does not work 
emergency_end_protocol=None #setting the emergency_end_protocol to None
mem_fit_time=None #signifies how much time each membrane fit takes; this includes yolo and the radius fitting
new_radius_calculated_updated_event = threading.Event() #Event to signal that a new images has been taken and radius calculated has been updated
values = [[None]*4 for _ in range(2)] #initializing the values list needed for the protocol GUI
is_protocol_written='no'#preventing the data acquisition from starting until protocols are submitted
record_data = 'no' #only recording data from teensy when the pressure protocol is running (not during times between sweep)
rolling_avg_radi_list=collections.deque([1], 5) #creating a list to determine the rolling average radi. Will hold the five most previously measured radi
on_off_signal=0 #this signal is used to derermine what voltage to send to HEKA to timestamp pressure stimuli; 0 means off, 1 means on


##################### THREADS TO RUN PROTOCOLS ############################################################

# Create threads for each function
thread_zero=threading.Thread(target=begin_protocol_GUI)
thread_one=threading.Thread(target=run_camera)
thread_two = threading.Thread(target=calc_radius, args=(circle_fit_convex, circle_fit_concave, crop_membrane, 
                                                        determine_radii, micro_to_yolo_converter, pick_circle_fit, _1gaussian, plot_fit)) 
thread_three = threading.Thread(target=run_pressure_protocol)
thread_four = threading.Thread(target=run_teensy_communication)


# Start both threads
thread_zero.start()
thread_one.start()
thread_two.start()
thread_three.start()
thread_four.start()

# Wait for all threads to complete
thread_zero.join()
thread_one.join()
thread_two.join()
thread_three.join()
thread_four.join()


